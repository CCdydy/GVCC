"""
diagnose_gop_decay.py — Analyze per-frame quality decay within a single GOP

Runs one GOP and logs per-frame diagnostics at each SDE step:
  - Per-frame residual MSE (x0_true - x0_hat)
  - Per-frame codebook correlation (how well atoms match residual)
  - ODE-only baseline (no codebook) vs DDCM

This identifies whether tail decay is due to:
  (a) Model prediction degradation (residual grows for later frames)
  (b) Codebook inefficiency (atoms can't match residual for later frames)
  (c) SDE noise accumulation
"""

import sys, os, time, json, gc
import torch
import numpy as np
from pathlib import Path
from PIL import Image

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from sde_rf_wan import WanWrapper
from sde_rf_wan.turbo_pipeline import TurboDDCMWanPipeline
from sde_rf_wan.sde_convert import velocity_to_score, diffusion_coeff, sde_drift, shifted_timesteps
from sde_rf_wan.turbo_codebook import TurboPerFrameCodebook
from uvg_data import find_uvg_sequence as find_uvg_sequence_shared

import re, cv2


def load_yuv420_frames(yuv_path, num_frames, start_frame=0):
    match = re.search(r'(\d+)x(\d+)', os.path.basename(yuv_path))
    W, H = int(match.group(1)), int(match.group(2))
    frame_size = H * W * 3 // 2
    frames = []
    with open(yuv_path, 'rb') as f:
        f.seek(start_frame * frame_size)
        for _ in range(num_frames):
            raw = f.read(frame_size)
            if len(raw) < frame_size:
                break
            yuv = np.frombuffer(raw, dtype=np.uint8)
            y = yuv[:H * W].reshape(H, W)
            u = yuv[H * W:H * W + H * W // 4].reshape(H // 2, W // 2)
            v = yuv[H * W + H * W // 4:].reshape(H // 2, W // 2)
            u = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(np.stack([y, u, v], axis=-1), cv2.COLOR_YUV2RGB)
            frames.append(Image.fromarray(rgb))
    return frames


def frames_to_tensor(frames):
    return torch.stack([
        torch.from_numpy(np.array(f).astype(np.float32) / 255.0).permute(2, 0, 1)
        for f in frames
    ])


def compute_per_frame_psnr(orig, recon):
    mse = ((orig - recon) ** 2).mean(dim=[1, 2, 3])
    psnr = -10.0 * torch.log10(mse + 1e-10)
    return psnr.tolist()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose GOP tail decay")
    parser.add_argument("--data_dir", default=os.path.join(_project_root, "data", "uvg"))
    parser.add_argument("--wan_ckpt", default="./Wan2.1-I2V-14B-720P")
    parser.add_argument("--sequence", default="Beauty")
    parser.add_argument("--num_frames", type=int, default=21)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--K", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim_tail", type=int, default=3)
    parser.add_argument("--g_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./diagnose_output")
    args = parser.parse_args()

    H, W = args.height, args.width
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Find sequence
    yuv_path = find_uvg_sequence_shared(args.data_dir, args.sequence)
    assert yuv_path, f"Sequence {args.sequence} not found"

    # Load frames
    raw = load_yuv420_frames(yuv_path, args.num_frames)
    frames = [f.resize((W, H), Image.LANCZOS) for f in raw]
    ref_image = frames[0]  # GT ref (free)
    print(f"Loaded {len(frames)} frames of {args.sequence}, {W}x{H}")

    # Load model
    flow_shift = 3.0 if H <= 480 else 5.0
    model = WanWrapper(args.wan_ckpt, config_name="i2v-14B", flow_shift=flow_shift)
    model.load("cuda", torch.bfloat16)

    # =============================================
    # Test 1: ODE baseline (no codebook, no bits)
    # =============================================
    # =============================================
    # DDCM with per-frame residual logging
    # =============================================
    print("\n" + "="*60)
    print(f"DDCM M={args.M}, per-frame residual analysis")
    print("="*60)

    pipe = TurboDDCMWanPipeline(
        model, K=args.K, M=args.M,
        num_steps=args.steps, num_ddim_tail=args.ddim_tail,
        guidance_scale=1.0, g_scale=args.g_scale,
        num_frames=args.num_frames, height=H, width=W, seed=args.seed,
    )

    # Encode with detailed logging — offload DiT during VAE ops
    embeds = model.encode_prompt("")

    # Must offload DiT for VAE ops (14B model fills most of 48GB)
    model.model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    i2v_cond = model.encode_image(ref_image, args.num_frames, H, W)
    x0_true = model.encode_video(frames, H, W)

    # Bring DiT back
    model.model.to("cuda")
    torch.cuda.empty_cache()

    F_lat = pipe.num_latent_frames

    def model_fn(x_t, t):
        return model.predict_velocity_cfg(x_t, t, embeds, 1.0, i2v_cond)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    x_t = torch.randn(1, *pipe.latent_shape, generator=gen).to("cuda")

    timesteps = shifted_timesteps(args.steps, shift=flow_shift, device="cuda")
    num_sde = pipe.num_sde_steps

    per_step_per_frame_residual = []

    sde_idx = 0
    for i in range(args.steps):
        t_curr = timesteps[i].item()
        t_next = timesteps[i + 1].item()
        delta_t = t_curr - t_next

        v_t = model_fn(x_t, t_curr)

        if t_next < 1e-6:
            x_t = x_t - v_t * delta_t
            break

        # MMSE estimate
        x0_hat = x_t - t_curr * v_t

        # Per-frame residual MSE
        residual = (x0_true - x0_hat).squeeze(0)  # (C, F, H, W)
        per_frame_mse = []
        for f in range(F_lat):
            mse_f = (residual[:, f] ** 2).mean().item()
            per_frame_mse.append(mse_f)

        if i < num_sde:
            # SDE step with codebook
            score = velocity_to_score(v_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, args.g_scale)
            f_t = sde_drift(v_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            z_full = torch.zeros_like(residual)
            per_frame_corr = []
            for f in range(F_lat):
                idx, sgn, z_f = pipe.codebook.select_atoms(residual[:, f], sde_idx, f)
                z_full[:, f] = z_f
                # Correlation: how well codebook noise matches residual direction
                r_flat = residual[:, f].flatten()
                z_flat = z_f.flatten()
                corr = torch.dot(r_flat, z_flat) / (r_flat.norm() * z_flat.norm() + 1e-10)
                per_frame_corr.append(corr.item())

            x_t = x_t - f_t * delta_t + noise_coeff * z_full.unsqueeze(0)
            sde_idx += 1

            per_step_per_frame_residual.append({
                "step": i, "t": round(t_curr, 4),
                "per_frame_mse": [round(m, 6) for m in per_frame_mse],
                "per_frame_corr": [round(c, 4) for c in per_frame_corr],
            })

            if i % 4 == 0 or i == num_sde - 1:
                print(f"\n  Step {i+1}/{args.steps} (t={t_curr:.3f}):")
                print(f"    Latent frame MSE: {' '.join(f'{m:.4f}' for m in per_frame_mse)}")
                print(f"    Codebook corr:    {' '.join(f'{c:.4f}' for c in per_frame_corr)}")
        else:
            # ODE tail
            x_t = x_t - v_t * delta_t

    # Decode
    model.model.cpu(); gc.collect(); torch.cuda.empty_cache()
    x_t_cpu = x_t.cpu()
    x_t = x_t_cpu.to("cuda")
    ddcm_frames = model.decode_latent(x_t)
    model.model.to("cuda"); torch.cuda.empty_cache()

    t_gt = frames_to_tensor(frames)
    t_ddcm = frames_to_tensor(ddcm_frames[:len(frames)])
    ddcm_psnr = compute_per_frame_psnr(t_gt, t_ddcm)

    # =============================================
    # Summary
    # =============================================
    print("\n" + "="*60)
    print("SUMMARY: Per-frame PSNR")
    print("="*60)
    print(f"{'Frame':>5} | {'DDCM M={}'.format(args.M):>12}")
    print("-" * 25)
    for i in range(len(frames)):
        print(f"  {i:3d} | {ddcm_psnr[i]:10.2f} dB")
    print("-" * 25)
    print(f"  Avg | {np.mean(ddcm_psnr):10.2f} dB")
    print(f" Last | {ddcm_psnr[-1]:10.2f} dB")

    # Latent frame analysis
    print(f"\nLatent frame residual MSE at final SDE step:")
    last = per_step_per_frame_residual[-1]
    for f in range(F_lat):
        print(f"  Latent frame {f}: MSE={last['per_frame_mse'][f]:.6f}, corr={last['per_frame_corr'][f]:.4f}")

    # Save
    results = {
        "sequence": args.sequence,
        "config": {"M": args.M, "K": args.K, "steps": args.steps,
                   "ddim_tail": args.ddim_tail, "g_scale": args.g_scale,
                   "num_frames": args.num_frames},
        "ddcm_per_frame_psnr": [round(p, 2) for p in ddcm_psnr],
        "per_step_residuals": per_step_per_frame_residual,
    }
    with open(out / "diagnosis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}/diagnosis.json")


if __name__ == "__main__":
    main()
