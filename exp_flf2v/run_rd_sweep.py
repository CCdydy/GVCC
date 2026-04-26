"""
run_rd_sweep.py — FLF2V Rate-Distortion Sweep (M × K joint)

Sweeps (M, K) combinations on UVG 720p (3 GOPs) to trace an R-D curve.
Bits per atom = ceil(log2(K)) + 1 (index + sign).
Model loaded once, all configs share the same decoded boundary frames.

Usage:
    python run_rd_sweep.py --wan_ckpt ./Wan2.1-FLF2V-14B-720P
"""

import sys
import os
import re
import math
import time
import json
import gc
import csv
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from sde_rf_wan.wan_flf2v_wrapper import WanFLF2VWrapper
from sde_rf_wan.turbo_pipeline import TurboDDCMWanPipeline
from sde_rf_wan.sde_convert import velocity_to_score, diffusion_coeff, sde_drift
from sde_rf_wan.ref_codec import compress_ref
from uvg_data import find_uvg_sequences as find_uvg_sequences_shared


# ==================================================================
# UVG Loading (same as run_flf2v_experiment.py)
# ==================================================================

def load_yuv420_frames(yuv_path, num_frames, start_frame=0):
    import cv2
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
            yuv_img = np.stack([y, u, v], axis=-1)
            rgb = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
            frames.append(Image.fromarray(rgb))
    return frames


def find_uvg_sequences(data_dir):
    return find_uvg_sequences_shared(data_dir)


def resize_frames(frames, target_w, target_h):
    return [f.resize((target_w, target_h), Image.LANCZOS) for f in frames]


def compress_boundary_frame(image, ref_codec="compressai", ref_quality=4):
    decoded, _, nbytes = compress_ref(image, codec=ref_codec, quality=ref_quality)
    return decoded, nbytes


# ==================================================================
# FLF2V encode/decode (copied from run_flf2v_experiment.py)
# ==================================================================

def flf2v_encode(pipe, model, gop_frames, flf2v_cond, height, width):
    embeds = model.encode_prompt("")
    model.model.cpu()
    torch.cuda.empty_cache()
    x0_true = model.encode_video(gop_frames, height, width)
    model.model.to(model.device)
    torch.cuda.empty_cache()

    model_fn = pipe._model_fn(embeds, flf2v_cond)
    gen = torch.Generator(device="cpu").manual_seed(pipe.seed)
    x_t = torch.randn(1, *pipe.latent_shape, generator=gen).to(pipe.device)
    step_data = []
    sde_idx = 0

    for i in range(pipe.num_steps):
        t_curr = pipe.timesteps[i].item()
        t_next = pipe.timesteps[i + 1].item()
        delta_t = t_curr - t_next
        u_t = model_fn(x_t, t_curr)

        if t_next < 1e-6:
            x_t = x_t - u_t * delta_t
            break
        if i >= pipe.num_sde_steps:
            x_t = x_t - u_t * delta_t
            continue

        x0_hat = x_t - t_curr * u_t
        residual = (x0_true - x0_hat).squeeze(0)
        score = velocity_to_score(u_t, x_t, t_curr)
        g_t = diffusion_coeff(t_curr, pipe.g_scale)
        f_t = sde_drift(u_t, score, g_t)
        noise_coeff = g_t * (delta_t ** 0.5)

        frame_entries = []
        noise_frames = []
        for f in range(pipe.num_latent_frames):
            r_f = residual[:, f, :, :]
            M_f = pipe._get_M_for_frame(f)
            idx, sgn, z_f = pipe.codebook.select_atoms(r_f, sde_idx, f, M_override=M_f)
            frame_entries.append((idx, sgn))
            noise_frames.append(z_f)

        step_data.append(frame_entries)
        noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)
        x_t = x_t - f_t * delta_t + noise_coeff * noise_3d
        sde_idx += 1

    return step_data, x0_true


def flf2v_decode(pipe, model, step_data, flf2v_cond):
    embeds = model.encode_prompt("")
    model.model.to(model.device)
    torch.cuda.empty_cache()
    model_fn = pipe._model_fn(embeds, flf2v_cond)

    gen = torch.Generator(device="cpu").manual_seed(pipe.seed)
    x_t = torch.randn(1, *pipe.latent_shape, generator=gen).to(pipe.device)
    sde_idx = 0

    for i in range(pipe.num_steps):
        t_curr = pipe.timesteps[i].item()
        t_next = pipe.timesteps[i + 1].item()
        delta_t = t_curr - t_next
        u_t = model_fn(x_t, t_curr)

        if t_next < 1e-6:
            x_t = x_t - u_t * delta_t
            break
        if i >= pipe.num_sde_steps:
            x_t = x_t - u_t * delta_t
            continue

        score = velocity_to_score(u_t, x_t, t_curr)
        g_t = diffusion_coeff(t_curr, pipe.g_scale)
        f_t = sde_drift(u_t, score, g_t)
        noise_coeff = g_t * (delta_t ** 0.5)

        noise_frames = []
        for f in range(pipe.num_latent_frames):
            idx, sgn = step_data[sde_idx][f]
            z_f = pipe.codebook.reconstruct(idx, sgn, sde_idx, f)
            noise_frames.append(z_f)

        noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)
        x_t = x_t - f_t * delta_t + noise_coeff * noise_3d
        sde_idx += 1

    x_t_cpu = x_t.cpu()
    del x_t, model_fn, embeds
    model.model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    x_t = x_t_cpu.to(model.device)
    del x_t_cpu
    frames_recon = model.decode_latent(x_t)
    model.model.to(model.device)
    torch.cuda.empty_cache()
    return frames_recon


# ==================================================================
# Metrics
# ==================================================================

def frames_to_tensor(frames):
    return torch.stack([
        torch.from_numpy(np.array(f).astype(np.float32) / 255.0).permute(2, 0, 1)
        for f in frames
    ])


def compute_psnr(orig, recon):
    mse = ((orig - recon) ** 2).mean(dim=[1, 2, 3])
    psnr = -10.0 * torch.log10(mse + 1e-10)
    return psnr.mean().item()


def compute_msssim(orig, recon):
    try:
        from pytorch_msssim import ms_ssim
        vals = []
        for i in range(0, orig.shape[0], 4):
            v = ms_ssim(orig[i:i+4], recon[i:i+4], data_range=1.0, size_average=False)
            vals.extend(v.cpu().tolist())
        return sum(vals) / len(vals)
    except ImportError:
        return None


def compute_lpips(orig, recon, device="cuda"):
    import lpips
    loss_fn = lpips.LPIPS(net='alex').to(device)
    orig_lp = (2.0 * orig - 1.0).to(device)
    recon_lp = (2.0 * recon - 1.0).to(device)
    vals = []
    for i in range(0, orig_lp.shape[0], 4):
        with torch.no_grad():
            d = loss_fn(orig_lp[i:i+4], recon_lp[i:i+4])
        vals.extend(d.flatten().cpu().tolist())
    del loss_fn
    torch.cuda.empty_cache()
    return sum(vals) / len(vals)


# ==================================================================
# Main
# ==================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FLF2V R-D Sweep (M x K)")
    parser.add_argument("--data_dir", default=os.path.join(_project_root, "data", "uvg"))
    parser.add_argument("--wan_ckpt", default="./Wan2.1-FLF2V-14B-720P")
    parser.add_argument("--output_dir", default="./rd_sweep_720p")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_gops", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim_tail", type=int, default=3)
    parser.add_argument("--g_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_codec", default="compressai")
    parser.add_argument("--ref_quality", type=int, default=4)
    parser.add_argument("--flow_shift", type=float, default=None)
    parser.add_argument("--sequences", nargs="*", default=None)
    args = parser.parse_args()

    if args.flow_shift is None:
        args.flow_shift = 3.0 if args.height <= 480 else 5.0

    HEIGHT, WIDTH = args.height, args.width
    FPG = 33
    frames_per_gop_excl_first = FPG - 1
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # R-D operating points: proportional joint scaling
    # Parameters: M (atoms), K (codebook size), ref_quality (boundary)
    # All scaled together from low→high bitrate
    # ================================================================
    # Level:  M    K       ref_q   description
    operating_points = [
        (8,    256,    1),   # ultra-low
        (16,   512,    1),   # very low
        (16,  1024,    2),   # low
        (32,  2048,    2),   # low-mid
        (32,  4096,    3),   # mid-low
        (64,  4096,    3),   # mid
        (64, 16384,    4),   # baseline
        (128,16384,    4),   # mid-high
        (128,65536,    5),   # high
        (256,65536,    5),   # very high
        (256,65536,    6),   # ultra-high
    ]

    # Preview BPP estimates
    T_sde = args.steps - args.ddim_tail  # 17
    F_lat = (FPG - 1) // 4 + 1  # 9
    total_pixels = FPG * HEIGHT * WIDTH * 3

    print("=" * 70)
    print(f"FLF2V R-D Sweep (proportional scaling) — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Resolution: {WIDTH}x{HEIGHT}, {args.num_gops} GOPs")
    print(f"  T_sde={T_sde}, F_lat={F_lat}, steps={args.steps}")
    print(f"\n  Operating points (M, K, ref_quality jointly scaled):")
    print(f"  {'M':>4} {'K':>6} {'ref_q':>6} {'bit/atom':>9} {'CB bits':>10} {'CB BPP':>10}")
    print("  " + "-" * 52)
    for M, K, rq in operating_points:
        bpa = math.ceil(math.log2(K)) + 1
        cb_bits = T_sde * F_lat * M * bpa
        cb_bpp = cb_bits / total_pixels
        print(f"  {M:>4} {K:>6} {rq:>6} {bpa:>9} {cb_bits:>10} {cb_bpp:>10.6f}")
    print("=" * 70)

    # ================================================================
    # Load model & sequences
    # ================================================================
    all_seqs = find_uvg_sequences(args.data_dir)
    if args.sequences:
        all_seqs = [(n, p) for n, p in all_seqs if n in args.sequences]
    print(f"\nSequences: {[s[0] for s in all_seqs]}")

    model = WanFLF2VWrapper(
        args.wan_ckpt, config_name="flf2v-14B", flow_shift=args.flow_shift
    )
    model.load("cuda", torch.bfloat16)

    all_results = []

    for M, K, ref_q in operating_points:
        tag = f"M{M}_K{K}_q{ref_q}"
        bpa = math.ceil(math.log2(K)) + 1
        print(f"\n{'='*70}")
        print(f"  Config: M={M}, K={K} ({bpa} bits/atom), ref_quality={ref_q}")
        print(f"  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}")

        seq_psnrs = []
        seq_msssims = []
        seq_lpips_vals = []
        seq_bpps = []
        seq_kbps = []

        for seq_idx, (seq_name, yuv_path) in enumerate(all_seqs):
            print(f"\n  [{seq_idx+1}/{len(all_seqs)}] {seq_name}")

            # Load frames
            total_unique = frames_per_gop_excl_first * args.num_gops + 1
            raw_frames = load_yuv420_frames(yuv_path, total_unique)
            frames_resized = resize_frames(raw_frames, WIDTH, HEIGHT)
            del raw_frames

            actual_gops = min(args.num_gops,
                              max(1, (len(frames_resized) - 1) // frames_per_gop_excl_first))
            actual_total = frames_per_gop_excl_first * actual_gops + 1
            frames_resized = frames_resized[:actual_total]

            # Build GOPs
            gops_gt = []
            for g in range(actual_gops):
                start = g * frames_per_gop_excl_first
                end = start + FPG
                gops_gt.append(frames_resized[start:end])

            # Compress boundary frames with this config's ref_quality
            boundary_indices = [g * frames_per_gop_excl_first for g in range(actual_gops + 1)]
            boundary_indices = [min(idx, len(frames_resized) - 1) for idx in boundary_indices]

            boundary_compressed = {}
            for idx in boundary_indices:
                gt_frame = frames_resized[idx]
                decoded, nbytes = compress_boundary_frame(
                    gt_frame, ref_codec=args.ref_codec, ref_quality=ref_q)
                boundary_compressed[idx] = (decoded, nbytes)

            # Process GOPs
            gop_psnrs = []
            gop_bpps = []
            gop_kbps_list = []

            for g in range(actual_gops):
                gop_frames = gops_gt[g]
                first_idx = g * frames_per_gop_excl_first
                last_idx = (g + 1) * frames_per_gop_excl_first

                first_decoded, first_bytes = boundary_compressed[first_idx]
                last_decoded, last_bytes = boundary_compressed[last_idx]
                gop_first_bytes = first_bytes if g == 0 else 0

                pipe = TurboDDCMWanPipeline(
                    model, K=K, M=M,
                    num_steps=args.steps, num_ddim_tail=args.ddim_tail,
                    guidance_scale=1.0, g_scale=args.g_scale,
                    num_frames=FPG,
                    height=HEIGHT, width=WIDTH, seed=args.seed,
                )

                # FLF2V conditioning
                model.model.cpu()
                torch.cuda.empty_cache()
                flf2v_cond = model.encode_first_last_frames(
                    first_decoded, last_decoded, FPG, HEIGHT, WIDTH)
                model.model.to(model.device)
                torch.cuda.empty_cache()

                # Encode
                step_data, _ = flf2v_encode(
                    pipe, model, gop_frames, flf2v_cond, HEIGHT, WIDTH)

                # Decode
                frames_recon = flf2v_decode(pipe, model, step_data, flf2v_cond)

                # Metrics
                n = min(len(gop_frames), len(frames_recon))
                t_gt = frames_to_tensor(gop_frames[:n])
                t_rec = frames_to_tensor(frames_recon[:n])
                psnr = compute_psnr(t_gt, t_rec)

                # Bitrate
                codebook_bytes = pipe._total_codebook_bits // 8
                boundary_bytes = gop_first_bytes + last_bytes
                gop_total_bytes = codebook_bytes + boundary_bytes
                gop_total_bits = gop_total_bytes * 8
                gop_pixels = n * HEIGHT * WIDTH * 3
                bpp = gop_total_bits / gop_pixels
                duration_s = n / 16.0
                kbps = gop_total_bits / duration_s / 1000.0

                gop_psnrs.append(psnr)
                gop_bpps.append(bpp)
                gop_kbps_list.append(kbps)

                print(f"    GOP {g}: PSNR={psnr:.2f} dB, BPP={bpp:.6f}, "
                      f"{kbps:.1f} kbps (CB={codebook_bytes}B, BD={boundary_bytes}B)")

                del pipe, frames_recon, t_gt, t_rec, step_data
                gc.collect()
                torch.cuda.empty_cache()

            avg_psnr = sum(gop_psnrs) / len(gop_psnrs)
            avg_bpp = sum(gop_bpps) / len(gop_bpps)
            avg_kbps = sum(gop_kbps_list) / len(gop_kbps_list)
            seq_psnrs.append(avg_psnr)
            seq_bpps.append(avg_bpp)
            seq_kbps.append(avg_kbps)

            del frames_resized, gops_gt
            gc.collect()

        # Average across sequences
        overall_psnr = sum(seq_psnrs) / len(seq_psnrs)
        overall_bpp = sum(seq_bpps) / len(seq_bpps)
        overall_kbps = sum(seq_kbps) / len(seq_kbps)

        result = {
            "M": M, "K": K, "ref_quality": ref_q,
            "bits_per_atom": bpa,
            "PSNR_dB": round(overall_psnr, 2),
            "BPP": round(overall_bpp, 6),
            "bitrate_kbps": round(overall_kbps, 1),
            "per_seq_PSNR": {s[0]: round(p, 2) for s, p in zip(all_seqs, seq_psnrs)},
            "per_seq_BPP": {s[0]: round(b, 6) for s, b in zip(all_seqs, seq_bpps)},
        }
        all_results.append(result)

        print(f"\n  >> M={M}, K={K}, ref_q={ref_q}: PSNR={overall_psnr:.2f} dB, "
              f"BPP={overall_bpp:.6f}, {overall_kbps:.1f} kbps")

    # ================================================================
    # Save results
    # ================================================================
    del model
    gc.collect()
    torch.cuda.empty_cache()

    summary = {
        "experiment": "FLF2V R-D Sweep (M x K)",
        "timestamp": datetime.now().isoformat(),
        "resolution": f"{WIDTH}x{HEIGHT}",
        "num_gops": args.num_gops,
        "steps": args.steps,
        "ddim_tail": args.ddim_tail,
        "g_scale": args.g_scale,
        "ref_codec": args.ref_codec,
        "ref_quality": args.ref_quality,
        "sequences": [s[0] for s in all_seqs],
        "results": all_results,
    }
    with open(out / "rd_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV
    fields = ["M", "K", "ref_quality", "bits_per_atom", "PSNR_dB", "BPP", "bitrate_kbps"]
    with open(out / "rd_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    # Print summary table
    print(f"\n{'='*70}")
    print("R-D SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"{'M':>4} {'K':>6} {'ref_q':>6} {'bit/atom':>9} {'PSNR':>8} {'BPP':>10} {'kbps':>8}")
    print("-" * 58)
    for r in sorted(all_results, key=lambda x: x['BPP']):
        print(f"{r['M']:>4} {r['K']:>6} {r['ref_quality']:>6} {r['bits_per_atom']:>9} "
              f"{r['PSNR_dB']:>7.2f} {r['BPP']:>10.6f} {r['bitrate_kbps']:>7.1f}")

    print(f"\nSaved: {out}/rd_sweep.json, {out}/rd_sweep.csv")


if __name__ == "__main__":
    main()
