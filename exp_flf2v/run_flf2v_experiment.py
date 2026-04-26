"""
run_flf2v_experiment.py — FLF2V (First-Last-Frame) Chained-GOP Experiment on UVG

Key differences from I2V chained-GOP experiment:
  1. Conditions on BOTH first and last frames (not just first)
  2. Boundary frame sharing: last frame of GOP N = first frame of GOP N+1
  3. Bitrate accounting: first frame only compressed at GOP 0, thereafter reused
  4. Frame alignment: need 32*N+1 frames for N GOPs of 33 frames each

GOP chaining:
  GOP 0: first=compress(GT[0]),  last=compress(GT[32])  -> 33 frames [0..32]
  GOP 1: first=compress(GT[32]), last=compress(GT[64])   -> 33 frames [32..64]
  GOP 2: first=compress(GT[64]), last=compress(GT[96])   -> 33 frames [64..96]
  Concatenation: GOP0[0:33] + GOP1[1:33] + GOP2[1:33] + ...
  Total: 33 + 32*(N-1) = 32*N + 1 unique frames

Bitrate per GOP:
  GOP 0: first_bytes + last_bytes + codebook_bytes
  GOP k>0: last_bytes + codebook_bytes  (first frame reused from previous GOP)
"""

import sys
import os
import re
import time
import json
import gc
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
# UVG Loading
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
    """Compress a boundary frame. Returns (decoded_image, num_bytes)."""
    decoded, _, nbytes = compress_ref(
        image, codec=ref_codec, quality=ref_quality
    )
    return decoded, nbytes


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
    return psnr.mean().item(), psnr


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


def save_video_mp4(frames, path, fps=16):
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec='libx264', quality=8)
    for f in frames:
        writer.append_data(np.array(f))
    writer.close()


# ==================================================================
# FLF2V encode / decode helpers
# ==================================================================

def flf2v_encode(pipe, model, gop_frames, flf2v_cond, height, width):
    """Encode a GOP using FLF2V conditioning.

    Manually drives the SDE loop (cannot use pipe.encode() because that
    calls model.encode_image which is I2V-only).

    Returns:
        step_data: list of per-step per-frame (indices, signs)
        x0_true: ground truth latent (kept for external metric use)
    """
    embeds = model.encode_prompt("")

    # Offload DiT for VAE encoding of GT video
    model.model.cpu()
    torch.cuda.empty_cache()

    x0_true = model.encode_video(gop_frames, height, width)

    # Bring DiT back
    model.model.to(model.device)
    torch.cuda.empty_cache()

    model_fn = pipe._model_fn(embeds, flf2v_cond)

    # Deterministic initial noise (shared with decoder)
    gen = torch.Generator(device="cpu").manual_seed(pipe.seed)
    x_t = torch.randn(1, *pipe.latent_shape, generator=gen).to(pipe.device)

    step_data = []
    sde_idx = 0

    for i in range(pipe.num_steps):
        t_curr = pipe.timesteps[i].item()
        t_next = pipe.timesteps[i + 1].item()
        delta_t = t_curr - t_next

        u_t = model_fn(x_t, t_curr)

        # Final step -> ODE to t=0
        if t_next < 1e-6:
            x_t = x_t - u_t * delta_t
            break

        # DDIM tail -> ODE step, no bits
        if i >= pipe.num_sde_steps:
            x_t = x_t - u_t * delta_t
            continue

        # --- Turbo-DDCM SDE step ---
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

        if (i + 1) % 5 == 0 or i == 0:
            mse = ((x0_true - x0_hat) ** 2).mean().item()
            print(f"    Encode step {i+1}/{pipe.num_steps}: "
                  f"residual_MSE={mse:.4f}, noise_coeff={noise_coeff:.4f}")

    return step_data, x0_true


def flf2v_decode(pipe, model, step_data, flf2v_cond):
    """Decode a GOP from step_data using FLF2V conditioning.

    Replays the exact SDE trajectory as the encoder.
    """
    embeds = model.encode_prompt("")

    model.model.to(model.device)
    torch.cuda.empty_cache()

    model_fn = pipe._model_fn(embeds, flf2v_cond)

    # Same deterministic initial noise
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

        # Reconstruct noise from stored indices + signs
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

        if (i + 1) % 5 == 0:
            print(f"    Decode step {i+1}/{pipe.num_steps}")

    # VAE decode with aggressive memory cleanup
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
# Main
# ==================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FLF2V Chained-GOP UVG Experiment")
    parser.add_argument("--data_dir", default=os.path.join(_project_root, "data", "uvg"))
    parser.add_argument("--wan_ckpt", default="./Wan2.1-FLF2V-14B-720P")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--num_frames_per_gop", type=int, default=33,
                        help="Frames per GOP (must be 4k+1: 17, 21, 25, 29, 33, ...)")
    parser.add_argument("--num_gops", type=int, default=3)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--K", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim_tail", type=int, default=3)
    parser.add_argument("--g_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_codec", default="compressai",
                        choices=["compressai", "webp", "gt"],
                        help="Boundary frame codec: compressai (learned), webp, or gt (free)")
    parser.add_argument("--ref_quality", type=int, default=4,
                        help="Boundary codec quality (compressai: 1-6, webp: 0-100)")
    parser.add_argument("--flow_shift", type=float, default=None,
                        help="Timestep shift (auto: 3.0 for 480p, 5.0 for 720p+)")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Specific sequences to test (default: all)")
    args = parser.parse_args()

    # Auto flow_shift based on resolution
    if args.flow_shift is None:
        args.flow_shift = 3.0 if args.height <= 480 else 5.0

    HEIGHT, WIDTH = args.height, args.width
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    FPG = args.num_frames_per_gop
    # FLF2V chained GOP: need (FPG-1)*N + 1 unique frames for N GOPs
    frames_per_gop_excl_first = FPG - 1  # 32 for FPG=33

    print("=" * 70)
    print(f"FLF2V Chained-GOP UVG Experiment -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Resolution: {WIDTH}x{HEIGHT}")
    print(f"  Frames per GOP: {FPG}, num_gops: {args.num_gops} (0=auto/all)")
    print(f"  M={args.M}, K={args.K}, g_scale={args.g_scale}, flow_shift={args.flow_shift}")
    if args.ref_codec == "gt":
        print(f"  Boundary frames: GT (raw, free)")
    else:
        print(f"  Boundary frames: {args.ref_codec}(q={args.ref_quality})")
    print(f"  Shared boundaries: last frame of GOP N = first frame of GOP N+1")
    print("=" * 70)

    # ================================================================
    # Find sequences
    # ================================================================
    all_seqs = find_uvg_sequences(args.data_dir)
    if args.sequences:
        all_seqs = [(n, p) for n, p in all_seqs if n in args.sequences]
    print(f"\nSequences: {[s[0] for s in all_seqs]}")

    if not all_seqs:
        print("ERROR: No sequences found. Check --data_dir path.")
        sys.exit(1)

    # ================================================================
    # Load model
    # ================================================================
    model = WanFLF2VWrapper(
        args.wan_ckpt, config_name="flf2v-14B", flow_shift=args.flow_shift
    )
    model.load("cuda", torch.bfloat16)

    all_seq_results = []

    for seq_idx, (seq_name, yuv_path) in enumerate(all_seqs):
        print(f"\n{'='*70}")
        print(f"  [{seq_idx+1}/{len(all_seqs)}] Sequence: {seq_name}  ({datetime.now().strftime('%H:%M:%S')})")
        print(f"{'='*70}")

        seq_dir = out / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        # Load frames — num_gops=0 means use all available frames
        if args.num_gops > 0:
            total_unique_frames = frames_per_gop_excl_first * args.num_gops + 1
        else:
            total_unique_frames = 999999

        raw_frames = load_yuv420_frames(yuv_path, total_unique_frames, start_frame=0)

        if args.num_gops <= 0:
            actual_gops = max(1, (len(raw_frames) - 1) // frames_per_gop_excl_first)
        elif len(raw_frames) < total_unique_frames:
            print(f"  WARNING: Only got {len(raw_frames)} frames, reducing num_gops")
            actual_gops = max(1, (len(raw_frames) - 1) // frames_per_gop_excl_first)
        else:
            actual_gops = args.num_gops

        actual_total = frames_per_gop_excl_first * actual_gops + 1
        raw_frames = raw_frames[:actual_total]
        frames_resized = resize_frames(raw_frames, WIDTH, HEIGHT)
        del raw_frames
        print(f"  Loaded {len(frames_resized)} unique frames, resized to {WIDTH}x{HEIGHT}")

        # Build GOP frame ranges (with shared boundaries)
        gops_gt = []
        for g in range(actual_gops):
            start = g * frames_per_gop_excl_first  # 0, 32, 64, ...
            end = start + FPG                       # 33, 65, 97, ...
            gop_frames = frames_resized[start:end]
            gops_gt.append(gop_frames)
            print(f"  GOP {g}: frames [{start}, {end})  "
                  f"(first=GT[{start}], last=GT[{end-1}])")

        # Save original full sequence
        save_video_mp4(frames_resized, seq_dir / "original.mp4")

        # ============================================================
        # Compress boundary frames (with sharing)
        # ============================================================
        # Boundary frame indices: 0, 32, 64, 96, ...
        # For N GOPs we have N+1 boundary frames
        boundary_indices = [g * frames_per_gop_excl_first for g in range(actual_gops + 1)]
        boundary_indices = [min(idx, len(frames_resized) - 1) for idx in boundary_indices]

        boundary_compressed = {}  # index -> (decoded_image, num_bytes)
        total_boundary_bytes = 0

        print(f"\n  Compressing {len(boundary_indices)} boundary frames...")
        for idx in boundary_indices:
            gt_frame = frames_resized[idx]
            if args.ref_codec == "gt":
                boundary_compressed[idx] = (gt_frame, 0)
            else:
                decoded, nbytes = compress_boundary_frame(
                    gt_frame, ref_codec=args.ref_codec, ref_quality=args.ref_quality
                )
                boundary_compressed[idx] = (decoded, nbytes)
                total_boundary_bytes += nbytes
                print(f"    Boundary frame {idx}: {nbytes} bytes")

        # ============================================================
        # Process each GOP
        # ============================================================
        gop_results = []
        all_recon_frames = []

        for g in range(actual_gops):
            print(f"\n  --- {seq_name} GOP {g}/{actual_gops-1}  ({datetime.now().strftime('%H:%M:%S')}) ---")
            gop_frames = gops_gt[g]
            first_idx = g * frames_per_gop_excl_first
            last_idx = (g + 1) * frames_per_gop_excl_first

            first_decoded, first_bytes = boundary_compressed[first_idx]
            last_decoded, last_bytes = boundary_compressed[last_idx]

            # For GOP > 0, first frame is reused from previous GOP (no extra bytes)
            if g > 0:
                gop_first_bytes = 0  # already counted in previous GOP
                first_source = f"reused from GOP {g-1}"
            else:
                gop_first_bytes = first_bytes
                if args.ref_codec == "gt":
                    first_source = f"GT[{first_idx}] (free)"
                else:
                    first_source = (f"{args.ref_codec}(GT[{first_idx}], "
                                    f"q={args.ref_quality}, {first_bytes}B)")

            if args.ref_codec == "gt":
                last_source = f"GT[{last_idx}] (free)"
            else:
                last_source = (f"{args.ref_codec}(GT[{last_idx}], "
                               f"q={args.ref_quality}, {last_bytes}B)")

            print(f"  First frame: {first_source}")
            print(f"  Last frame:  {last_source}")

            # Create pipeline (reuses model for velocity predictions)
            pipe = TurboDDCMWanPipeline(
                model, K=args.K, M=args.M,
                num_steps=args.steps, num_ddim_tail=args.ddim_tail,
                guidance_scale=1.0, g_scale=args.g_scale,
                num_frames=FPG,
                height=HEIGHT, width=WIDTH, seed=args.seed,
            )

            # Compute FLF2V conditioning (offload DiT for CLIP+VAE)
            model.model.cpu()
            torch.cuda.empty_cache()
            flf2v_cond = model.encode_first_last_frames(
                first_decoded, last_decoded, FPG, HEIGHT, WIDTH
            )
            model.model.to(model.device)
            torch.cuda.empty_cache()

            # ---- Encode ----
            print(f"  Encoding...")
            t0 = time.time()
            step_data, x0_true = flf2v_encode(
                pipe, model, gop_frames, flf2v_cond, HEIGHT, WIDTH
            )
            t_enc = time.time() - t0

            # ---- Decode ----
            print(f"  Decoding...")
            t0 = time.time()

            # Recompute conditioning for decode (encoder may have shuffled GPU state)
            model.model.cpu()
            torch.cuda.empty_cache()
            flf2v_cond_dec = model.encode_first_last_frames(
                first_decoded, last_decoded, FPG, HEIGHT, WIDTH
            )
            model.model.to(model.device)
            torch.cuda.empty_cache()

            frames_recon = flf2v_decode(pipe, model, step_data, flf2v_cond_dec)
            t_dec = time.time() - t0

            # ---- Save GOP outputs ----
            gop_dir = seq_dir / f"gop{g}"
            gop_dir.mkdir(parents=True, exist_ok=True)
            save_video_mp4(frames_recon, gop_dir / "reconstructed.mp4")
            pipe.save_compressed(step_data, str(gop_dir / "codebook.tdcm"))
            first_decoded.save(gop_dir / "first_frame_used.png")
            last_decoded.save(gop_dir / "last_frame_used.png")

            # Collect for full video (skip shared first frame for GOP > 0)
            if g == 0:
                all_recon_frames.extend(frames_recon)
            else:
                all_recon_frames.extend(frames_recon[1:])

            # ---- Metrics ----
            n = min(len(gop_frames), len(frames_recon))
            t_gt = frames_to_tensor(gop_frames[:n])
            t_rec = frames_to_tensor(frames_recon[:n])

            mean_psnr, per_frame_psnr = compute_psnr(t_gt, t_rec)
            mean_msssim = compute_msssim(t_gt, t_rec)
            mean_lpips = compute_lpips(t_gt, t_rec)

            # ---- Bitrate ----
            T_sde = pipe.num_sde_steps
            F_lat = pipe.num_latent_frames
            bits_per_fs = pipe.codebook.bits_per_frame_step
            codebook_bits = T_sde * F_lat * bits_per_fs
            codebook_bytes = codebook_bits // 8

            gop_boundary_bytes = gop_first_bytes + last_bytes
            gop_total_bytes = codebook_bytes + gop_boundary_bytes
            gop_total_bits = gop_total_bytes * 8

            # Unique frames contributed by this GOP
            gop_unique_frames = FPG if g == 0 else FPG - 1

            total_pixels = gop_unique_frames * HEIGHT * WIDTH * 3
            bpp = gop_total_bits / total_pixels if total_pixels > 0 else 0
            duration_s = gop_unique_frames / 16.0
            bitrate_kbps = gop_total_bits / duration_s / 1000.0 if duration_s > 0 else 0

            result = {
                "sequence": seq_name,
                "gop": g,
                "first_frame_source": first_source,
                "last_frame_source": last_source,
                "PSNR_dB": round(mean_psnr, 2),
                "MS_SSIM": round(mean_msssim, 4) if mean_msssim else None,
                "LPIPS": round(mean_lpips, 4),
                "BPP": round(bpp, 6),
                "codebook_bytes": codebook_bytes,
                "first_frame_bytes": gop_first_bytes,
                "last_frame_bytes": last_bytes,
                "boundary_bytes": gop_boundary_bytes,
                "gop_total_bytes": gop_total_bytes,
                "gop_unique_frames": gop_unique_frames,
                "bitrate_kbps": round(bitrate_kbps, 2),
                "encode_s": round(t_enc, 1),
                "decode_s": round(t_dec, 1),
                "per_frame_psnr": [round(p, 2) for p in per_frame_psnr.tolist()],
            }
            gop_results.append(result)

            with open(gop_dir / "metrics.json", "w") as mf:
                json.dump(result, mf, indent=2)

            ms_str = f"{mean_msssim:.4f}" if mean_msssim else "N/A"
            print(f"  PSNR={mean_psnr:.2f} dB, MS-SSIM={ms_str}, "
                  f"LPIPS={mean_lpips:.4f}")
            print(f"  Codebook: {codebook_bytes}B + Boundary: {gop_boundary_bytes}B "
                  f"(first={gop_first_bytes}B, last={last_bytes}B) = {gop_total_bytes}B")
            print(f"  BPP={bpp:.6f}, {bitrate_kbps:.2f} kbps "
                  f"({gop_unique_frames} unique frames)")
            print(f"  Time: enc={t_enc:.0f}s, dec={t_dec:.0f}s")

            del pipe, frames_recon, t_gt, t_rec, step_data, x0_true
            gc.collect()
            torch.cuda.empty_cache()

        # Save full reconstructed video
        save_video_mp4(all_recon_frames, seq_dir / "reconstructed_full.mp4")
        print(f"\n  Full reconstruction: {len(all_recon_frames)} frames")

        # Per-sequence averages
        avg_psnr = np.mean([r["PSNR_dB"] for r in gop_results])
        avg_lpips = np.mean([r["LPIPS"] for r in gop_results])
        total_seq_bytes = sum(r["gop_total_bytes"] for r in gop_results)
        total_seq_unique = sum(r["gop_unique_frames"] for r in gop_results)
        total_seq_pixels = total_seq_unique * HEIGHT * WIDTH * 3
        seq_bpp = (total_seq_bytes * 8) / total_seq_pixels if total_seq_pixels > 0 else 0
        seq_duration = total_seq_unique / 16.0
        seq_kbps = (total_seq_bytes * 8) / seq_duration / 1000.0 if seq_duration > 0 else 0

        all_seq_results.append({
            "sequence": seq_name,
            "gop_results": gop_results,
            "avg_PSNR": round(float(avg_psnr), 2),
            "avg_LPIPS": round(float(avg_lpips), 4),
            "total_bytes": total_seq_bytes,
            "total_unique_frames": total_seq_unique,
            "seq_BPP": round(float(seq_bpp), 6),
            "seq_bitrate_kbps": round(float(seq_kbps), 2),
        })

        del frames_resized, all_recon_frames
        gc.collect()

    # ================================================================
    # Summary
    # ================================================================
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY -- FLF2V Chained-GOP, M={args.M}")
    print(f"{'='*70}")

    hdr = "{:<12} {:>4} {:>8} {:>9} {:>8} {:>8} {:>8} {:>10} {:>8}"
    row = "{:<12} {:>4} {:>7.2f} {:>9} {:>7.4f} {:>8} {:>8} {:>10} {:>7.2f}"
    print(hdr.format('Seq', 'GOP', 'PSNR', 'MS-SSIM', 'LPIPS',
                      'CB(B)', 'Bnd(B)', 'Total(B)', 'kbps'))
    print("-" * 90)

    for sr in all_seq_results:
        for r in sr["gop_results"]:
            ms = "{:.4f}".format(r['MS_SSIM']) if r['MS_SSIM'] else "N/A"
            print(row.format(r['sequence'], r['gop'], r['PSNR_dB'], ms,
                             r['LPIPS'], r['codebook_bytes'], r['boundary_bytes'],
                             r['gop_total_bytes'], r['bitrate_kbps']))

    print("-" * 90)
    overall_psnr = np.mean([sr["avg_PSNR"] for sr in all_seq_results])
    overall_lpips = np.mean([sr["avg_LPIPS"] for sr in all_seq_results])
    overall_bpp = np.mean([sr["seq_BPP"] for sr in all_seq_results])
    overall_kbps = np.mean([sr["seq_bitrate_kbps"] for sr in all_seq_results])
    print(f"{'OVERALL':<12} {'':>4} {overall_psnr:>7.2f} {'':>9} {overall_lpips:>7.4f} "
          f"{'':>8} {'':>8} {'':>10} {overall_kbps:>7.2f}")

    print(f"\n  Boundary frame sharing saves:")
    n_gops_total = sum(len(sr["gop_results"]) for sr in all_seq_results)
    n_reused = sum(max(0, len(sr["gop_results"]) - 1) for sr in all_seq_results)
    print(f"    {n_reused}/{n_gops_total} GOPs reused first frame (0 extra bytes)")
    print(f"    Total boundary bytes: {total_boundary_bytes}")

    # Save summary
    summary = {
        "experiment": "FLF2V Chained-GOP UVG",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "resolution": f"{WIDTH}x{HEIGHT}",
            "num_gops": args.num_gops,
            "frames_per_gop": FPG,
            "total_unique_frames_per_seq": total_unique_frames,
            "M": args.M, "K": args.K,
            "g_scale": args.g_scale,
            "flow_shift": args.flow_shift,
            "guidance_scale": 1.0,
            "ref_codec": args.ref_codec,
            "ref_quality": args.ref_quality if args.ref_codec != "gt" else None,
            "shared_boundaries": True,
        },
        "sequences": all_seq_results,
        "overall": {
            "PSNR_dB": round(float(overall_psnr), 2),
            "LPIPS": round(float(overall_lpips), 4),
            "BPP": round(float(overall_bpp), 6),
            "bitrate_kbps": round(float(overall_kbps), 2),
        },
    }
    with open(out / "summary.json", "w") as sf:
        json.dump(summary, sf, indent=2)
    print(f"  Summary: {out}/summary.json")


if __name__ == "__main__":
    main()
