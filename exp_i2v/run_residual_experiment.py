"""
run_residual_experiment.py — Latent Residual Video Compression

Replace codebook noise steering with direct latent-space residual coding:
  1. ODE generate x0_pred from shared seed + compressed ref
  2. Residual r = VAE_encode(GT) - x0_pred
  3. Compress r (quantize + entropy code) → transmit
  4. Reconstruct: x0_pred + decompress(r) → VAE decode → video

Advantage: quality depends on residual compression fidelity,
NOT on first-frame conditioning quality (the I2V bottleneck).

Bitstream: compressed_ref_bytes + compressed_residual_bytes
"""

import sys
import os
import re
import io
import time
import json
import gc
import zlib
import torch
import torch.nn.functional as F_torch
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from sde_rf_wan import WanWrapper
from sde_rf_wan.sde_convert import ode_sample_loop, shifted_timesteps
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


# ==================================================================
# Latent Residual Compression
# ==================================================================

def compress_latent_residual(residual, downsample=1, bits=8):
    """Compress latent residual tensor.

    Args:
        residual: (C, F, H, W) float tensor
        downsample: spatial downsample factor (1/2/4)
        bits: quantization bits (4/8/16)

    Returns:
        (compressed_bytes, nbytes, meta_dict)
    """
    r = residual.float().cpu()
    C, Fr, H, W = r.shape

    # Spatial downsample
    if downsample > 1:
        r = r.reshape(C * Fr, 1, H, W)
        r = F_torch.interpolate(r, scale_factor=1.0 / downsample,
                                mode='bilinear', align_corners=False)
        _, _, Hd, Wd = r.shape
        r = r.reshape(C, Fr, Hd, Wd)
    else:
        Hd, Wd = H, W

    # Per-channel min/max quantization
    maxval = r.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
    r_norm = r / maxval  # [-1, 1]

    levels = 2 ** bits
    r_quant = ((r_norm + 1.0) * (levels - 1) / 2.0).round().clamp(0, levels - 1)

    # Pack to bytes
    if bits <= 8:
        packed = r_quant.to(torch.uint8).numpy().tobytes()
    else:
        packed = r_quant.to(torch.int16).numpy().tobytes()

    # Entropy coding
    compressed = zlib.compress(packed, 9)

    meta = {
        'C': C, 'F': Fr, 'H': H, 'W': W,
        'Hd': Hd, 'Wd': Wd,
        'downsample': downsample, 'bits': bits,
        'maxval': maxval.squeeze().tolist(),
    }
    return compressed, len(compressed), meta


def decompress_latent_residual(compressed, meta, device='cuda'):
    """Decompress latent residual back to original shape."""
    C, Fr = meta['C'], meta['F']
    Hd, Wd = meta['Hd'], meta['Wd']
    H, W = meta['H'], meta['W']
    bits = meta['bits']
    downsample = meta['downsample']
    maxval_list = meta['maxval']
    if isinstance(maxval_list, (int, float)):
        maxval_list = [maxval_list]
    maxval = torch.tensor(maxval_list).float().reshape(C, 1, 1, 1)

    packed = zlib.decompress(compressed)
    levels = 2 ** bits

    if bits <= 8:
        arr = np.frombuffer(packed, dtype=np.uint8).copy()
        r_quant = torch.from_numpy(arr).float().reshape(C, Fr, Hd, Wd)
    else:
        arr = np.frombuffer(packed, dtype=np.int16).copy()
        r_quant = torch.from_numpy(arr).float().reshape(C, Fr, Hd, Wd)

    # Dequantize
    r_norm = r_quant * 2.0 / (levels - 1) - 1.0
    r = r_norm * maxval

    # Upsample if needed
    if downsample > 1:
        r = r.reshape(C * Fr, 1, Hd, Wd)
        r = F_torch.interpolate(r, size=(H, W), mode='bilinear', align_corners=False)
        r = r.reshape(C, Fr, H, W)

    return r.to(device)


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
# Main
# ==================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Latent Residual Compression Experiment")
    parser.add_argument("--data_dir", default=os.path.join(_project_root, "data", "uvg"))
    parser.add_argument("--wan_ckpt", default="./Wan2.1-I2V-14B-720P")
    parser.add_argument("--output_dir", default="./results_residual")
    parser.add_argument("--num_frames_per_gop", type=int, default=33)
    parser.add_argument("--num_gops", type=int, default=1)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_codec", default="compressai",
                        choices=["compressai", "webp", "gt"])
    parser.add_argument("--ref_quality", type=int, default=4)
    parser.add_argument("--flow_shift", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--sequences", nargs="*", default=None)
    # Residual compression settings
    parser.add_argument("--residual_bits", type=int, default=8,
                        help="Quantization bits for residual (4/8/16)")
    parser.add_argument("--residual_downsample", type=int, default=1,
                        help="Spatial downsample for residual (1/2/4)")
    args = parser.parse_args()

    if args.flow_shift is None:
        args.flow_shift = 3.0 if args.height <= 480 else 5.0

    HEIGHT, WIDTH = args.height, args.width
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    FPG = args.num_frames_per_gop
    total_frames_per_seq = FPG * args.num_gops

    print("=" * 70)
    print(f"Latent Residual Experiment — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Resolution: {WIDTH}x{HEIGHT}")
    print(f"  GOPs: {args.num_gops} x {FPG} frames = {total_frames_per_seq} frames/seq")
    print(f"  ODE steps: {args.steps}, flow_shift: {args.flow_shift}")
    print(f"  Residual: {args.residual_bits}bit, ds{args.residual_downsample}x")
    if args.ref_codec == "gt":
        print(f"  Ref: GT (free)")
    else:
        print(f"  Ref: {args.ref_codec} q={args.ref_quality}")
    print("=" * 70)

    # Find sequences
    all_seqs = find_uvg_sequences(args.data_dir)
    if args.sequences:
        all_seqs = [(n, p) for n, p in all_seqs if n in args.sequences]
    print(f"\nSequences: {[s[0] for s in all_seqs]}")

    # Load model
    model = WanWrapper(args.wan_ckpt, config_name="i2v-14B", flow_shift=args.flow_shift)
    model.load("cuda", torch.bfloat16)

    timesteps = shifted_timesteps(args.steps, shift=args.flow_shift, device="cuda")
    latent_shape = model.get_latent_shape(FPG, HEIGHT, WIDTH)
    print(f"  Latent shape: {latent_shape}")

    all_seq_results = []

    for seq_name, yuv_path in all_seqs:
        print(f"\n{'='*70}")
        print(f"  Sequence: {seq_name}")
        print(f"{'='*70}")

        seq_dir = out / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        raw_frames = load_yuv420_frames(yuv_path, total_frames_per_seq, start_frame=0)
        frames_720 = resize_frames(raw_frames, WIDTH, HEIGHT)
        del raw_frames
        print(f"  Loaded {len(frames_720)} frames → {WIDTH}x{HEIGHT}")

        gops_gt = []
        for g in range(args.num_gops):
            start = g * FPG
            end = start + FPG
            gops_gt.append(frames_720[start:end])

        save_video_mp4(frames_720, seq_dir / "original.mp4")

        gop_results = []
        all_recon_frames = []

        for g in range(args.num_gops):
            print(f"\n  --- {seq_name} GOP {g} ---")
            gop_frames = gops_gt[g]

            # ============================================================
            # Step 1: Compress reference frame
            # ============================================================
            if args.ref_codec == "gt":
                ref_image = gop_frames[0]
                ref_bytes = 0
            else:
                ref_image, _, ref_bytes = compress_ref(
                    gop_frames[0], codec=args.ref_codec, quality=args.ref_quality)
            print(f"  Ref: {args.ref_codec}, {ref_bytes}B")

            # ============================================================
            # Step 2: VAE encode ground-truth → x0_true
            # ============================================================
            model.model.cpu()
            torch.cuda.empty_cache()

            # I2V conditioning from compressed ref
            i2v_cond = model.encode_image(ref_image, FPG, HEIGHT, WIDTH)
            # VAE encode GT
            x0_true = model.encode_video(gop_frames, HEIGHT, WIDTH)
            print(f"  x0_true shape: {x0_true.shape}")

            # ============================================================
            # Step 3: ODE generate x0_pred (deterministic, same as decoder)
            # ============================================================
            model.model.to("cuda")
            torch.cuda.empty_cache()

            embeds = model.encode_prompt("")

            def model_fn(x_t, t):
                return model.predict_velocity_cfg(
                    x_t, t, embeds, args.guidance_scale, i2v_cond)

            gen = torch.Generator(device="cpu").manual_seed(args.seed)
            x_T = torch.randn(1, *latent_shape, generator=gen).to("cuda")

            print(f"  Running ODE ({args.steps} steps)...")
            t0 = time.time()
            x0_pred = ode_sample_loop(model_fn, x_T, timesteps)
            t_ode = time.time() - t0
            print(f"  ODE done in {t_ode:.1f}s")

            # ============================================================
            # Step 4: Compute & compress latent residual
            # ============================================================
            residual = (x0_true - x0_pred).squeeze(0)  # (C, F, H, W)
            res_mse = (residual ** 2).mean().item()
            print(f"  Residual MSE: {res_mse:.6f}")

            compressed, res_bytes, meta = compress_latent_residual(
                residual,
                downsample=args.residual_downsample,
                bits=args.residual_bits,
            )
            print(f"  Residual compressed: {res_bytes}B "
                  f"(ds{args.residual_downsample}x, {args.residual_bits}bit)")

            # ============================================================
            # Step 5: Reconstruct (simulate decoder)
            # ============================================================
            r_decompressed = decompress_latent_residual(compressed, meta, device="cuda")
            x0_recon = x0_pred + r_decompressed.unsqueeze(0).to(x0_pred.dtype)

            # Measure latent reconstruction quality
            latent_recon_mse = ((x0_true - x0_recon) ** 2).mean().item()
            print(f"  Latent recon MSE: {latent_recon_mse:.6f} "
                  f"(residual quant loss)")

            # ============================================================
            # Step 6: VAE decode → pixel frames
            # ============================================================
            # Decode residual-corrected latent
            x0_recon_cpu = x0_recon.cpu()
            x0_pred_cpu = x0_pred.cpu()
            del x0_true, x0_pred, x0_recon, residual, r_decompressed
            gc.collect()
            torch.cuda.empty_cache()

            model.model.cpu()
            gc.collect()
            torch.cuda.empty_cache()

            frames_recon = model.decode_latent(x0_recon_cpu.to("cuda"))
            del x0_recon_cpu
            gc.collect()
            torch.cuda.empty_cache()

            # Decode ODE-only (no residual) for comparison
            frames_ode_only = model.decode_latent(x0_pred_cpu.to("cuda"))
            del x0_pred_cpu
            gc.collect()
            torch.cuda.empty_cache()

            # ============================================================
            # Step 7: Metrics
            # ============================================================
            gop_dir = seq_dir / f"gop{g}"
            gop_dir.mkdir(parents=True, exist_ok=True)
            save_video_mp4(frames_recon, gop_dir / "reconstructed.mp4")
            save_video_mp4(frames_ode_only, gop_dir / "ode_only.mp4")
            ref_image.save(gop_dir / "ref_used.png")

            all_recon_frames.extend(frames_recon)

            n = min(len(gop_frames), len(frames_recon))
            t_gt = frames_to_tensor(gop_frames[:n])
            t_rec = frames_to_tensor(frames_recon[:n])
            t_ode_only = frames_to_tensor(frames_ode_only[:n])

            mean_psnr, per_frame_psnr = compute_psnr(t_gt, t_rec)
            mean_msssim = compute_msssim(t_gt, t_rec)
            mean_lpips = compute_lpips(t_gt, t_rec)

            # ODE-only metrics (baseline without residual)
            ode_psnr, ode_per_frame = compute_psnr(t_gt, t_ode_only)
            ode_lpips = compute_lpips(t_gt, t_ode_only)

            # Bitrate
            total_bytes = ref_bytes + res_bytes
            total_bits = total_bytes * 8
            total_pixels = len(gop_frames) * HEIGHT * WIDTH * 3
            bpp = total_bits / total_pixels
            duration_s = len(gop_frames) / 16.0
            bitrate_kbps = total_bits / duration_s / 1000.0

            result = {
                "sequence": seq_name,
                "gop": g,
                "PSNR_dB": round(mean_psnr, 2),
                "MS_SSIM": round(mean_msssim, 4) if mean_msssim else None,
                "LPIPS": round(mean_lpips, 4),
                "BPP": round(bpp, 6),
                "ref_bytes": ref_bytes,
                "residual_bytes": res_bytes,
                "total_bytes": total_bytes,
                "bitrate_kbps": round(bitrate_kbps, 2),
                "residual_bits": args.residual_bits,
                "residual_downsample": args.residual_downsample,
                "residual_mse": round(res_mse, 6),
                "latent_recon_mse": round(latent_recon_mse, 6),
                "ode_time_s": round(t_ode, 1),
                "ode_only_PSNR": round(ode_psnr, 2),
                "ode_only_LPIPS": round(ode_lpips, 4),
                "per_frame_psnr": [round(p, 2) for p in per_frame_psnr.tolist()],
                "ode_per_frame_psnr": [round(p, 2) for p in ode_per_frame.tolist()],
            }
            gop_results.append(result)

            with open(gop_dir / "metrics.json", "w") as f:
                json.dump(result, f, indent=2)

            print(f"\n  === Results ===")
            print(f"  ODE-only:      PSNR={ode_psnr:.2f}dB  LPIPS={ode_lpips:.4f}")
            print(f"  + Residual:    PSNR={mean_psnr:.2f}dB  LPIPS={mean_lpips:.4f}")
            print(f"  Bitrate:       ref={ref_bytes}B + res={res_bytes}B = {total_bytes}B")
            print(f"                 BPP={bpp:.6f}, {bitrate_kbps:.2f} kbps")

            del frames_recon, frames_ode_only, t_gt, t_rec, t_ode_only
            gc.collect()
            torch.cuda.empty_cache()

        save_video_mp4(all_recon_frames, seq_dir / "reconstructed_full.mp4")

        avg_psnr = np.mean([r["PSNR_dB"] for r in gop_results])
        avg_lpips = np.mean([r["LPIPS"] for r in gop_results])
        avg_ode_psnr = np.mean([r["ode_only_PSNR"] for r in gop_results])

        all_seq_results.append({
            "sequence": seq_name,
            "gop_results": gop_results,
            "avg_PSNR": round(float(avg_psnr), 2),
            "avg_LPIPS": round(float(avg_lpips), 4),
            "avg_ode_only_PSNR": round(float(avg_ode_psnr), 2),
        })

        del frames_720, all_recon_frames
        gc.collect()

    # ================================================================
    # Summary
    # ================================================================
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"SUMMARY — Latent Residual, {args.residual_bits}bit ds{args.residual_downsample}x")
    print(f"{'='*70}")

    hdr = "{:<12} {:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}"
    row = "{:<12} {:>4} {:>7.2f} {:>7.2f} {:>7.4f} {:>8} {:>8} {:>10} {:>7.2f}"
    print(hdr.format('Seq', 'GOP', 'PSNR', 'ODE', 'LPIPS',
                      'Ref(B)', 'Res(B)', 'Total(B)', 'kbps'))
    print("-" * 90)

    for sr in all_seq_results:
        for r in sr["gop_results"]:
            print(row.format(
                r['sequence'], r['gop'], r['PSNR_dB'], r['ode_only_PSNR'],
                r['LPIPS'], r['ref_bytes'], r['residual_bytes'],
                r['total_bytes'], r['bitrate_kbps']))

    print("-" * 70)
    overall_psnr = np.mean([sr["avg_PSNR"] for sr in all_seq_results])
    overall_lpips = np.mean([sr["avg_LPIPS"] for sr in all_seq_results])
    overall_ode = np.mean([sr["avg_ode_only_PSNR"] for sr in all_seq_results])
    print(f"{'OVERALL':<12} {'':>4} {overall_psnr:>7.2f} {overall_ode:>7.2f} "
          f"{overall_lpips:>7.4f}")

    summary = {
        "experiment": "Latent Residual Compression",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "resolution": f"{WIDTH}x{HEIGHT}",
            "num_gops": args.num_gops,
            "frames_per_gop": FPG,
            "ode_steps": args.steps,
            "flow_shift": args.flow_shift,
            "guidance_scale": args.guidance_scale,
            "ref_codec": args.ref_codec,
            "ref_quality": args.ref_quality if args.ref_codec != "gt" else None,
            "residual_bits": args.residual_bits,
            "residual_downsample": args.residual_downsample,
        },
        "sequences": all_seq_results,
        "overall": {
            "PSNR_dB": round(float(overall_psnr), 2),
            "LPIPS": round(float(overall_lpips), 4),
            "ode_only_PSNR": round(float(overall_ode), 2),
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {out}/summary.json")


if __name__ == "__main__":
    main()
