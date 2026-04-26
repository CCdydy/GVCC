"""
run_ar_gop.py — Autoregressive GOP chaining experiment

Autoregressive: decoded last frame of GOP N → reference frame for GOP N+1
Optional tail boost: allocate more codebook atoms (M_tail) to tail latent frames

Flow:
  GOP 0:  ref = GT first frame (free)
          DDCM encode/decode → 33 frames
          last decoded frame → GOP 1 reference

  GOP 1:  ref = decoded_last_frame from GOP 0
          (optionally + WebP correction for the last frame)
          DDCM encode/decode → 33 frames
          ...

480p default: fast iteration, ~12 GB VRAM for 14B model at 480p.
"""

import sys
import os
import re
import io
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
from sde_rf_wan import WanWrapper
from sde_rf_wan.turbo_pipeline import TurboDDCMWanPipeline
from uvg_data import find_uvg_sequences as find_uvg_sequences_shared


# ==================================================================
# UVG Loading (same as run_uvg_chained_gop.py)
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


def compress_ref_webp(image, quality=5):
    """Compress reference frame with WebP. Returns (decoded_image, num_bytes)."""
    buf = io.BytesIO()
    image.save(buf, format="WebP", quality=quality)
    ref_bytes = buf.tell()
    buf.seek(0)
    decoded = Image.open(buf).copy()
    return decoded, ref_bytes


# ==================================================================
# Latent Residual Compression
# ==================================================================

def compress_latent_residual(residual, downsample=1, bits=8):
    """Compress a single latent frame residual.

    Args:
        residual: (C, H, W) float tensor — latent residual
        downsample: spatial downsample factor (1=none, 2=2x, 4=4x)
        bits: quantization bits (8 or 16)

    Returns:
        compressed_bytes: bytes object
        num_bytes: total byte count
    """
    import zlib

    C, H, W = residual.shape
    r = residual.float().cpu()

    # Spatial downsample
    if downsample > 1:
        r = torch.nn.functional.interpolate(
            r.unsqueeze(0), scale_factor=1.0 / downsample,
            mode='bilinear', align_corners=False
        ).squeeze(0)

    Cd, Hd, Wd = r.shape

    # Per-channel min/max for quantization
    mins = r.reshape(Cd, -1).min(dim=1).values  # (C,)
    maxs = r.reshape(Cd, -1).max(dim=1).values  # (C,)
    ranges = maxs - mins
    ranges[ranges < 1e-8] = 1.0

    levels = 2**bits - 1
    normalized = (r - mins.reshape(Cd, 1, 1)) / ranges.reshape(Cd, 1, 1)
    quantized = (normalized * levels).round().clamp(0, levels)

    if bits <= 8:
        raw = quantized.to(torch.uint8).numpy().tobytes()
    else:
        raw = quantized.to(torch.int16).numpy().tobytes()

    compressed = zlib.compress(raw, level=9)

    # Pack: metadata + compressed data
    import struct
    header = struct.pack('<3I', Cd, Hd, Wd)  # shape after downsample
    header += struct.pack(f'<{Cd}f', *mins.tolist())
    header += struct.pack(f'<{Cd}f', *maxs.tolist())
    header += struct.pack('<2I', downsample, bits)

    data = header + compressed
    return data, len(data)


def decompress_latent_residual(data, orig_shape, device="cuda"):
    """Decompress latent residual back to original shape.

    Args:
        data: compressed bytes from compress_latent_residual
        orig_shape: (C, H, W) original latent frame shape
        device: target device

    Returns:
        residual: (C, H, W) float tensor
    """
    import zlib, struct

    off = 0
    Cd, Hd, Wd = struct.unpack_from('<3I', data, off); off += 12
    mins = torch.tensor(struct.unpack_from(f'<{Cd}f', data, off)); off += Cd * 4
    maxs = torch.tensor(struct.unpack_from(f'<{Cd}f', data, off)); off += Cd * 4
    downsample, bits = struct.unpack_from('<2I', data, off); off += 8

    compressed = data[off:]
    raw = zlib.decompress(compressed)

    levels = 2**bits - 1
    if bits <= 8:
        quantized = torch.from_numpy(
            np.frombuffer(raw, dtype=np.uint8).reshape(Cd, Hd, Wd).copy()
        ).float()
    else:
        quantized = torch.from_numpy(
            np.frombuffer(raw, dtype=np.int16).reshape(Cd, Hd, Wd).copy()
        ).float()

    ranges = maxs - mins
    r = quantized / levels * ranges.reshape(Cd, 1, 1) + mins.reshape(Cd, 1, 1)

    # Upsample back to original spatial size
    if downsample > 1:
        C, H, W = orig_shape
        r = torch.nn.functional.interpolate(
            r.unsqueeze(0), size=(H, W),
            mode='bilinear', align_corners=False
        ).squeeze(0)

    return r.to(device)


# ==================================================================
# Main
# ==================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autoregressive GOP Chaining")
    parser.add_argument("--data_dir", default=os.path.join(_project_root, "data", "uvg"))
    parser.add_argument("--wan_ckpt", default="./Wan2.1-I2V-14B-720P")
    parser.add_argument("--output_dir", default="./results_ar")
    parser.add_argument("--sequence", default="Beauty")
    parser.add_argument("--num_frames_per_gop", type=int, default=33)
    parser.add_argument("--num_gops", type=int, default=2)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--K", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim_tail", type=int, default=3)
    parser.add_argument("--g_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)

    # Tail boost: more codebook atoms for last N latent frames
    parser.add_argument("--M_tail", type=int, default=None,
                        help="Higher M for tail latent frames (e.g., 128, 192)")
    parser.add_argument("--tail_latent_frames", type=int, default=2,
                        help="Number of tail latent frames to boost (default: 2)")

    # Tail correction: WebP of GT at GOP boundary
    parser.add_argument("--tail_webp_q", type=int, default=0,
                        help="WebP quality for GT tail frame correction (0=disabled)")

    # Latent residual correction for tail frame
    parser.add_argument("--latent_residual", action="store_true",
                        help="Enable latent residual correction for last frame")
    parser.add_argument("--residual_downsample", type=int, default=1,
                        help="Spatial downsample factor for residual (1/2/4)")
    parser.add_argument("--residual_bits", type=int, default=8,
                        help="Quantization bits for residual (8 or 16)")
    parser.add_argument("--residual_frames", type=int, default=1,
                        help="Number of tail latent frames to correct (default: 1)")

    args = parser.parse_args()

    # Auto flow_shift
    flow_shift = 3.0 if args.height <= 480 else 5.0

    H, W = args.height, args.width
    FPG = args.num_frames_per_gop
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total_frames = FPG * args.num_gops

    print("=" * 70)
    print(f"Autoregressive GOP Chaining — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Resolution: {W}x{H} (flow_shift={flow_shift})")
    print(f"  GOPs: {args.num_gops} x {FPG} frames")
    print(f"  M={args.M}, K={args.K}, steps={args.steps}, ddim_tail={args.ddim_tail}")
    if args.M_tail:
        print(f"  Tail boost: M_tail={args.M_tail} for last {args.tail_latent_frames} latent frames")
    if args.latent_residual:
        print(f"  Latent residual: {args.residual_frames} tail frame(s), "
              f"ds={args.residual_downsample}x, {args.residual_bits}bit")
    if args.tail_webp_q > 0:
        print(f"  Tail WebP correction: Q={args.tail_webp_q}")
    print(f"  Chain: GT → (decoded_last + residual) → ...")
    print("=" * 70)

    # Find sequence
    all_seqs = find_uvg_sequences(args.data_dir)
    seq_matches = [(n, p) for n, p in all_seqs if n == args.sequence]
    assert seq_matches, f"Sequence {args.sequence} not found in {args.data_dir}"
    seq_name, yuv_path = seq_matches[0]

    # Load frames
    raw_frames = load_yuv420_frames(yuv_path, total_frames, start_frame=0)
    frames_all = [f.resize((W, H), Image.LANCZOS) for f in raw_frames]
    del raw_frames
    print(f"Loaded {len(frames_all)} frames of {seq_name}, {W}x{H}")

    # Split into GOPs
    gops_gt = []
    for g in range(args.num_gops):
        start = g * FPG
        end = start + FPG
        gops_gt.append(frames_all[start:end])
        print(f"  GOP {g}: frames [{start}, {end})")

    # Load model
    model = WanWrapper(args.wan_ckpt, config_name="i2v-14B", flow_shift=flow_shift)
    model.load("cuda", torch.bfloat16)

    # Save GT video
    save_video_mp4(frames_all, out / "original.mp4")

    # ================================================================
    # Autoregressive GOP loop
    # ================================================================
    gop_results = []
    all_recon_frames = []
    next_ref = None  # will be decoded last frame

    for g in range(args.num_gops):
        print(f"\n{'='*60}")
        print(f"  GOP {g}")
        print(f"{'='*60}")

        gop_frames = gops_gt[g]

        pipe = TurboDDCMWanPipeline(
            model, K=args.K, M=args.M,
            num_steps=args.steps, num_ddim_tail=args.ddim_tail,
            guidance_scale=1.0, g_scale=args.g_scale,
            num_frames=FPG, height=H, width=W, seed=args.seed,
            M_tail=args.M_tail, tail_latent_frames=args.tail_latent_frames,
        )

        # Determine reference frame
        extra_bytes = 0
        if g == 0:
            # First GOP: use GT first frame (free)
            ref_image = gop_frames[0]
            ref_source = "GT[0] (free)"
        else:
            # Subsequent GOPs: use decoded last frame from previous GOP
            ref_image = next_ref
            ref_source = f"decoded_last[GOP{g-1}]"

            # Optional: WebP correction for the reference
            if args.tail_webp_q > 0:
                # Blend decoded last frame with WebP of GT at this position
                gt_at_boundary = gop_frames[0]  # GT frame at GOP boundary
                webp_ref, webp_bytes = compress_ref_webp(gt_at_boundary, args.tail_webp_q)
                extra_bytes = webp_bytes

                # Use WebP GT as reference (it's more reliable than decoded)
                # The decoded frame told us "roughly where we are",
                # but the WebP GT anchors us back to ground truth
                ref_image = webp_ref
                ref_source += f" + WebP(GT[{g*FPG}], Q={args.tail_webp_q}, {webp_bytes}B)"

        print(f"  Ref: {ref_source}")

        # Encode
        print(f"  Encoding...")
        t0 = time.time()
        step_data, x_t_enc = pipe.encode(gop_frames, prompt="", ref_image=ref_image)
        t_enc = time.time() - t0

        # Compute latent residual for tail frames (encoder has both x0_true and x_t)
        residual_bytes = 0
        latent_correction = None
        if args.latent_residual:
            x0_true = pipe._gt_latent  # (1, C, F, H, W)
            F_lat = pipe.num_latent_frames
            n_corr = min(args.residual_frames, F_lat)
            frame_shape = pipe.frame_shape  # (C, H, W)

            latent_correction = torch.zeros_like(x_t_enc)
            total_res_bytes = 0

            for fi in range(n_corr):
                f_idx = F_lat - 1 - fi  # last frame first
                residual_f = (x0_true[0, :, f_idx] - x_t_enc[0, :, f_idx])
                compressed, nbytes = compress_latent_residual(
                    residual_f, downsample=args.residual_downsample,
                    bits=args.residual_bits
                )
                # Decompress to verify round-trip
                restored = decompress_latent_residual(compressed, frame_shape)
                latent_correction[0, :, f_idx] = restored
                total_res_bytes += nbytes

                # Log quality of residual compression
                res_mse = ((residual_f.cpu() - restored.cpu()) ** 2).mean().item()
                print(f"    Residual frame {f_idx}: {nbytes}B, quant_MSE={res_mse:.6f}")

            residual_bytes = total_res_bytes
            extra_bytes += residual_bytes
            print(f"    Total residual: {residual_bytes}B")

        # Decode (with optional latent correction)
        print(f"  Decoding...")
        t0 = time.time()
        frames_recon = pipe.decode(step_data, prompt="", ref_image=ref_image,
                                   latent_correction=latent_correction)
        t_dec = time.time() - t0

        # Save the decoded LAST frame as reference for next GOP
        # (this frame benefits from the latent residual correction)
        next_ref = frames_recon[-1].copy()

        # Save outputs
        gop_dir = out / f"gop{g}"
        gop_dir.mkdir(parents=True, exist_ok=True)
        save_video_mp4(frames_recon, gop_dir / "reconstructed.mp4")
        ref_image.save(gop_dir / "ref_used.png")
        next_ref.save(gop_dir / "last_frame_decoded.png")

        # Also save GT last frame for comparison
        gop_frames[-1].save(gop_dir / "last_frame_gt.png")

        all_recon_frames.extend(frames_recon)

        # Metrics
        n = min(len(gop_frames), len(frames_recon))
        t_gt = frames_to_tensor(gop_frames[:n])
        t_rec = frames_to_tensor(frames_recon[:n])

        mean_psnr, per_frame_psnr = compute_psnr(t_gt, t_rec)
        mean_msssim = compute_msssim(t_gt, t_rec)
        mean_lpips = compute_lpips(t_gt, t_rec)

        # Last frame quality (critical for chaining)
        last_psnr = per_frame_psnr[-1].item()

        # Bitrate: codebook + extra (residual + webp correction)
        codebook_bytes = pipe._total_codebook_bits // 8
        ref_bytes = 0 if g == 0 else extra_bytes
        gop_total_bytes = codebook_bytes + ref_bytes
        gop_total_bits = gop_total_bytes * 8
        total_pixels = len(gop_frames) * H * W * 3
        bpp = gop_total_bits / total_pixels
        duration_s = len(gop_frames) / 16.0
        bitrate_kbps = gop_total_bits / duration_s / 1000.0

        result = {
            "gop": g,
            "ref_source": ref_source,
            "PSNR_dB": round(mean_psnr, 2),
            "MS_SSIM": round(mean_msssim, 4) if mean_msssim else None,
            "LPIPS": round(mean_lpips, 4),
            "last_frame_PSNR": round(last_psnr, 2),
            "BPP": round(bpp, 6),
            "codebook_bytes": codebook_bytes,
            "residual_bytes": residual_bytes,
            "extra_bytes": ref_bytes,
            "gop_total_bytes": gop_total_bytes,
            "bitrate_kbps": round(bitrate_kbps, 2),
            "encode_s": round(t_enc, 1),
            "decode_s": round(t_dec, 1),
            "per_frame_psnr": [round(p, 2) for p in per_frame_psnr.tolist()],
        }
        gop_results.append(result)

        with open(gop_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"  PSNR={mean_psnr:.2f} dB (last={last_psnr:.2f}), LPIPS={mean_lpips:.4f}")
        print(f"  Codebook: {codebook_bytes}B, residual: {residual_bytes}B, "
              f"extra: {ref_bytes}B → {gop_total_bytes}B")
        print(f"  {bitrate_kbps:.2f} kbps, enc={t_enc:.0f}s, dec={t_dec:.0f}s")

        # Per-frame PSNR printout
        pf = per_frame_psnr.tolist()
        print(f"  Per-frame: {' '.join(f'{p:.1f}' for p in pf[:5])} ... {' '.join(f'{p:.1f}' for p in pf[-5:])}")

        del pipe, frames_recon, t_gt, t_rec, step_data
        gc.collect()
        torch.cuda.empty_cache()

    # Save full video
    save_video_mp4(all_recon_frames, out / "reconstructed_full.mp4")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY — Autoregressive GOP, {seq_name}, {W}x{H}")
    print(f"{'='*70}")
    print(f"{'GOP':>4} | {'Ref Source':<35} | {'PSNR':>6} | {'Last':>6} | {'LPIPS':>6} | {'kbps':>7}")
    print("-" * 80)
    for r in gop_results:
        ref_short = r['ref_source'][:35]
        print(f" {r['gop']:>3} | {ref_short:<35} | {r['PSNR_dB']:>5.2f} | "
              f"{r['last_frame_PSNR']:>5.2f} | {r['LPIPS']:>5.4f} | {r['bitrate_kbps']:>6.1f}")

    # Quality stability across GOPs
    psnrs = [r['PSNR_dB'] for r in gop_results]
    last_psnrs = [r['last_frame_PSNR'] for r in gop_results]
    print(f"\n  Avg PSNR:      {np.mean(psnrs):.2f} dB")
    print(f"  PSNR spread:   {max(psnrs) - min(psnrs):.2f} dB (smaller = more stable)")
    print(f"  Last frame:    {' → '.join(f'{p:.1f}' for p in last_psnrs)}")

    # Save summary
    summary = {
        "experiment": "Autoregressive GOP",
        "timestamp": datetime.now().isoformat(),
        "sequence": seq_name,
        "config": {
            "resolution": f"{W}x{H}",
            "num_gops": args.num_gops,
            "frames_per_gop": FPG,
            "M": args.M, "K": args.K,
            "M_tail": args.M_tail,
            "tail_latent_frames": args.tail_latent_frames,
            "tail_webp_q": args.tail_webp_q,
            "steps": args.steps,
            "ddim_tail": args.ddim_tail,
            "g_scale": args.g_scale,
            "flow_shift": flow_shift,
        },
        "gop_results": gop_results,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {out}/summary.json")


if __name__ == "__main__":
    main()
