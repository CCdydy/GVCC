"""
run_t2v_experiment.py — T2V Unconditional Compression Experiment

Uses Wan2.1 T2V 1.3B with EMPTY text prompt to test pure model-prior
+ codebook compression capability. No reference frame, no text conditioning.

Key differences from I2V chained-GOP experiment:
  - T2V model (1.3B) instead of I2V (14B)
  - Empty prompt — purely tests model prior + codebook
  - No reference frame — bitrate = codebook bits only
  - Resolution: 480x832 (T2V 1.3B supported size)
  - guidance_scale=1.0 (no CFG, since prompt is empty)
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

# Add project root (parent of exp_t2v/) to path for shared modules
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from sde_rf_wan.wan_t2v_wrapper import WanT2VDiffusersWrapper
from sde_rf_wan.turbo_pipeline import TurboDDCMWanPipeline
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


def load_image_sequence(img_dir, num_frames, start_frame=0):
    """Load frames from a directory of images (png/jpg)."""
    img_dir = Path(img_dir)
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in exts])
    frames = []
    for f in files[start_frame:start_frame + num_frames]:
        frames.append(Image.open(f).convert('RGB'))
    return frames


def find_uvg_sequences(data_dir):
    return find_uvg_sequences_shared(data_dir)


def resize_frames(frames, target_w, target_h):
    return [f.resize((target_w, target_h), Image.LANCZOS) for f in frames]


def blend_frames(frames_a, frames_b, num_overlap):
    """Linear cross-fade between tail of frames_a and head of frames_b.

    Args:
        frames_a: list of PIL Images (previous GOP recon)
        frames_b: list of PIL Images (next GOP recon)
        num_overlap: number of overlapping frames

    Returns:
        blended: list of num_overlap PIL Images
    """
    blended = []
    for i in range(num_overlap):
        w_b = (i + 1) / (num_overlap + 1)
        w_a = 1.0 - w_b
        arr_a = np.array(frames_a[-(num_overlap - i)]).astype(np.float32)
        arr_b = np.array(frames_b[i]).astype(np.float32)
        mixed = (w_a * arr_a + w_b * arr_b).clip(0, 255).astype(np.uint8)
        blended.append(Image.fromarray(mixed))
    return blended


def stitch_gops(gop_recons, overlap):
    """Stitch decoded GOPs with overlap blending.

    With overlap=2, GOP boundaries get a 2-frame linear cross-fade.
    Each GOP contributes its non-overlapping interior plus blended edges.

    Args:
        gop_recons: list of list-of-PIL-Images, one per GOP
        overlap: number of overlapping frames between adjacent GOPs

    Returns:
        stitched: list of PIL Images (full video)
    """
    if overlap == 0 or len(gop_recons) == 1:
        result = []
        for recon in gop_recons:
            result.extend(recon)
        return result

    stitched = []
    for g, recon in enumerate(gop_recons):
        if g == 0:
            # First GOP: keep everything except last `overlap` frames
            stitched.extend(recon[:len(recon) - overlap])
        else:
            # Blend overlap zone
            blended = blend_frames(gop_recons[g - 1], recon, overlap)
            stitched.extend(blended)
            # Non-overlapping tail
            if g < len(gop_recons) - 1:
                stitched.extend(recon[overlap:len(recon) - overlap])
            else:
                # Last GOP: keep rest
                stitched.extend(recon[overlap:])

    return stitched


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
    parser = argparse.ArgumentParser(description="T2V Unconditional Compression Experiment")
    parser.add_argument("--data_dir", default=os.path.join(_project_root, "data", "uvg"))
    parser.add_argument("--wan_ckpt", default="./Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--output_dir", default="./t2v_experiment")
    parser.add_argument("--num_frames", type=int, default=33,
                        help="Frames per GOP (must be 4k+1)")
    parser.add_argument("--num_gops", type=int, default=2)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--K", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim_tail", type=int, default=3)
    parser.add_argument("--g_scale", type=float, default=3.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="CFG scale (1.0 = no CFG, recommended for empty prompt)")
    parser.add_argument("--flow_shift", type=float, default=3.0,
                        help="Timestep shift (3.0 for 480p)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Specific UVG sequences to test (default: all)")
    parser.add_argument("--img_dir", default=None,
                        help="Directory of images to use instead of UVG YUV files")
    parser.add_argument("--overlap", type=int, default=0,
                        help="Number of overlapping frames between adjacent GOPs for blending")
    args = parser.parse_args()

    HEIGHT, WIDTH = args.height, args.width
    FPG = args.num_frames
    overlap = args.overlap
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"T2V Unconditional Compression — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Model: Wan2.1 T2V 1.3B (diffusers)")
    print(f"  Prompt: (empty) — pure model prior + codebook")
    print(f"  Resolution: {WIDTH}x{HEIGHT}")
    print(f"  GOPs: {args.num_gops} x {FPG} frames, overlap={overlap}")
    print(f"  M={args.M}, K={args.K}, g_scale={args.g_scale}")
    print(f"  guidance_scale={args.guidance_scale}, flow_shift={args.flow_shift}")
    print(f"  No reference frame — bitrate = codebook only")
    print("=" * 70)

    # ================================================================
    # Find sequences
    # ================================================================
    if args.img_dir:
        all_seqs = [("custom", args.img_dir)]
    else:
        all_seqs = find_uvg_sequences(args.data_dir)
        if args.sequences:
            all_seqs = [(n, p) for n, p in all_seqs if n in args.sequences]
    print(f"\nSequences: {[s[0] for s in all_seqs]}")

    # ================================================================
    # Load model
    # ================================================================
    model = WanT2VDiffusersWrapper(args.wan_ckpt, flow_shift=args.flow_shift)
    model.load("cuda", torch.bfloat16)

    all_seq_results = []

    for seq_idx, (seq_name, seq_path) in enumerate(all_seqs):
        print(f"\n{'='*70}")
        print(f"  [{seq_idx+1}/{len(all_seqs)}] Sequence: {seq_name}  ({datetime.now().strftime('%H:%M:%S')})")
        print(f"{'='*70}")

        seq_dir = out / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        # With overlap, each GOP after the first starts `overlap` frames earlier
        gop_stride = FPG - overlap

        # Load frames — num_gops=0 means use all available frames
        if args.num_gops > 0:
            total_frames_needed = FPG + (args.num_gops - 1) * gop_stride
        else:
            total_frames_needed = 999999  # load all

        if args.img_dir:
            raw_frames = load_image_sequence(seq_path, total_frames_needed)
        else:
            raw_frames = load_yuv420_frames(seq_path, total_frames_needed, start_frame=0)

        frames_resized = resize_frames(raw_frames, WIDTH, HEIGHT)
        del raw_frames

        # Auto-compute GOPs if num_gops=0
        if args.num_gops <= 0:
            num_gops_seq = len(frames_resized) // FPG
        else:
            num_gops_seq = args.num_gops
        print(f"  Loaded {len(frames_resized)} frames, resized to {WIDTH}x{HEIGHT}")
        print(f"  {num_gops_seq} GOPs x {FPG} frames")

        # Split into GOPs (with overlap)
        gops_gt = []
        for g in range(num_gops_seq):
            start = g * gop_stride
            end = start + FPG
            if end > len(frames_resized):
                print(f"  Warning: not enough frames for GOP {g}, skipping")
                break
            gops_gt.append(frames_resized[start:end])
            print(f"  GOP {g}: frames [{start}, {end})")

        # Original video = unique frames only (for fair metric comparison)
        total_unique = FPG + (len(gops_gt) - 1) * gop_stride
        save_video_mp4(frames_resized[:total_unique], seq_dir / "original.mp4")

        gop_results = []
        gop_recons = []  # store all GOP recon frames for stitching

        for g in range(len(gops_gt)):
            print(f"\n  --- {seq_name} GOP {g}/{len(gops_gt)-1}  ({datetime.now().strftime('%H:%M:%S')}) ---")
            gop_frames = gops_gt[g]

            pipe = TurboDDCMWanPipeline(
                model, K=args.K, M=args.M,
                num_steps=args.steps, num_ddim_tail=args.ddim_tail,
                guidance_scale=args.guidance_scale, g_scale=args.g_scale,
                num_frames=FPG,
                height=HEIGHT, width=WIDTH, seed=args.seed,
            )

            # Encode (no reference image for T2V)
            print(f"  Encoding (empty prompt, no ref)...")
            t0 = time.time()
            step_data, _ = pipe.encode(gop_frames, prompt="")
            t_enc = time.time() - t0

            # Decode
            print(f"  Decoding...")
            t0 = time.time()
            frames_recon = pipe.decode(step_data, prompt="")
            t_dec = time.time() - t0

            # Save per-GOP video
            gop_dir = seq_dir / f"gop{g}"
            gop_dir.mkdir(parents=True, exist_ok=True)
            save_video_mp4(frames_recon, gop_dir / "reconstructed.mp4")
            pipe.save_compressed(step_data, str(gop_dir / "codebook.tdcm"))

            gop_recons.append(frames_recon)

            # Per-GOP metrics (on full GOP, before stitching)
            n = min(len(gop_frames), len(frames_recon))
            t_gt = frames_to_tensor(gop_frames[:n])
            t_rec = frames_to_tensor(frames_recon[:n])

            mean_psnr, per_frame_psnr = compute_psnr(t_gt, t_rec)
            mean_msssim = compute_msssim(t_gt, t_rec)
            mean_lpips = compute_lpips(t_gt, t_rec)

            # Bitrate: codebook only (no reference frame)
            codebook_bits = pipe._total_codebook_bits
            codebook_bytes = codebook_bits // 8
            total_pixels = len(gop_frames) * HEIGHT * WIDTH
            bpp = codebook_bits / total_pixels
            duration_s = len(gop_frames) / 16.0
            bitrate_kbps = codebook_bits / duration_s / 1000.0

            result = {
                "sequence": seq_name,
                "gop": g,
                "PSNR_dB": round(mean_psnr, 2),
                "MS_SSIM": round(mean_msssim, 4) if mean_msssim else None,
                "LPIPS": round(mean_lpips, 4),
                "BPP": round(bpp, 6),
                "codebook_bytes": codebook_bytes,
                "ref_bytes": 0,
                "total_bytes": codebook_bytes,
                "bitrate_kbps": round(bitrate_kbps, 2),
                "encode_s": round(t_enc, 1),
                "decode_s": round(t_dec, 1),
                "per_frame_psnr": [round(p, 2) for p in per_frame_psnr.tolist()],
            }
            gop_results.append(result)

            with open(gop_dir / "metrics.json", "w") as fp:
                json.dump(result, fp, indent=2)

            ms_str = f"{mean_msssim:.4f}" if mean_msssim else "N/A"
            print(f"  PSNR={mean_psnr:.2f} dB, MS-SSIM={ms_str}, LPIPS={mean_lpips:.4f}")
            print(f"  Codebook: {codebook_bytes}B, BPP={bpp:.6f}, {bitrate_kbps:.2f} kbps")
            print(f"  Time: enc={t_enc:.0f}s, dec={t_dec:.0f}s")

            del pipe, t_gt, t_rec, step_data
            gc.collect()
            torch.cuda.empty_cache()

        # Stitch GOPs with overlap blending
        all_recon_frames = stitch_gops(gop_recons, overlap)
        del gop_recons

        if overlap > 0 and len(gops_gt) > 1:
            print(f"\n  --- Stitched video ({len(all_recon_frames)} frames, overlap={overlap}) ---")
            # Metrics on stitched video vs original (unique frames)
            gt_unique = frames_resized[:len(all_recon_frames)]
            t_gt_full = frames_to_tensor(gt_unique)
            t_rec_full = frames_to_tensor(all_recon_frames)
            full_psnr, full_per_frame = compute_psnr(t_gt_full, t_rec_full)
            full_msssim = compute_msssim(t_gt_full, t_rec_full)
            full_lpips = compute_lpips(t_gt_full, t_rec_full)
            ms_str = f"{full_msssim:.4f}" if full_msssim else "N/A"
            print(f"  Stitched PSNR={full_psnr:.2f} dB, MS-SSIM={ms_str}, LPIPS={full_lpips:.4f}")

            # Total bitrate for stitched video
            total_cb_bits = sum(r["codebook_bytes"] for r in gop_results) * 8
            total_pixels_full = len(all_recon_frames) * HEIGHT * WIDTH
            full_bpp = total_cb_bits / total_pixels_full
            full_dur = len(all_recon_frames) / 16.0
            full_kbps = total_cb_bits / full_dur / 1000.0
            print(f"  Stitched BPP={full_bpp:.6f}, {full_kbps:.2f} kbps")

            del t_gt_full, t_rec_full

        # Save full reconstructed video
        if all_recon_frames:
            save_video_mp4(all_recon_frames, seq_dir / "reconstructed_full.mp4")

        # Per-sequence average
        if gop_results:
            avg_psnr = np.mean([r["PSNR_dB"] for r in gop_results])
            avg_msssim_vals = [r["MS_SSIM"] for r in gop_results if r["MS_SSIM"] is not None]
            avg_msssim = float(np.mean(avg_msssim_vals)) if avg_msssim_vals else None
            avg_lpips = np.mean([r["LPIPS"] for r in gop_results])
            avg_bpp = np.mean([r["BPP"] for r in gop_results])
            avg_kbps = np.mean([r["bitrate_kbps"] for r in gop_results])
            total_bytes = sum(r["total_bytes"] for r in gop_results)

            all_seq_results.append({
                "sequence": seq_name,
                "gop_results": gop_results,
                "avg_PSNR": round(float(avg_psnr), 2),
                "avg_MS_SSIM": round(avg_msssim, 4) if avg_msssim else None,
                "avg_LPIPS": round(float(avg_lpips), 4),
                "avg_BPP": round(float(avg_bpp), 6),
                "avg_bitrate_kbps": round(float(avg_kbps), 2),
                "total_bytes": total_bytes,
            })

        del frames_resized, all_recon_frames
        gc.collect()

    # ================================================================
    # Summary
    # ================================================================
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if not all_seq_results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY — T2V Unconditional, M={args.M}, Codebook-only bitrate")
    print(f"{'='*70}")

    hdr = "{:<12} {:>4} {:>8} {:>9} {:>8} {:>8} {:>10}"
    row = "{:<12} {:>4} {:>7.2f} {:>9} {:>7.4f} {:>8} {:>10.2f}"
    print(hdr.format('Seq', 'GOP', 'PSNR', 'MS-SSIM', 'LPIPS', 'Bytes', 'kbps'))
    print("-" * 64)

    for sr in all_seq_results:
        for r in sr["gop_results"]:
            ms = "{:.4f}".format(r['MS_SSIM']) if r['MS_SSIM'] else "N/A"
            print(row.format(r['sequence'], r['gop'], r['PSNR_dB'], ms,
                             r['LPIPS'], r['total_bytes'], r['bitrate_kbps']))

    print("-" * 64)
    overall_psnr = np.mean([sr["avg_PSNR"] for sr in all_seq_results])
    overall_msssim_vals = [sr["avg_MS_SSIM"] for sr in all_seq_results if sr["avg_MS_SSIM"] is not None]
    overall_msssim = float(np.mean(overall_msssim_vals)) if overall_msssim_vals else None
    overall_lpips = np.mean([sr["avg_LPIPS"] for sr in all_seq_results])
    overall_bpp = np.mean([sr["avg_BPP"] for sr in all_seq_results])
    overall_kbps = np.mean([sr["avg_bitrate_kbps"] for sr in all_seq_results])
    ms_overall = f"{overall_msssim:.4f}" if overall_msssim else "N/A"
    print(f"{'OVERALL':<12} {'':>4} {overall_psnr:>7.2f} {ms_overall:>9} {overall_lpips:>7.4f}")
    print(f"  BPP={overall_bpp:.6f}, {overall_kbps:.2f} kbps")

    codebook_bytes = all_seq_results[0]["gop_results"][0]["codebook_bytes"]
    print(f"\n  Codebook per GOP: {codebook_bytes} bytes")
    print(f"  No reference frame (T2V unconditional)")

    # Save summary
    summary = {
        "experiment": "T2V Unconditional Compression",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": "Wan2.1-T2V-14B",
            "prompt": "(empty)",
            "resolution": f"{WIDTH}x{HEIGHT}",
            "num_gops": args.num_gops,
            "frames_per_gop": FPG,
            "M": args.M, "K": args.K,
            "steps": args.steps,
            "ddim_tail": args.ddim_tail,
            "g_scale": args.g_scale,
            "guidance_scale": args.guidance_scale,
            "flow_shift": args.flow_shift,
            "ref_frame": "none",
        },
        "sequences": all_seq_results,
        "overall": {
            "PSNR_dB": round(float(overall_psnr), 2),
            "MS_SSIM": round(float(overall_msssim), 4) if overall_msssim else None,
            "LPIPS": round(float(overall_lpips), 4),
            "BPP": round(float(overall_bpp), 6),
            "bitrate_kbps": round(float(overall_kbps), 2),
        },
    }
    with open(out / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"  Summary: {out}/summary.json")


if __name__ == "__main__":
    main()
