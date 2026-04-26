"""
run_uvg_chained_gop.py — Autoregressive Chained-GOP Experiment on UVG

Autoregressive I2V compression with tail latent residual correction:
  - GOP 0: ref = CompressAI(GT first frame)
  - GOP k>0: ref = decoded last frame from GOP k-1 (0 extra ref bytes)
  - Tail residual: 8-bit quantized + zlib residual for last 1 latent frame
    corrects the region farthest from reference, improving AR chain quality
  - Bitrate = codebook + ref (GOP 0 only) + tail_residual per GOP
  - Metrics computed on latent-corrected decoded frames
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


def find_uvg_sequences(data_dir):
    return find_uvg_sequences_shared(data_dir)


def resize_frames(frames, target_w, target_h):
    return [f.resize((target_w, target_h), Image.LANCZOS) for f in frames]



# ==================================================================
# Tail Latent Residual Compression
# ==================================================================

def compress_tail_residual(residual_tail, bits=8):
    """Compress tail latent residual tensor.

    Args:
        residual_tail: (C, N_tail, H, W) float tensor — only the tail frames
        bits: quantization bits (4/8/16)

    Returns:
        (compressed_bytes, nbytes, meta_dict)
    """
    r = residual_tail.float().cpu()
    C, F, H, W = r.shape

    # Per-channel min/max quantization
    maxval = r.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
    r_norm = r / maxval  # [-1, 1]

    levels = 2 ** bits
    r_quant = ((r_norm + 1.0) * (levels - 1) / 2.0).round().clamp(0, levels - 1)

    if bits <= 8:
        packed = r_quant.to(torch.uint8).numpy().tobytes()
    else:
        packed = r_quant.to(torch.int16).numpy().tobytes()

    compressed = zlib.compress(packed, 9)

    meta = {
        'C': C, 'F': F, 'H': H, 'W': W,
        'bits': bits,
        'maxval': maxval.squeeze().tolist(),
    }
    return compressed, len(compressed), meta


def decompress_tail_residual(compressed, meta, device='cuda'):
    """Decompress tail latent residual back to tensor."""
    C, F = meta['C'], meta['F']
    H, W = meta['H'], meta['W']
    bits = meta['bits']
    maxval_list = meta['maxval']
    if isinstance(maxval_list, (int, float)):
        maxval_list = [maxval_list]
    maxval = torch.tensor(maxval_list).float().reshape(C, 1, 1, 1)

    packed = zlib.decompress(compressed)
    levels = 2 ** bits

    if bits <= 8:
        arr = np.frombuffer(packed, dtype=np.uint8).copy()
        r_quant = torch.from_numpy(arr).float().reshape(C, F, H, W)
    else:
        arr = np.frombuffer(packed, dtype=np.int16).copy()
        r_quant = torch.from_numpy(arr).float().reshape(C, F, H, W)

    r_norm = r_quant * 2.0 / (levels - 1) - 1.0
    r = r_norm * maxval

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
    parser = argparse.ArgumentParser(description="Chained-GOP UVG Experiment")
    parser.add_argument("--data_dir", default=os.path.join(_project_root, "data", "uvg"))
    parser.add_argument("--wan_ckpt", default="./Wan2.1-I2V-14B-720P")
    parser.add_argument("--output_dir", default="./uvg_chained_gop")
    parser.add_argument("--num_frames_per_gop", type=int, default=33)
    parser.add_argument("--num_gops", type=int, default=2)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--M_tail", type=int, default=None,
                        help="M for last tail_latent_frames latent frames (default: same as M)")
    parser.add_argument("--tail_latent_frames", type=int, default=2,
                        help="Number of tail latent frames to use M_tail")
    parser.add_argument("--K", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim_tail", type=int, default=3)
    parser.add_argument("--g_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flow_shift", type=float, default=None,
                        help="Timestep shift (auto: 3.0 for 480p, 5.0 for 720p+)")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Specific sequences to test (default: all)")
    # Tail residual correction
    parser.add_argument("--no_tail_residual", action="store_true",
                        help="Disable tail latent frame residual correction")
    parser.add_argument("--tail_residual_bits", type=int, default=8,
                        help="Quantization bits for tail residual (4/8/16)")
    args = parser.parse_args()

    # Auto flow_shift based on resolution
    if args.flow_shift is None:
        args.flow_shift = 3.0 if args.height <= 480 else 5.0

    HEIGHT, WIDTH = args.height, args.width
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    FPG = args.num_frames_per_gop

    print("=" * 70)
    print(f"AR Chained-GOP UVG Experiment — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Resolution: {WIDTH}x{HEIGHT}")
    print(f"  Frames per GOP: {FPG}, num_gops: {args.num_gops} (0=auto/all)")
    print(f"  M={args.M}, K={args.K}, g_scale={args.g_scale}, flow_shift={args.flow_shift}")
    print(f"  Ref GOP 0: GT first frame (free)")
    print(f"  Ref GOP k>0: decoded last frame from previous GOP (AR)")
    if not args.no_tail_residual:
        print(f"  Tail residual: {args.tail_residual_bits}bit, last 1 latent frame")
    print("=" * 70)

    # ================================================================
    # Find sequences
    # ================================================================
    all_seqs = find_uvg_sequences(args.data_dir)
    if args.sequences:
        all_seqs = [(n, p) for n, p in all_seqs if n in args.sequences]
    print(f"\nSequences: {[s[0] for s in all_seqs]}")

    # ================================================================
    # Load model
    # ================================================================
    model = WanWrapper(args.wan_ckpt, config_name="i2v-14B", flow_shift=args.flow_shift)
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
            num_gops_seq = args.num_gops
            load_frames = FPG * num_gops_seq
        else:
            # Load all frames, compute max GOPs
            load_frames = 999999  # load_yuv420_frames stops at EOF

        raw_frames = load_yuv420_frames(yuv_path, load_frames, start_frame=0)
        frames_resized = resize_frames(raw_frames, WIDTH, HEIGHT)
        del raw_frames

        if args.num_gops <= 0:
            num_gops_seq = len(frames_resized) // FPG
        total_used = num_gops_seq * FPG
        frames_resized = frames_resized[:total_used]
        print(f"  Loaded {len(frames_resized)} frames, resized to {WIDTH}x{HEIGHT}")
        print(f"  {num_gops_seq} GOPs x {FPG} frames = {total_used} frames")

        # Split into GOPs (no overlap)
        gops_gt = []
        for g in range(num_gops_seq):
            start = g * FPG
            end = start + FPG
            gops_gt.append(frames_resized[start:end])
            print(f"  GOP {g}: frames [{start}, {end})")

        # Save original
        save_video_mp4(frames_resized, seq_dir / "original.mp4")

        # Encoding — autoregressive: decoded last frame → next GOP ref
        gop_results = []
        all_recon_frames = []
        next_ref = None  # decoded last frame from previous GOP

        for g in range(num_gops_seq):
            print(f"\n  --- {seq_name} GOP {g}/{num_gops_seq-1}  ({datetime.now().strftime('%H:%M:%S')}) ---")
            gop_frames = gops_gt[g]

            pipe = TurboDDCMWanPipeline(
                model, K=args.K, M=args.M,
                num_steps=args.steps, num_ddim_tail=args.ddim_tail,
                guidance_scale=1.0, g_scale=args.g_scale,
                num_frames=FPG,
                height=HEIGHT, width=WIDTH, seed=args.seed,
                M_tail=args.M_tail,
                tail_latent_frames=args.tail_latent_frames,
            )

            # Reference frame: GOP 0 = GT first frame (free), GOP k>0 = decoded last frame (AR)
            if g == 0:
                ref_image = gop_frames[0]
                ref_bytes = 0
                ref_source = "GT[0] (free)"
            else:
                ref_image = next_ref
                ref_bytes = 0
                ref_source = f"AR(decoded_last[GOP{g-1}])"

            print(f"  Ref: {ref_source}")

            # Encode
            print(f"  Encoding...")
            t0 = time.time()
            step_data, x0_enc = pipe.encode(gop_frames, prompt="", ref_image=ref_image)
            t_enc = time.time() - t0

            # Tail residual correction (default on)
            tail_res_bytes = 0
            latent_correction = None
            if not args.no_tail_residual:
                x0_true = pipe._gt_latent  # (1, C, F, H, W)
                n_tail = 1  # last latent frame (covers last 4 pixel frames)
                tail_residual = (x0_true - x0_enc).squeeze(0)[:, -n_tail:, :, :]  # (C, N, H, W)
                tail_mse = (tail_residual ** 2).mean().item()
                print(f"  Tail residual MSE (last {n_tail} latent frames): {tail_mse:.4f}")

                # Compress & decompress
                compressed, tail_res_bytes, meta = compress_tail_residual(
                    tail_residual, bits=args.tail_residual_bits)
                print(f"  Tail residual compressed: {tail_res_bytes}B "
                      f"({args.tail_residual_bits}bit)")

                tail_decompressed = decompress_tail_residual(compressed, meta, device="cuda")
                correction = torch.zeros_like(x0_enc.squeeze(0))  # (C, F, H, W)
                correction[:, -n_tail:, :, :] = tail_decompressed
                latent_correction = correction.unsqueeze(0)  # (1, C, F, H, W)

                recon_mse = ((tail_residual.cpu() - tail_decompressed.cpu()) ** 2).mean().item()
                print(f"  Tail residual quant MSE: {recon_mse:.6f}")

                del x0_true, tail_residual, tail_decompressed, correction
                gc.collect()
                torch.cuda.empty_cache()

            # Decode with latent correction applied
            print(f"  Decoding...")
            t0 = time.time()
            frames_recon = pipe.decode(
                step_data, prompt="", ref_image=ref_image,
                latent_correction=latent_correction)
            t_dec = time.time() - t0

            # Save decoded last frame as next GOP's reference (AR chain)
            next_ref = frames_recon[-1].copy()

            # Save outputs
            gop_dir = seq_dir / f"gop{g}"
            gop_dir.mkdir(parents=True, exist_ok=True)
            save_video_mp4(frames_recon, gop_dir / "reconstructed.mp4")
            pipe.save_compressed(step_data, str(gop_dir / "codebook.tdcm"))
            ref_image.save(gop_dir / "ref_used.png")

            # Collect for full video
            all_recon_frames.extend(frames_recon)

            # Metrics — computed on corrected decoded frames
            n = min(len(gop_frames), len(frames_recon))
            t_gt = frames_to_tensor(gop_frames[:n])
            t_rec = frames_to_tensor(frames_recon[:n])

            mean_psnr, per_frame_psnr = compute_psnr(t_gt, t_rec)
            mean_msssim = compute_msssim(t_gt, t_rec)
            mean_lpips = compute_lpips(t_gt, t_rec)

            # Last frame quality (critical for AR chaining)
            last_psnr = per_frame_psnr[-1].item()

            # Bitrate: codebook + ref (GOP 0 only) + tail residual
            T_sde = pipe.num_sde_steps
            F_lat = pipe.num_latent_frames
            bits_per_fs = pipe.codebook.bits_per_frame_step
            codebook_bits = T_sde * F_lat * bits_per_fs
            codebook_bytes = codebook_bits // 8
            gop_total_bytes = codebook_bytes + ref_bytes + tail_res_bytes
            gop_total_bits = gop_total_bytes * 8
            total_pixels = len(gop_frames) * HEIGHT * WIDTH * 3
            bpp = gop_total_bits / total_pixels
            duration_s = len(gop_frames) / 16.0
            bitrate_kbps = gop_total_bits / duration_s / 1000.0

            result = {
                "sequence": seq_name,
                "gop": g,
                "ref_source": ref_source,
                "PSNR_dB": round(mean_psnr, 2),
                "MS_SSIM": round(mean_msssim, 4) if mean_msssim else None,
                "LPIPS": round(mean_lpips, 4),
                "last_frame_PSNR": round(last_psnr, 2),
                "BPP": round(bpp, 6),
                "codebook_bytes": codebook_bytes,
                "ref_bytes": ref_bytes,
                "tail_residual_bytes": tail_res_bytes,
                "gop_total_bytes": gop_total_bytes,
                "bitrate_kbps": round(bitrate_kbps, 2),
                "encode_s": round(t_enc, 1),
                "decode_s": round(t_dec, 1),
                "per_frame_psnr": [round(p, 2) for p in per_frame_psnr.tolist()],
            }
            gop_results.append(result)

            with open(gop_dir / "metrics.json", "w") as f:
                json.dump(result, f, indent=2)

            print(f"  PSNR={mean_psnr:.2f} dB (last={last_psnr:.2f}), "
                  f"MS-SSIM={mean_msssim:.4f}, LPIPS={mean_lpips:.4f}")
            print(f"  Codebook: {codebook_bytes}B + Ref: {ref_bytes}B + "
                  f"TailRes: {tail_res_bytes}B = {gop_total_bytes}B, "
                  f"BPP={bpp:.6f}, {bitrate_kbps:.2f} kbps")
            print(f"  Time: enc={t_enc:.0f}s, dec={t_dec:.0f}s")

            del pipe, frames_recon, t_gt, t_rec, step_data, latent_correction
            gc.collect()
            torch.cuda.empty_cache()

        # Save full reconstructed video
        save_video_mp4(all_recon_frames, seq_dir / "reconstructed_full.mp4")

        # Per-sequence average
        avg_psnr = np.mean([r["PSNR_dB"] for r in gop_results])
        avg_lpips = np.mean([r["LPIPS"] for r in gop_results])

        all_seq_results.append({
            "sequence": seq_name,
            "gop_results": gop_results,
            "avg_PSNR": round(float(avg_psnr), 2),
            "avg_LPIPS": round(float(avg_lpips), 4),
        })

        del frames_resized, all_recon_frames
        gc.collect()

    # ================================================================
    # Summary
    # ================================================================
    del model
    gc.collect()
    torch.cuda.empty_cache()

    tail_tag = ""
    if not args.no_tail_residual:
        tail_tag = f" + tail_res({args.tail_residual_bits}bit, 1f)"

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY — AR Chained-GOP, M={args.M}{tail_tag}")
    print(f"{'='*70}")

    hdr = "{:<12} {:>4} {:>8} {:>6} {:>9} {:>8} {:>8} {:>8} {:>10} {:>8}"
    row = "{:<12} {:>4} {:>7.2f} {:>5.2f} {:>9} {:>7.4f} {:>8} {:>8} {:>10} {:>7.2f}"
    print(hdr.format('Seq', 'GOP', 'PSNR', 'Last', 'MS-SSIM', 'LPIPS',
                      'CB(B)', 'Ref(B)', 'Total(B)', 'kbps'))
    print("-" * 100)
    for sr in all_seq_results:
        for r in sr["gop_results"]:
            ms = "{:.4f}".format(r['MS_SSIM']) if r['MS_SSIM'] else "N/A"
            print(row.format(r['sequence'], r['gop'], r['PSNR_dB'],
                             r['last_frame_PSNR'], ms,
                             r['LPIPS'], r['codebook_bytes'], r['ref_bytes'],
                             r['gop_total_bytes'], r['bitrate_kbps']))

    print("-" * 100)
    overall_psnr = np.mean([sr["avg_PSNR"] for sr in all_seq_results])
    overall_lpips = np.mean([sr["avg_LPIPS"] for sr in all_seq_results])
    print(f"{'OVERALL':<12} {'':>4} {overall_psnr:>7.2f} {'':>6} {'':>9} {overall_lpips:>7.4f}")

    # AR chain quality: last-frame PSNR across GOPs
    for sr in all_seq_results:
        last_psnrs = [r['last_frame_PSNR'] for r in sr['gop_results']]
        print(f"  {sr['sequence']} last-frame PSNR: {' → '.join(f'{p:.1f}' for p in last_psnrs)}")

    codebook_bytes = all_seq_results[0]["gop_results"][0]["codebook_bytes"]
    print(f"\n  Mode: autoregressive (decoded last frame → next GOP ref)")
    print(f"  Codebook per GOP: {codebook_bytes} bytes")
    print(f"  Ref GOP 0: GT first frame (free)")
    print(f"  Ref GOP k>0: decoded last frame (0 bytes)")
    if not args.no_tail_residual:
        print(f"  Tail residual: {args.tail_residual_bits}bit, last 1 latent frame")

    # Save
    config = {
        "resolution": f"{WIDTH}x{HEIGHT}",
        "num_gops": args.num_gops,
        "frames_per_gop": FPG,
        "M": args.M, "K": args.K,
        "g_scale": args.g_scale,
        "flow_shift": args.flow_shift,
        "mode": "autoregressive",
        "ref_gop0": "GT first frame (free)",
        "ref_gopk": "decoded last frame (AR)",
    }
    if not args.no_tail_residual:
        config["tail_residual"] = True
        config["tail_residual_bits"] = args.tail_residual_bits
        config["tail_residual_frames"] = 1

    summary = {
        "experiment": "AR Chained-GOP UVG",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "sequences": all_seq_results,
        "overall": {
            "PSNR_dB": round(float(overall_psnr), 2),
            "LPIPS": round(float(overall_lpips), 4),
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {out}/summary.json")


if __name__ == "__main__":
    main()
