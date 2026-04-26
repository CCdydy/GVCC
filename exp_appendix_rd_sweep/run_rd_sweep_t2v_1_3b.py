"""
run_rd_sweep_t2v_1_3b.py — T2V 1.3B R-D Sweep (M × K joint scaling)

Traces a rate-distortion curve by proportionally scaling (M, K) on
UVG 720p with Wan2.1-T2V-1.3B. Lightweight enough for 32GB VRAM.

Usage:
    python run_rd_sweep_t2v_1_3b.py
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
from sde_rf_wan.wan_t2v_wrapper import WanT2VDiffusersWrapper
from sde_rf_wan.turbo_pipeline import TurboDDCMWanPipeline


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


def resize_frames(frames, target_w, target_h):
    return [f.resize((target_w, target_h), Image.LANCZOS) for f in frames]


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
    parser = argparse.ArgumentParser(description="T2V 1.3B R-D Sweep (M x K)")
    parser.add_argument("--data_dir", default=os.path.join(_project_root, "data", "uvg"))
    parser.add_argument("--wan_ckpt", default=os.path.join(
        _project_root, "checkpoints", "Wan2.1-T2V-1.3B-Diffusers"))
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--num_gops", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ddim_tail", type=int, default=3)
    parser.add_argument("--g_scale", type=float, default=3.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--flow_shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Specific sequences (default: all)")
    args = parser.parse_args()

    HEIGHT, WIDTH = args.height, args.width
    FPG = args.num_frames
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # R-D operating points: (M, K) proportional joint scaling
    # bits_per_atom = ceil(log2(K)) + 1 (index + sign)
    # T2V has no boundary frames, bitrate = codebook only
    # ================================================================
    operating_points = [
        # (M,   K)       bits/atom  description
        (8,    256),     # 9        ultra-low
        (16,   512),     # 11       very low
        (16,  1024),     # 12       low
        (32,  2048),     # 13       low-mid
        (32,  4096),     # 13       mid-low
        (64,  4096),     # 13       mid
        (64, 16384),     # 15       baseline
        (128,16384),     # 15       mid-high
        (128,65536),     # 17       high
        (256,16384),     # 15       very high
        (256,65536),     # 17       ultra-high
    ]

    T_sde = args.steps - args.ddim_tail
    F_lat = (FPG - 1) // 4 + 1
    total_pixels = FPG * HEIGHT * WIDTH * 3

    print("=" * 70)
    print(f"T2V 1.3B R-D Sweep — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Resolution: {WIDTH}x{HEIGHT}, {args.num_gops} GOPs, {FPG} frames/GOP")
    print(f"  T_sde={T_sde}, F_lat={F_lat}")
    print(f"  Model: {args.wan_ckpt}")
    print(f"\n  {'M':>4} {'K':>6} {'bit/atom':>9} {'CB bits':>10} {'CB BPP':>10}")
    print("  " + "-" * 46)
    for M, K in operating_points:
        bpa = math.ceil(math.log2(K)) + 1
        cb_bits = T_sde * F_lat * M * bpa
        cb_bpp = cb_bits / total_pixels
        print(f"  {M:>4} {K:>6} {bpa:>9} {cb_bits:>10} {cb_bpp:>10.6f}")
    print("=" * 70)

    # ================================================================
    # Load sequences & model
    # ================================================================
    from uvg_data import find_uvg_sequences
    all_seqs = find_uvg_sequences(args.data_dir)
    if args.sequences:
        all_seqs = [(n, p) for n, p in all_seqs if n in args.sequences]
    print(f"\nSequences: {[s[0] for s in all_seqs]}")

    print(f"\nLoading model from {args.wan_ckpt}...")
    model = WanT2VDiffusersWrapper(args.wan_ckpt, flow_shift=args.flow_shift)
    model.load("cuda", torch.bfloat16)

    all_results = []

    for M, K in operating_points:
        bpa = math.ceil(math.log2(K)) + 1
        print(f"\n{'='*70}")
        print(f"  Config: M={M}, K={K} ({bpa} bits/atom)")
        print(f"  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}")

        seq_psnrs = []
        seq_msssims = []
        seq_lpips_vals = []
        seq_bpps = []
        seq_kbps = []

        for seq_idx, (seq_name, yuv_path) in enumerate(all_seqs):
            print(f"\n  [{seq_idx+1}/{len(all_seqs)}] {seq_name}")

            raw_frames = load_yuv420_frames(yuv_path, FPG * args.num_gops)
            frames_resized = resize_frames(raw_frames, WIDTH, HEIGHT)
            del raw_frames

            num_gops_seq = min(args.num_gops, len(frames_resized) // FPG)
            gop_psnrs = []
            gop_bpps = []
            gop_kbps_list = []

            for g in range(num_gops_seq):
                start = g * FPG
                gop_frames = frames_resized[start:start + FPG]

                pipe = TurboDDCMWanPipeline(
                    model, K=K, M=M,
                    num_steps=args.steps, num_ddim_tail=args.ddim_tail,
                    guidance_scale=args.guidance_scale, g_scale=args.g_scale,
                    num_frames=FPG,
                    height=HEIGHT, width=WIDTH, seed=args.seed,
                )

                t0 = time.time()
                step_data, _ = pipe.encode(gop_frames, prompt="")
                t_enc = time.time() - t0

                t0 = time.time()
                frames_recon = pipe.decode(step_data, prompt="")
                t_dec = time.time() - t0

                n = min(len(gop_frames), len(frames_recon))
                t_gt = frames_to_tensor(gop_frames[:n])
                t_rec = frames_to_tensor(frames_recon[:n])
                psnr = compute_psnr(t_gt, t_rec)
                msssim = compute_msssim(t_gt, t_rec)
                lpips_val = compute_lpips(t_gt, t_rec)

                codebook_bits = pipe._total_codebook_bits
                bpp = codebook_bits / (n * HEIGHT * WIDTH * 3)
                kbps = codebook_bits / (n / 16.0) / 1000.0

                gop_psnrs.append(psnr)
                gop_bpps.append(bpp)
                gop_kbps_list.append(kbps)

                ms_str = f"{msssim:.4f}" if msssim else "N/A"
                print(f"    GOP {g}: PSNR={psnr:.2f}, MS-SSIM={ms_str}, "
                      f"LPIPS={lpips_val:.4f}, BPP={bpp:.6f}, "
                      f"{kbps:.1f} kbps (enc={t_enc:.0f}s, dec={t_dec:.0f}s)")

                del pipe, frames_recon, t_gt, t_rec, step_data
                gc.collect()
                torch.cuda.empty_cache()

            avg_psnr = sum(gop_psnrs) / len(gop_psnrs)
            avg_bpp = sum(gop_bpps) / len(gop_bpps)
            avg_kbps = sum(gop_kbps_list) / len(gop_kbps_list)
            seq_psnrs.append(avg_psnr)
            seq_bpps.append(avg_bpp)
            seq_kbps.append(avg_kbps)

            del frames_resized
            gc.collect()

        overall_psnr = sum(seq_psnrs) / len(seq_psnrs)
        overall_bpp = sum(seq_bpps) / len(seq_bpps)
        overall_kbps = sum(seq_kbps) / len(seq_kbps)

        result = {
            "M": M, "K": K,
            "bits_per_atom": bpa,
            "PSNR_dB": round(overall_psnr, 2),
            "BPP": round(overall_bpp, 6),
            "bitrate_kbps": round(overall_kbps, 1),
            "per_seq_PSNR": {s[0]: round(p, 2) for s, p in zip(all_seqs, seq_psnrs)},
            "per_seq_BPP": {s[0]: round(b, 6) for s, b in zip(all_seqs, seq_bpps)},
        }
        all_results.append(result)

        print(f"\n  >> M={M}, K={K}: PSNR={overall_psnr:.2f} dB, "
              f"BPP={overall_bpp:.6f}, {overall_kbps:.1f} kbps")

    # ================================================================
    # Save
    # ================================================================
    del model
    gc.collect()
    torch.cuda.empty_cache()

    summary = {
        "experiment": "T2V 1.3B R-D Sweep (M x K)",
        "timestamp": datetime.now().isoformat(),
        "resolution": f"{WIDTH}x{HEIGHT}",
        "num_gops": args.num_gops,
        "frames_per_gop": FPG,
        "steps": args.steps,
        "ddim_tail": args.ddim_tail,
        "g_scale": args.g_scale,
        "flow_shift": args.flow_shift,
        "sequences": [s[0] for s in all_seqs],
        "results": all_results,
    }
    with open(out / "rd_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)

    fields = ["M", "K", "bits_per_atom", "PSNR_dB", "BPP", "bitrate_kbps"]
    with open(out / "rd_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*70}")
    print("R-D SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"{'M':>4} {'K':>6} {'bit/atom':>9} {'PSNR':>8} {'BPP':>10} {'kbps':>8}")
    print("-" * 50)
    for r in sorted(all_results, key=lambda x: x['BPP']):
        print(f"{r['M']:>4} {r['K']:>6} {r['bits_per_atom']:>9} "
              f"{r['PSNR_dB']:>7.2f} {r['BPP']:>10.6f} {r['bitrate_kbps']:>7.1f}")

    print(f"\nSaved: {out}/rd_sweep.json, {out}/rd_sweep.csv")


if __name__ == "__main__":
    main()
