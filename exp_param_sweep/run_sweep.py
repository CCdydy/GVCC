"""
run_sweep.py — Parameter Sweep for T2V Compression

Sweeps M, K, num_frames, steps, g_scale one-at-a-time on a single UVG
sequence (default: Beauty) at 720p to find optimal parameters.

Each sweep varies one parameter while keeping others at baseline defaults.
Results are saved per-run and as a combined CSV/JSON summary.
"""

import sys
import os
import re
import time
import json
import gc
import itertools
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
from uvg_data import find_uvg_sequence as find_uvg_sequence_shared


# ==================================================================
# UVG Loading (same as run_t2v_experiment.py)
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


def find_uvg_sequence(data_dir, seq_name):
    """Find a specific UVG sequence by name."""
    return find_uvg_sequence_shared(data_dir, seq_name)


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
# Single Run
# ==================================================================

def run_single(model, gt_frames, height, width, num_frames, M, K, steps,
               ddim_tail, g_scale, guidance_scale, seed):
    """Run one encode/decode and return metrics dict."""
    pipe = TurboDDCMWanPipeline(
        model, K=K, M=M,
        num_steps=steps, num_ddim_tail=ddim_tail,
        guidance_scale=guidance_scale, g_scale=g_scale,
        num_frames=num_frames,
        height=height, width=width, seed=seed,
    )

    t0 = time.time()
    step_data, _ = pipe.encode(gt_frames[:num_frames], prompt="")
    t_enc = time.time() - t0

    t0 = time.time()
    frames_recon = pipe.decode(step_data, prompt="")
    t_dec = time.time() - t0

    n = min(len(gt_frames[:num_frames]), len(frames_recon))
    t_gt = frames_to_tensor(gt_frames[:n])
    t_rec = frames_to_tensor(frames_recon[:n])

    psnr = compute_psnr(t_gt, t_rec)
    msssim = compute_msssim(t_gt, t_rec)
    lpips_val = compute_lpips(t_gt, t_rec)

    codebook_bits = pipe._total_codebook_bits
    total_pixels = n * height * width * 3
    bpp = codebook_bits / total_pixels

    del pipe, t_gt, t_rec, step_data, frames_recon
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "PSNR_dB": round(psnr, 2),
        "MS_SSIM": round(msssim, 4) if msssim else None,
        "LPIPS": round(lpips_val, 4),
        "BPP": round(bpp, 6),
        "codebook_bytes": codebook_bits // 8,
        "encode_s": round(t_enc, 1),
        "decode_s": round(t_dec, 1),
    }


# ==================================================================
# Main
# ==================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="T2V Parameter Sweep")
    parser.add_argument("--data_dir", default=os.path.join(_project_root, "data"))
    parser.add_argument("--wan_ckpt", default="../exp_t2v/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--sequence", default="Beauty",
                        help="UVG sequence to use for sweep")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--flow_shift", type=float, default=3.0)
    parser.add_argument("--ddim_tail", type=int, default=3)

    # Sweep ranges (comma-separated values override defaults)
    parser.add_argument("--sweep_M", default="16,32,64,128,256",
                        help="M values to sweep (comma-separated)")
    parser.add_argument("--sweep_K", default="1024,4096,16384,65536",
                        help="K values to sweep (comma-separated)")
    parser.add_argument("--sweep_num_frames", default="17,33,49",
                        help="num_frames values to sweep (comma-separated, must be 4k+1)")
    parser.add_argument("--sweep_steps", default="5,10,15,20,30",
                        help="steps values to sweep (comma-separated)")
    parser.add_argument("--sweep_g_scale", default="1.0,2.0,3.0,5.0,8.0",
                        help="g_scale values to sweep (comma-separated)")

    # Baselines (used when a parameter is not being swept)
    parser.add_argument("--base_M", type=int, default=64)
    parser.add_argument("--base_K", type=int, default=16384)
    parser.add_argument("--base_num_frames", type=int, default=33)
    parser.add_argument("--base_steps", type=int, default=20)
    parser.add_argument("--base_g_scale", type=float, default=3.0)

    # Which sweeps to run (default: all)
    parser.add_argument("--sweeps", nargs="*",
                        default=["M", "K", "num_frames", "steps", "g_scale"],
                        help="Which parameters to sweep (default: all)")

    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Parse sweep values
    sweep_vals = {
        "M": [int(x) for x in args.sweep_M.split(",")],
        "K": [int(x) for x in args.sweep_K.split(",")],
        "num_frames": [int(x) for x in args.sweep_num_frames.split(",")],
        "steps": [int(x) for x in args.sweep_steps.split(",")],
        "g_scale": [float(x) for x in args.sweep_g_scale.split(",")],
    }
    baselines = {
        "M": args.base_M,
        "K": args.base_K,
        "num_frames": args.base_num_frames,
        "steps": args.base_steps,
        "g_scale": args.base_g_scale,
    }

    print("=" * 70)
    print(f"T2V Parameter Sweep — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Sequence: {args.sequence}, Resolution: {args.width}x{args.height}")
    print(f"  Baselines: M={baselines['M']}, K={baselines['K']}, "
          f"frames={baselines['num_frames']}, steps={baselines['steps']}, "
          f"g_scale={baselines['g_scale']}")
    print(f"  Sweeps: {args.sweeps}")
    print("=" * 70)

    # Load sequence — need max frames across all num_frames sweep values
    max_frames = max(sweep_vals["num_frames"]) if "num_frames" in args.sweeps else baselines["num_frames"]
    yuv_path = find_uvg_sequence(args.data_dir, args.sequence)
    if not yuv_path:
        print(f"ERROR: Sequence '{args.sequence}' not found in {args.data_dir}")
        return
    print(f"  Loading {max_frames} frames from {yuv_path}")
    raw_frames = load_yuv420_frames(yuv_path, max_frames)
    gt_frames = resize_frames(raw_frames, args.width, args.height)
    del raw_frames
    print(f"  Loaded {len(gt_frames)} frames at {args.width}x{args.height}")

    # Load model
    print(f"\n  Loading model from {args.wan_ckpt}...")
    model = WanT2VDiffusersWrapper(args.wan_ckpt, flow_shift=args.flow_shift)
    model.load("cuda", torch.bfloat16)

    all_results = []

    for sweep_param in args.sweeps:
        print(f"\n{'='*70}")
        print(f"  SWEEP: {sweep_param}")
        print(f"{'='*70}")

        for val in sweep_vals[sweep_param]:
            # Build params: baseline + override current sweep param
            params = dict(baselines)
            params[sweep_param] = val

            tag = f"{sweep_param}={val}"
            print(f"\n  --- {tag} (M={params['M']}, K={params['K']}, "
                  f"frames={params['num_frames']}, steps={params['steps']}, "
                  f"g_scale={params['g_scale']}) ---")

            result = run_single(
                model, gt_frames,
                height=args.height, width=args.width,
                num_frames=params["num_frames"],
                M=params["M"], K=params["K"],
                steps=params["steps"],
                ddim_tail=args.ddim_tail,
                g_scale=params["g_scale"],
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )

            result["sweep_param"] = sweep_param
            result["sweep_value"] = val
            result["M"] = params["M"]
            result["K"] = params["K"]
            result["num_frames"] = params["num_frames"]
            result["steps"] = params["steps"]
            result["g_scale"] = params["g_scale"]

            all_results.append(result)

            ms_str = f"{result['MS_SSIM']:.4f}" if result['MS_SSIM'] else "N/A"
            print(f"    PSNR={result['PSNR_dB']:.2f} dB, MS-SSIM={ms_str}, "
                  f"LPIPS={result['LPIPS']:.4f}, BPP={result['BPP']:.6f}")
            print(f"    Time: enc={result['encode_s']}s, dec={result['decode_s']}s")

    # ==================================================================
    # Save results
    # ==================================================================
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # JSON
    summary = {
        "experiment": "T2V Parameter Sweep",
        "timestamp": datetime.now().isoformat(),
        "sequence": args.sequence,
        "resolution": f"{args.width}x{args.height}",
        "baselines": baselines,
        "results": all_results,
    }
    json_path = out / "sweep_results.json"
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    # CSV
    csv_path = out / "sweep_results.csv"
    fields = ["sweep_param", "sweep_value", "M", "K", "num_frames", "steps",
              "g_scale", "PSNR_dB", "MS_SSIM", "LPIPS", "BPP",
              "codebook_bytes", "encode_s", "decode_s"]
    with open(csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    # Print summary tables per sweep
    print(f"\n{'='*70}")
    print("SWEEP RESULTS SUMMARY")
    print(f"{'='*70}")

    for sweep_param in args.sweeps:
        rows = [r for r in all_results if r["sweep_param"] == sweep_param]
        if not rows:
            continue
        print(f"\n--- {sweep_param} sweep ---")
        hdr = f"  {'Value':>10}  {'PSNR':>8}  {'MS-SSIM':>9}  {'LPIPS':>8}  {'BPP':>10}  {'Bytes':>8}  {'Enc(s)':>7}  {'Dec(s)':>7}"
        print(hdr)
        print("  " + "-" * 78)
        for r in rows:
            ms = f"{r['MS_SSIM']:.4f}" if r['MS_SSIM'] else "N/A"
            print(f"  {r['sweep_value']:>10}  {r['PSNR_dB']:>7.2f}  {ms:>9}  "
                  f"{r['LPIPS']:>7.4f}  {r['BPP']:>10.6f}  {r['codebook_bytes']:>8}  "
                  f"{r['encode_s']:>7.1f}  {r['decode_s']:>7.1f}")

        # Highlight best PSNR
        best = max(rows, key=lambda r: r["PSNR_dB"])
        print(f"  >> Best: {sweep_param}={best['sweep_value']} → "
              f"PSNR={best['PSNR_dB']:.2f} dB, BPP={best['BPP']:.6f}")

    print(f"\nResults saved to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
