"""
benchmark_ref_codec.py — Compare image codecs for GOP reference frame

Tests WebP vs CompressAI (learned) at various byte budgets.
Goal: find the best codec for the I2V reference frame at ~10-20 KB.
"""

import sys, os, io, time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import re

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from uvg_data import find_uvg_sequence as find_uvg_sequence_shared


def load_first_frame(data_dir, sequence, width, height):
    """Load first frame of a UVG sequence."""
    import cv2
    yuv_path = find_uvg_sequence_shared(data_dir, sequence)
    if yuv_path is not None:
        yuv_file = Path(yuv_path)
        match = re.search(r'(\d+)x(\d+)', yuv_file.name)
        W, H = int(match.group(1)), int(match.group(2))
        frame_size = H * W * 3 // 2
        raw = yuv_file.read_bytes()[:frame_size]
        yuv = np.frombuffer(raw, dtype=np.uint8)
        y = yuv[:H*W].reshape(H, W)
        u = yuv[H*W:H*W+H*W//4].reshape(H//2, W//2)
        v = yuv[H*W+H*W//4:].reshape(H//2, W//2)
        u = cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(np.stack([y, u, v], axis=-1), cv2.COLOR_YUV2RGB)
        img = Image.fromarray(rgb).resize((width, height), Image.LANCZOS)
        return img
    raise FileNotFoundError(f"Sequence {sequence} not found")


def psnr_lpips(orig, recon):
    """Compute PSNR and LPIPS between two PIL images."""
    o = torch.from_numpy(np.array(orig).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    r = torch.from_numpy(np.array(recon).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    mse = ((o - r) ** 2).mean()
    psnr = -10.0 * torch.log10(mse + 1e-10).item()

    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex')
        with torch.no_grad():
            lp = loss_fn(2*o - 1, 2*r - 1).item()
        del loss_fn
    except:
        lp = None

    return psnr, lp


# ==================================================================
# Codecs
# ==================================================================

def test_webp(img, quality):
    """WebP compression. Returns (decoded, num_bytes)."""
    buf = io.BytesIO()
    img.save(buf, format="WebP", quality=quality)
    nbytes = buf.tell()
    buf.seek(0)
    decoded = Image.open(buf).copy()
    return decoded, nbytes


def test_compressai(img, model_name, quality_level, device="cuda"):
    """CompressAI learned compression. Returns (decoded, num_bytes)."""
    from compressai.zoo import models
    net = models[model_name](quality=quality_level, pretrained=True).eval().to(device)

    x = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)

    # Pad to multiple of 64 (required by most models)
    _, _, h, w = x.shape
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        out = net.compress(x)
        rec = net.decompress(out["strings"], out["shape"])

    # Count bytes
    total_bytes = sum(len(s[0]) for s in out["strings"])

    # Remove padding
    x_hat = rec["x_hat"][:, :, :h, :w].clamp(0, 1)
    decoded_np = (x_hat[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    decoded = Image.fromarray(decoded_np)

    del net
    torch.cuda.empty_cache()
    return decoded, total_bytes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(_project_root, "data", "uvg"))
    parser.add_argument("--sequence", default="Beauty")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--output_dir", default="./codec_benchmark")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = load_first_frame(args.data_dir, args.sequence, args.width, args.height)
    img.save(out / "original.png")
    print(f"Loaded {args.sequence} first frame: {args.width}x{args.height}")

    results = []

    # --- WebP sweep ---
    print("\n=== WebP ===")
    for q in [1, 3, 5, 10, 15, 20, 30, 50, 75]:
        dec, nbytes = test_webp(img, q)
        psnr, lp = psnr_lpips(img, dec)
        dec.save(out / f"webp_q{q}.png")
        results.append({"codec": "WebP", "param": f"Q={q}", "bytes": nbytes,
                        "psnr": round(psnr, 2), "lpips": round(lp, 4) if lp else None})
        print(f"  Q={q:3d}: {nbytes:>7d}B  PSNR={psnr:.2f} dB  LPIPS={lp:.4f}")

    # --- CompressAI: cheng2020-attn (best quality) ---
    print("\n=== CompressAI cheng2020-attn ===")
    for ql in range(1, 7):
        try:
            dec, nbytes = test_compressai(img, "cheng2020-attn", ql)
            psnr, lp = psnr_lpips(img, dec)
            dec.save(out / f"cheng2020attn_q{ql}.png")
            results.append({"codec": "cheng2020-attn", "param": f"q={ql}", "bytes": nbytes,
                            "psnr": round(psnr, 2), "lpips": round(lp, 4) if lp else None})
            print(f"  q={ql}: {nbytes:>7d}B  PSNR={psnr:.2f} dB  LPIPS={lp:.4f}")
        except Exception as e:
            print(f"  q={ql}: FAILED — {e}")

    # --- CompressAI: mbt2018-mean (good speed/quality) ---
    print("\n=== CompressAI mbt2018-mean ===")
    for ql in range(1, 7):
        try:
            dec, nbytes = test_compressai(img, "mbt2018-mean", ql)
            psnr, lp = psnr_lpips(img, dec)
            dec.save(out / f"mbt2018mean_q{ql}.png")
            results.append({"codec": "mbt2018-mean", "param": f"q={ql}", "bytes": nbytes,
                            "psnr": round(psnr, 2), "lpips": round(lp, 4) if lp else None})
            print(f"  q={ql}: {nbytes:>7d}B  PSNR={psnr:.2f} dB  LPIPS={lp:.4f}")
        except Exception as e:
            print(f"  q={ql}: FAILED — {e}")

    # --- Summary sorted by bytes ---
    print(f"\n{'='*70}")
    print(f"SUMMARY — sorted by bytes (target: 10-30 KB)")
    print(f"{'='*70}")
    results.sort(key=lambda x: x["bytes"])
    print(f"{'Codec':<22} {'Param':<8} {'Bytes':>8} {'PSNR':>7} {'LPIPS':>7}")
    print("-" * 55)
    for r in results:
        lp = f"{r['lpips']:.4f}" if r['lpips'] else "N/A"
        marker = " <<<" if 8000 <= r['bytes'] <= 30000 else ""
        print(f"{r['codec']:<22} {r['param']:<8} {r['bytes']:>7d}B {r['psnr']:>6.2f} {lp:>7}{marker}")

    import json
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}/results.json")


if __name__ == "__main__":
    main()
