# Demo

A qualitative example for the **FLF2V** configuration on a self-captured
out-of-distribution (OOD) clip at 720p — a scene that is stylistically
distant from common video-generator training distributions and therefore a
useful stress test for zero-shot generative compression.

| File | Description |
|---|---|
| `OOD_720p_original.mp4`              | Ground-truth video (3 GOPs × 33 frames = 97 unique frames) |
| `OOD_720p_flf2v_reconstructed.mp4`   | FLF2V reconstruction (M=64, K=16384, steps=20, g_scale=3.0) |

Reported metrics for this clip (matching Appendix B of the paper):

```
PSNR     = 25.15 dB
MS-SSIM  = 0.910
LPIPS    = 0.119
ΔE2000   = 2.77
BPP      ≈ 0.0146  (boundary-frame overhead ~66%)
```

The full UVG-720p / UVG-1080p results can be reproduced by following the
top-level `README.md` and:

```bash
bash exp_flf2v/run_flf2v.sh        # 720p, 3 GOPs per sequence
bash exp_flf2v/run_flf2v_1080p.sh  # 1080p, all available frames
```
