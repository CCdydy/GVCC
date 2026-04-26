# Demo

A single qualitative example for the **I2V** configuration on the UVG `HoneyBee` sequence at 720p.

| File | Description |
|---|---|
| `HoneyBee_720p_original.mp4`            | Ground-truth video (3 GOPs × 33 frames = 97 frames) |
| `HoneyBee_720p_i2v_reconstructed.mp4`   | I2V reconstruction (M=64, K=16384, steps=20, g_scale=3.0) |

Reported metrics for this clip (from the paper's 720p table):

```
PSNR     = 36.21 dB
MS-SSIM  = 0.9854
LPIPS    = 0.0195
Bitrate  ≈ 802 kbps  (BPP ≈ 0.0181)
```

To regenerate this and the full UVG-720p / UVG-1080p results yourself, follow the
instructions in the top-level `README.md` and run:

```bash
bash exp_i2v/run.sh        # 720p, 3 GOPs per sequence (this demo)
bash exp_i2v/run_1080p.sh  # 1080p, all available frames
```
