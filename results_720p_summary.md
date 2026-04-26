# 720p Results Summary — T2V vs I2V vs FLF2V

UVG dataset, 1280×720, 3 GOPs per sequence, 33 frames/GOP.

## Config

| Parameter | T2V | I2V | FLF2V |
|---|---|---|---|
| Model | Wan2.1-T2V-14B | Wan2.1-I2V-14B | Wan2.1-FLF2V-14B |
| M | 64 | 64 (M_tail=128) | 64 |
| K | 16384 | 16384 | 16384 |
| Steps | 20 | 20 | 20 |
| g_scale | 3.0 | 3.0 | 3.0 |
| Side info | None (empty prompt) | CompressAI ref frame (q=4) + tail residual (8bit) | CompressAI first+last frames (q=4) |

## Overall

| Method | PSNR (dB) | LPIPS | Bitrate (kbps) | BPP |
|---|---|---|---|---|
| T2V | 29.01 | 0.1171 | 71.2 | 0.001610 |
| I2V | 31.97 | 0.0739 | 801.5 | 0.018 |
| FLF2V | 31.40 | 0.0964 | 192.8 | ~0.004 |

## PSNR (dB)

| Sequence | T2V | I2V | FLF2V | I2V−T2V | FLF2V−T2V |
|---|---|---|---|---|---|
| Beauty | 31.79 | 32.90 | 32.26 | +1.11 | +0.47 |
| Bosphorus | 30.32 | 33.63 | 32.72 | +3.31 | +2.40 |
| HoneyBee | 30.78 | 36.21 | 34.73 | +5.43 | +3.95 |
| Jockey | 31.28 | 33.27 | 33.09 | +1.99 | +1.81 |
| ReadySteadyGo | 26.74 | 29.48 | 29.44 | +2.74 | +2.70 |
| ShakeNDry | 25.55 | 30.11 | 29.52 | +4.56 | +3.97 |
| YachtRide | 26.60 | 28.21 | 28.02 | +1.61 | +1.42 |
| **Overall** | **29.01** | **31.97** | **31.40** | **+2.96** | **+2.39** |

## MS-SSIM

| Sequence | T2V | I2V | FLF2V |
|---|---|---|---|
| Beauty | 0.9016 | 0.9223 | 0.9133 |
| Bosphorus | 0.9153 | 0.9628 | 0.9517 |
| HoneyBee | 0.9517 | 0.9854 | 0.9791 |
| Jockey | 0.9298 | 0.9550 | 0.9529 |
| ReadySteadyGo | 0.9287 | 0.9591 | 0.9599 |
| ShakeNDry | 0.7544 | 0.8938 | 0.8811 |
| YachtRide | 0.8923 | 0.9264 | 0.9218 |

## LPIPS (lower = better)

| Sequence | T2V | I2V | FLF2V |
|---|---|---|---|
| Beauty | 0.1535 | 0.1088 | 0.1581 |
| Bosphorus | 0.0976 | 0.0554 | 0.0861 |
| HoneyBee | 0.0516 | 0.0195 | 0.0328 |
| Jockey | 0.0899 | 0.0698 | 0.0800 |
| ReadySteadyGo | 0.0858 | 0.0528 | 0.0634 |
| ShakeNDry | 0.2389 | 0.1362 | 0.1660 |
| YachtRide | 0.1025 | 0.0749 | 0.0885 |
| **Overall** | **0.1171** | **0.0739** | **0.0964** |

## Bitrate (kbps) & BPP

| Sequence | T2V kbps | I2V kbps | FLF2V kbps | T2V BPP | I2V BPP | FLF2V BPP |
|---|---|---|---|---|---|---|
| Beauty | 71.2 | 836.5 | 111.4 | 0.001610 | 0.018909 | 0.002518 |
| Bosphorus | 71.2 | 806.6 | 158.0 | 0.001610 | 0.018233 | 0.003573 |
| HoneyBee | 71.2 | 801.9 | 209.4 | 0.001610 | 0.018127 | 0.004733 |
| Jockey | 71.2 | 789.2 | 152.7 | 0.001610 | 0.017841 | 0.003453 |
| ReadySteadyGo | 71.2 | 760.8 | 246.8 | 0.001610 | 0.017198 | 0.005580 |
| ShakeNDry | 71.2 | 808.5 | 262.8 | 0.001610 | 0.018276 | 0.005940 |
| YachtRide | 71.2 | 807.3 | 208.7 | 0.001610 | 0.018250 | 0.004718 |
| **Overall** | **71.2** | **801.5** | **192.8** | | | |

## I2V Bitrate Composition

| Sequence | Codebook (B) | Ref (B) | Tail Residual (B) | Total (B) |
|---|---|---|---|---|
| Beauty | 18,360 | 0 | 197,293 | 215,653 |
| Bosphorus | 18,360 | 0 | 189,587 | 207,947 |
| HoneyBee | 18,360 | 0 | 188,373 | 206,733 |
| Jockey | 18,360 | 0 | 185,115 | 203,475 |
| ReadySteadyGo | 18,360 | 0 | 177,782 | 196,142 |
| ShakeNDry | 18,360 | 0 | 190,072 | 208,432 |
| YachtRide | 18,360 | 0 | 189,782 | 208,142 |

Tail residual (8bit quantized, 1 latent frame) dominates I2V bitrate at ~190KB/GOP (~90% of total).

## Rate-Distortion Efficiency

| Method | PSNR | BPP | Extra BPP vs T2V | Extra PSNR vs T2V | Marginal dB/BPP |
|---|---|---|---|---|---|
| T2V | 29.01 | 0.00161 | — | — | — |
| FLF2V | 31.40 | 0.00400 | +0.00239 | +2.39 dB | 1000 |
| I2V | 31.97 | 0.01800 | +0.01639 | +2.96 dB | 181 |

FLF2V achieves 81% of I2V's PSNR gain at only 15% of I2V's extra bitrate cost.
