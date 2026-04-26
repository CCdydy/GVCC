# GVCC: Zero-Shot Video Compression via Codebook-Driven Stochastic Rectified Flow

Official open-source release for the paper
**[GVCC: Zero-Shot Video Compression via Codebook-Driven Stochastic Rectified Flow](https://arxiv.org/abs/2603.26571)**
(Ziyue Zeng, Xun Su, Haoyuan Liu, Bingyu Lu, Yui Tatsumi, Hiroshi Watanabe — Waseda University, arXiv 2603.26571).

GVCC is a generative video compression framework built on the multi-atom codebook idea from [Turbo-DDCM](https://arxiv.org/abs/2511.06424), adapted to the [Wan2.1](https://github.com/Wan-Video/Wan2.1) rectified-flow video generation model. No retraining of the generator is required — it is *zero-shot*.

> A qualitative example (UVG `HoneyBee` 720p, I2V configuration) is shipped under [`demo/`](demo/) — original vs. reconstructed at ~800 kbps, 36.2 dB PSNR. See `demo/README.md`.

Three conditioning strategies are compared on UVG-1080p, evaluating the trade-off between side-information cost and reconstruction quality:

| Method    | Side Information               | Bitrate Composition                     |
| --------- | ------------------------------ | --------------------------------------- |
| **T2V**   | None (empty prompt)            | Codebook only                           |
| **I2V**   | GT first frame (free)          | Codebook + tail residual (AR chaining)  |
| **FLF2V** | Compressed first + last frames | Codebook + boundary frames              |

## Method

### Core Idea

Diffusion-based compression replaces the random noise in the reverse diffusion process with codebook-selected noise that steers reconstruction toward the ground truth. Encoder and decoder share the same deterministic seed, so they follow identical sampling trajectories — only the noise indices (+signs) need to be transmitted.

### RF→SDE Conversion (Wan2.1)

Wan2.1 is a rectified-flow (RF) model with linear interpolant `x_t = (1-t)·x₀ + t·ε`. To enable stochastic sampling (required for noise replacement), we convert to an equivalent SDE:

1. **Score from velocity** (Eq.8):  `∇log p_t(x_t) = -[(1-t)·u_t + x_t] / t`
2. **SDE drift** (Eq.7):  `f_t = u_t - (g_t²/2)·∇log p_t(x_t)` with `g_t = scale·t²`
3. **Euler-Maruyama step** (Eq.9):  `x_{t-Δt} = x_t - f_t·Δt + g_t·√Δt·z`

where `z` is the codebook noise (unit variance, combined from M atoms).

### Multi-Atom Codebook (Turbo-DDCM)

At each SDE step, for each latent temporal frame:

1. Generate K i.i.d. Gaussian atoms `z_i ~ N(0,I)` deterministically from seed
2. Compute residual `r = x₀ - x̂₀|t` (MMSE prediction error)
3. Select top-M atoms by `|⟨z_i, r⟩|` (Eq.13)
4. Combine: `z = Σ sign(⟨z_i, r⟩)·z_i`, normalize to unit variance (Eq.10)
5. Transmit: M indices + M sign bits

**Bitstream per frame per step:** `M × (⌈log₂K⌉ + 1)` bits.

## Experiments

### T2V (Text-to-Video) — `exp_t2v/`

**Pure model prior + codebook compression.** Uses Wan2.1 T2V-14B with an empty text prompt — no reference frame, no spatial anchoring. Only codebook bits are transmitted. This establishes the lower bound of what the generative prior alone can achieve.

```text
Bitrate = codebook_bytes
GOP 间独立，无帧传递
```

### I2V (Image-to-Video) — `exp_i2v/`

**Autoregressive compression with tail residual correction.** Uses Wan2.1 I2V-14B. GOP 0 uses the GT first frame as reference (free, not counted in bitrate). GOP k>0 uses the decoded last frame from GOP k-1 as reference (0 extra bytes). Tail latent residual correction (enabled by default) transmits an 8-bit quantized+zlib residual for the last 1 latent frame, improving the decoded last frame quality before it propagates as the next GOP's reference.

```text
GOP 0: ref = GT first frame (free)
GOP k: ref = decoded last frame from GOP k-1 (0 bytes)
Bitrate per GOP = codebook_bytes + tail_residual_bytes
```

### FLF2V (First-Last-Frame-to-Video) — `exp_flf2v/`

**Boundary-frame-conditioned interpolation.** Uses Wan2.1 FLF2V-14B conditioned on both first and last frames (compressed via CompressAI). Consecutive GOPs share boundary frames — the last frame of GOP N is reused as the first frame of GOP N+1.

```text
GOP 0: first=compress(GT[0]),  last=compress(GT[32])  → 33 frames
GOP 1: first=reuse(GOP 0 last), last=compress(GT[64]) → 33 frames
GOP 2: first=reuse(GOP 1 last), last=compress(GT[96]) → 33 frames
Concat: GOP0[0:33] + GOP1[1:33] + GOP2[1:33] → 97 unique frames

Bitrate:
  GOP 0: first_bytes + last_bytes + codebook_bytes
  GOP k: last_bytes + codebook_bytes (first frame reused, 0 extra bytes)
```

## Setup

### 1. Dataset — UVG 1080p

Download UVG 1080p YUV sequences (7 sequences, 1920×1080, 120fps, YUV420) and place under `data/uvg/`:

```bash
mkdir -p data/uvg && cd data/uvg

# Download from Ultra Video Group (~17GB total after extraction)
for name in Beauty Bosphorus HoneyBee Jockey ReadySetGo ShakeNDry YachtRide; do
  wget "https://ultravideo.fi/video/${name}_1920x1080_120fps_420_8bit_YUV_RAW.7z"
done

# Extract (requires: apt install p7zip-full)
7z x "*.7z"
cd ../..
```

The loader scans recursively — `data/uvg/*.yuv`, `data/UVG/*.yuv`, and nested directories all work.

### 2. Model Checkpoints

Three 14B models (~90GB total):

| Experiment | HuggingFace Repo                   | Expected Path                        |
| ---------- | ---------------------------------- | ------------------------------------ |
| I2V        | `Wan-AI/Wan2.1-I2V-14B-720P`      | `exp_i2v/Wan2.1-I2V-14B-720P/`      |
| T2V        | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | `exp_t2v/Wan2.1-T2V-14B-Diffusers/` |
| FLF2V      | `Wan-AI/Wan2.1-FLF2V-14B-720P`    | `exp_flf2v/Wan2.1-FLF2V-14B-720P/`  |

Each path can be a real directory or a symlink. Download scripts:

```bash
bash exp_t2v/download_t2v_14b.sh
bash exp_flf2v/download_flf2v_14b.sh
# I2V: huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir exp_i2v/Wan2.1-I2V-14B-720P
```

### 3. Run Experiments

**720p** runs 3 GOPs (quick benchmark). **1080p** runs all available frames per sequence (`--num_gops 0`, matching DCVC-RT benchmark: 6 sequences × 18 GOPs + ShakeNDry × 9 GOPs).

```bash
# T2V — codebook only, no reference frame
bash exp_t2v/run_t2v.sh           # 720p, 3 GOPs
bash exp_t2v/run_t2v_1080p.sh     # 1080p, full UVG

# I2V — autoregressive, GT first frame + decoded last frame chain
bash exp_i2v/run.sh               # 720p, 3 GOPs
bash exp_i2v/run_1080p.sh         # 1080p, full UVG (all frames)

# FLF2V — CompressAI first+last boundary frames
bash exp_flf2v/run_flf2v.sh       # 720p, 3 GOPs
bash exp_flf2v/run_flf2v_1080p.sh # 1080p, full UVG
```

1080p runs print progress: `[1/7] Sequence: Beauty (HH:MM:SS)`, `GOP 0/17 (HH:MM:SS)`.

### 4. Output Structure

```text
exp_{method}/results_{resolution}/
  summary.json                    # Overall: PSNR, MS-SSIM, LPIPS, BPP, bitrate
  {sequence}/
    original.mp4
    reconstructed_full.mp4        # All GOPs concatenated
    gop{N}/
      metrics.json                # PSNR, MS-SSIM, LPIPS, BPP, per_frame_psnr
      reconstructed.mp4
      codebook.tdcm               # Compressed bitstream
      ref_used.png                # (I2V/FLF2V) Reference frame used
```

## Key Parameters

| Parameter            | 720p       | 1080p      | Description                                   |
| -------------------- | ---------- | ---------- | --------------------------------------------- |
| `M`                  | 64         | 80         | Atoms selected per step (quality knob)        |
| `M_tail`             | 128        | 128        | Atoms for tail latent frames (I2V only)       |
| `K`                  | 16384      | 16384      | Codebook size (atoms per step per frame)      |
| `num_frames`         | 33         | 33         | Frames per GOP (must be 4k+1)                 |
| `num_gops`           | 3          | 0 (all)    | GOPs per sequence (0 = all available frames)  |
| `steps`              | 20         | 20         | Total sampling steps                          |
| `ddim_tail`          | 3          | 3          | Last N steps deterministic (no bits)          |
| `g_scale`            | 3.0        | 3.0        | SDE diffusion coefficient                     |
| `guidance_scale`     | 1.0        | 1.0        | CFG scale (1.0 = no CFG)                      |
| `flow_shift`         | 5.0        | 5.0        | Timestep shift (auto: 3.0 for 480p)           |
| `ref_codec`          | compressai | compressai | Reference frame codec (FLF2V only)            |
| `ref_quality`        | 4          | 4          | CompressAI quality level 1-6 (FLF2V only)     |
| `no_tail_residual`   | false      | false      | Disable tail residual correction (I2V)         |
| `tail_residual_bits` | 8          | 8          | Quantization bits for tail residual (4/8/16)   |
| `seed`               | 42         | 42         | Shared encoder/decoder random seed             |

## Project Structure

```text
data/uvg/                    UVG 1080p YUV sequences

sde_rf_wan/                  Shared core library
  sde_convert.py               RF→SDE: score, drift, Euler-Maruyama
  turbo_codebook.py            Multi-atom codebook: top-M selection, bitstream I/O
  turbo_pipeline.py            Encode/decode pipeline (all experiments)
  ref_codec.py                 Reference frame compression (CompressAI / WebP)
  wan_wrapper.py               I2V model wrapper (Wan2.1 native)
  wan_t2v_wrapper.py           T2V model wrapper (diffusers)
  wan_flf2v_wrapper.py         FLF2V model wrapper (first+last frame)

exp_t2v/                     T2V: codebook-only compression
exp_i2v/                     I2V: autoregressive compression with tail residual
exp_flf2v/                   FLF2V: boundary-frame interpolation
exp_param_sweep/             Parameter sweep (M, K, steps, g_scale)

uvg_data.py                  UVG sequence discovery (recursive scan, alias mapping)
wan/                         Wan2.1 native modules (DiT, VAE, T5, CLIP)
```

## Requirements

- PyTorch >= 2.0 with CUDA
- CompressAI (learned image compression)
- diffusers (T2V / FLF2V experiments)
- pytorch_msssim, lpips (metrics)
- imageio, opencv-python (video I/O)

**VRAM:** 14B models need ~48 GB (720p) to ~70 GB (1080p). DiT is offloaded to CPU during VAE encode/decode to fit in memory.

## Remote Server Setup (Vast.ai)

```bash
cd /workspace
git clone https://github.com/CCdydy/GVCC.git
cd GVCC
bash vast_setup.sh    # installs deps, downloads 3x 14B models (~90GB)
```

Then download UVG data and run experiments as described above.

## References

- [GVCC: Zero-Shot Video Compression via Codebook-Driven Stochastic Rectified Flow](https://arxiv.org/abs/2603.26571) (this paper)
- [Turbo-DDCM: Diffusion-Based Data Compression via Multi-Atom Coding](https://arxiv.org/abs/2511.06424)
- [Wan2.1: Wan Video Generation](https://github.com/Wan-Video/Wan2.1)
- [CompressAI: Neural Image Compression](https://github.com/InterDigitalInc/CompressAI)

## Citation

If you find this work useful, please cite:

```bibtex
@article{zeng2026gvcc,
  title   = {GVCC: Zero-Shot Video Compression via Codebook-Driven Stochastic Rectified Flow},
  author  = {Zeng, Ziyue and Su, Xun and Liu, Haoyuan and Lu, Bingyu and Tatsumi, Yui and Watanabe, Hiroshi},
  journal = {arXiv preprint arXiv:2603.26571},
  year    = {2026}
}
```
