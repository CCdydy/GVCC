# Pipeline Design Document

This document describes the engineering design of the Turbo-DDCM video compression pipeline for each of the three experiment tracks. It is intended as a companion to the paper, focusing on implementation rationale and system-level design decisions.

## 1. Shared Core: RF-to-SDE Conversion and Multi-Atom Codebook

All three experiments share the same compression backbone, differing only in how the generative model is conditioned. The shared components are:

### 1.1 RF-to-SDE Conversion (`sde_convert.py`)

**Purpose.** Wan2.1 is a rectified-flow (RF) model trained to predict velocity `u_t` along the linear interpolant `x_t = (1-t)x_0 + t*epsilon`. RF models are ordinarily sampled with a deterministic ODE `dx = u_t dt`. To enable codebook-based noise replacement, we must convert the deterministic ODE into an equivalent stochastic differential equation (SDE), introducing a noise injection point at every sampling step that can be controlled by the encoder.

**Design rationale.** The conversion follows Eqs. 7-9 of the Turbo-DDCM paper:

1. **Score from velocity (Eq. 8).** Given the linear interpolant `alpha_t = 1-t, sigma_t = t`, the score function is derived analytically:
   ```
   score = -[(1-t)*u_t + x_t] / t
   ```
   This is computed in `velocity_to_score()`. The derivation reduces to `score = -epsilon/t`, matching the standard DDPM relationship, which serves as a correctness check.

2. **SDE drift (Eq. 7).** The reverse SDE drift is:
   ```
   f_t = u_t - (g_t^2 / 2) * score
   ```
   When the diffusion coefficient `g_t = 0`, this reduces to the original ODE velocity. The SDE thus generalizes the ODE — the same pretrained model serves both sampling modes.

3. **Euler-Maruyama step (Eq. 9).** Each reverse step is:
   ```
   x_{t-dt} = x_t - f_t*dt + g_t*sqrt(dt)*z
   ```
   The noise term `z` is where the codebook operates. In standard SDE sampling, `z ~ N(0,I)`. In Turbo-DDCM, `z` is a structured combination of codebook atoms selected to steer reconstruction toward the ground truth.

**Diffusion coefficient choice.** We use `g_t = scale * t^2` with `scale=3.0` (Appendix F, Table 8b of the paper). The quadratic schedule gives `g(0)=0` (smooth transition to clean data) and increases toward `g(1)=scale` (maximum stochasticity at the noise end). This schedule concentrates codebook influence in the high-noise region where the model's prediction uncertainty is largest and where steering has the most leverage.

**Timestep scheduling.** We use SD3-style shifted scheduling `t_shifted = shift * t / (1 + (shift-1)*t)` which allocates more steps to the high-noise region. The shift parameter is resolution-dependent: `shift=5.0` for 720p+ (matching Wan2.1's default) and `shift=3.0` for 480p.

### 1.2 Multi-Atom Codebook (`turbo_codebook.py`)

**Purpose.** The codebook is the core data structure for lossy compression. At each SDE step, for each latent temporal frame, the encoder selects a structured noise vector from a shared pseudo-random codebook. Only the selection indices and sign bits need to be transmitted.

**Design rationale.**

1. **Per-step, per-frame independence.** Each (step, frame) pair has its own independently seeded random codebook. This ensures that the atom selection for one frame does not affect another, enabling parallel evaluation and simplifying the bitstream format. The seed is deterministic: `seed = base_seed * 100003 + step * 10007 + frame`.

2. **Two-pass atom selection.** For each (step, frame):
   - **Pass 1 — Inner product scan:** Generate all K atoms (one at a time for RNG portability) and compute `ip_i = <z_i, r>` where `r = x_0 - x_hat_0|t` is the residual between ground truth and current MMSE estimate. Select the top-M atoms by `|ip_i|` (Eq. 13).
   - **Pass 2 — Reconstruction:** Regenerate only the selected atoms, combine with signs: `z = sum(sign(ip_i) * z_i)`, normalize to unit variance (Eq. 10).

   The two-pass design avoids storing all K atoms in memory simultaneously (K=16384 atoms of dimension ~100K floats each would require ~6 GB per frame). Instead, Pass 1 streams atoms in configurable batches (auto-tuned to ~64 MB), keeping only the scalar inner products.

3. **One-at-a-time atom generation.** Atoms are generated via `torch.randn(D)` individually rather than `torch.randn(K, D)` in bulk. This is critical for encoder-decoder consistency: PyTorch's RNG produces different sequences when the batch dimension changes (due to internal block alignment). Per-atom generation ensures the random sequence is identical regardless of `gen_batch` or available VRAM.

4. **Unit variance normalization.** The combined noise vector `z = sum(s_i * z_i)` does not have unit variance (its variance scales with M). Normalizing by `z / std(z)` ensures the SDE step's noise magnitude matches the theoretical requirement of `g_t * sqrt(dt)`, preserving the correct diffusion dynamics.

**Bitstream format.** Each GOP produces a `.tdcm` file with:
- Header: magic bytes, K, M, step/frame counts, seed, video dimensions, optional prompt
- Data: for each SDE step, for each latent frame: M indices (uint16 if K <= 65536) + ceil(M/8) sign bytes

The bitrate per GOP is exactly `T_sde * F_lat * M * (ceil(log2(K)) + 1)` bits, where `T_sde` is the number of SDE steps (total steps minus DDIM tail).

### 1.3 DDIM Tail

**Purpose.** The last N sampling steps use deterministic ODE updates (no noise injection, no bits). This is a rate-quality tradeoff: tail steps contribute diminishing quality improvement per bit because the model's prediction is already accurate near `t=0`. Empirically, `ddim_tail=3` (720p) or `ddim_tail=1` (1080p) gives the best rate-distortion balance.

### 1.4 Encode-Decode Symmetry

**Purpose.** The encoder and decoder must follow identical SDE trajectories so that the reconstructed video matches. This is guaranteed by three shared invariants:

1. **Same initial noise.** Both sides generate `x_T ~ N(0,I)` from the same CPU-seeded generator (`seed=42`). CPU seeding ensures cross-platform reproducibility.
2. **Same model predictions.** Given identical `(x_t, t)` inputs and conditioning, the model produces identical velocity outputs. Both sides compute the same drift `f_t` and noise coefficient `g_t * sqrt(dt)`.
3. **Same codebook noise.** The encoder transmits indices and signs; the decoder regenerates atoms from the same per-(step, frame) seed and combines them identically.

### 1.5 Shared Pipeline (`turbo_pipeline.py`)

The `TurboDDCMWanPipeline` class orchestrates encoding and decoding:

- **`encode(frames, prompt, ref_image)`**: VAE-encodes ground truth, runs the forward SDE loop with residual-guided atom selection, returns step_data (indices + signs per step per frame).
- **`decode(step_data, prompt, ref_image)`**: Runs the identical SDE loop, reconstructing noise from step_data at each step, then VAE-decodes the final latent.
- **`save_compressed()` / `load_and_decode()`**: Bitstream serialization via `TurboBitstream`.

The pipeline is model-agnostic — it accepts any wrapper that implements `encode_prompt()`, `encode_video()`, `decode_latent()`, `predict_velocity_cfg()`, and `encode_image()`. The three experiments plug in different wrappers.

---

## 2. Experiment 1: I2V (Image-to-Video) — Autoregressive

**File:** `exp_i2v/run_uvg_chained_gop.py`
**Wrapper:** `sde_rf_wan/wan_wrapper.py` (WanWrapper, native Wan2.1 format)
**Model:** Wan2.1 I2V-14B-720P

### 2.1 Design Purpose

I2V conditioning provides the strongest spatial anchor among the three experiments. The model receives a CLIP visual embedding and a VAE-encoded first frame, giving it rich information about appearance, texture, and layout. Combined with autoregressive chaining and tail residual correction, this creates a practical video compression codec where only the first frame is given as free side information, and subsequent GOPs chain autoregressively.

### 2.2 Conditioning Pipeline

The I2V reference frame is fed through CLIP and VAE to produce conditioning:

```
ref_image --> CLIP visual encoder --> clip_fea (semantic features)
ref_image --> VAE encode (padded to F frames) --> y (latent conditioning)
          --> binary mask (frame 0 visible, rest masked) --> msk
y = concat(msk, y) along channel dim --> (C+4, F_lat, H_lat, W_lat)
```

The conditioning tensor `y` is concatenated with the mask along the channel dimension, giving the DiT both the latent content of the reference frame and a binary indicator of which frames are conditioned. This follows Wan2.1's native I2V interface exactly.

### 2.3 Autoregressive GOP Structure

GOPs chain autoregressively — the decoded last frame of each GOP becomes the reference for the next:

```
GOP 0: frames [0,  33)  ref = GT[0] (free, not in bitrate)
       → DDCM encode/decode with tail residual correction
       → decoded last frame → next_ref

GOP 1: frames [33, 66)  ref = next_ref (decoded last frame from GOP 0, 0 bytes)
       → DDCM encode/decode with tail residual correction
       → decoded last frame → next_ref

GOP 2: frames [66, 99)  ref = next_ref (decoded last frame from GOP 1, 0 bytes)
       ...
```

**Key design decisions:**
- **GOP 0 reference is GT first frame (free).** The first frame is given to both encoder and decoder as side information and does not count toward bitrate. This mirrors the standard assumption in video compression that the first frame is an I-frame.
- **GOP k>0 reference costs 0 bytes.** The decoded last frame is already available at the decoder (it was decoded as part of the previous GOP). Reusing it as the next GOP's reference requires no additional transmission.
- **Error accumulation is controlled by tail residual correction** (Section 2.5), which ensures the decoded last frame is high quality before it propagates to the next GOP.

### 2.4 M_tail: Adaptive Atom Allocation

**Purpose.** Temporal frames near the end of the GOP (farthest from the conditioned first frame) tend to have higher reconstruction error. The `M_tail` parameter allocates more codebook atoms to the last `tail_latent_frames` latent frames (default: last 2), boosting quality where the model prior is weakest.

**Implementation.** `_get_M_for_frame(f)` returns `M_tail` for frame indices `>= num_latent_frames - tail_latent_frames`, and `M` for all others. This requires variable-length bitstream entries — each frame's contribution to the bitstream is `M_f * (bits_per_index + 1)` bits, where `M_f` varies by frame position.

**Default values:** `M=64, M_tail=128` at 720p; `M=80, M_tail=160` at 1080p.

### 2.5 Tail Latent Residual Correction

**Purpose.** The last latent frame (covering the last 4 pixel frames via Wan-VAE's 4x temporal stride) is farthest from the reference and has the highest reconstruction error. In the autoregressive chain, this frame becomes the next GOP's reference — its quality directly determines whether the chain accumulates error or stays stable. Tail residual correction fixes this frame cheaply.

**Implementation.** After SDE encoding, `encode()` returns the decoded latent `x0_enc` — identical to what `decode()` would produce (same SDE trajectory). The residual between the ground-truth latent and `x0_enc` for the last 1 latent frame is:

```
residual = gt_latent[:, :, -1:, :, :] - x0_enc[:, :, -1:, :, :]
```

This residual is compressed via per-channel min/max quantization (8-bit by default) + zlib:
1. Normalize per channel to [-1, 1] using `maxval = abs(residual).amax(per_channel)`
2. Quantize to 256 levels (8-bit), pack as uint8
3. zlib compress (level 9)

At the decoder, the decompressed residual is added to the final latent before VAE decode via the `latent_correction` parameter in `TurboDDCMWanPipeline.decode()`. All metrics (PSNR, MS-SSIM, LPIPS) are computed on the corrected decoded frames — the residual is part of the bitstream and costs bits.

**Overhead.** The tail residual bytes count toward bitrate. Controlled by `--tail_residual_bits` (4/8/16). Disable with `--no_tail_residual`.

### 2.6 Model Wrapper Design

`WanWrapper` uses Wan2.1's native Python modules (DiT, VAE, T5, CLIP) rather than the HuggingFace diffusers pipeline. This avoids version incompatibilities and gives direct control over the forward pass.

Key interface methods:
- `encode_prompt(prompt)` — T5 encoding, returns `{prompt_embeds, negative_prompt_embeds}` as list-of-tensor
- `encode_image(image, num_frames, height, width)` — CLIP visual + VAE encode + mask construction, returns `{clip_fea, y}`
- `predict_velocity_cfg(x_t, t, embeds, guidance_scale, i2v_cond)` — DiT forward pass with optional CFG

The timestep convention is `[0, 1000]` scale (Wan native), converted from the pipeline's `[0, 1]` scale via `t * 1000`.

### 2.7 Bitrate Accounting

```
Per-GOP bitrate:
  codebook: T_sde * F_lat * M * (ceil(log2(K)) + 1) bits
  + tail residual: zlib(quantize(residual)) bytes * 8 bits  (default on)
  GOP 0 ref: GT first frame (free, 0 bits)
  GOP k>0 ref: decoded last frame (free, 0 bits)
  Total pixels: num_frames * H * W * 3
  BPP = total_bits / total_pixels
  Bitrate (kbps) = total_bits / (num_frames / fps) / 1000
```

---

## 3. Experiment 2: T2V (Text-to-Video)

**File:** `exp_t2v/run_t2v_experiment.py`
**Wrapper:** `sde_rf_wan/wan_t2v_wrapper.py` (WanT2VDiffusersWrapper)
**Model:** Wan2.1 T2V-14B-Diffusers

### 3.1 Design Purpose

T2V with an empty text prompt tests the lower bound of compression quality achievable from the generative model prior alone. No reference frame is transmitted — the entire bitrate is codebook bits. This experiment isolates the contribution of the pretrained video generation model as a learned prior for compression.

### 3.2 Conditioning Pipeline

```
empty string "" --> T5 tokenizer + encoder --> prompt_embeds (near-zero content)
negative_prompt --> T5 tokenizer + encoder --> neg_prompt_embeds
guidance_scale = 1.0 --> no CFG (single forward pass per step)
```

With `guidance_scale=1.0`, only the conditional forward pass is executed (no unconditional pass), halving the compute cost per step compared to CFG-enabled experiments.

### 3.3 Diffusers Format Wrapper

T2V uses the HuggingFace diffusers checkpoint format (`WanTransformer3DModel`, `AutoencoderKLWan`, `UMT5EncoderModel`), unlike the native format used by I2V and FLF2V.

**Design rationale.** The T2V-14B checkpoint is distributed only in diffusers format. Rather than converting formats, we wrote a dedicated wrapper that maps the diffusers API to the same interface expected by `TurboDDCMWanPipeline`.

**Key difference: latent normalization.** The diffusers VAE requires explicit normalization:
```python
z_norm = (z_raw - mean) / std    # encode: raw VAE output -> model space
z_raw  = z_norm * std + mean     # decode: model space -> raw VAE input
```
The normalization constants (`latents_mean`, `latents_std`) are stored in the VAE config. The native VAE (used by I2V/FLF2V) handles this internally. Failing to normalize causes severe quality degradation because the DiT was trained on normalized latents.

### 3.4 GOP Structure

GOPs are non-overlapping with no reference frames:

```
GOP 0: frames [0,  33)  no ref
GOP 1: frames [33, 66)  no ref
GOP 2: frames [66, 99)  no ref
```

**Known limitation.** Without spatial anchoring from a reference frame, the model may generate content with subtle positional drift across GOPs. The motion and appearance are governed entirely by the model's prior and the codebook steering.

### 3.5 GOP Stitching

An optional overlap-blending mechanism (`stitch_gops()`) is implemented but disabled by default (`overlap=0`). When enabled, adjacent GOPs share a configurable number of frames, and a linear cross-fade is applied at the boundary:

```python
weight_b = (i + 1) / (overlap + 1)  # increases from 0 to 1
blended = weight_a * frame_a + weight_b * frame_b
```

This can reduce visual discontinuities at GOP boundaries but increases the bitrate (overlapping frames are encoded twice).

### 3.6 Bitrate Accounting

```
Per-GOP bitrate:
  codebook: T_sde * F_lat * M * (ceil(log2(K)) + 1) bits
  ref frame: 0 bits
  BPP = codebook_bits / total_pixels
```

The T2V experiment achieves the lowest bitrate among the three tracks since no side information is transmitted.

---

## 4. Experiment 3: FLF2V (First-Last-Frame-to-Video)

**File:** `exp_flf2v/run_flf2v_experiment.py`
**Wrapper:** `sde_rf_wan/wan_flf2v_wrapper.py` (WanFLF2VWrapper)
**Model:** Wan2.1 FLF2V-14B-720P

### 4.1 Design Purpose

FLF2V conditioning provides two spatial anchors per GOP — both the first and last frames. This constrains the generative model from both temporal ends, reducing accumulation error within the GOP. The key engineering contribution is a boundary-sharing scheme that amortizes the cost of reference frames across consecutive GOPs.

### 4.2 Conditioning Pipeline

For each GOP, two boundary frames are compressed and encoded:

```
GT first frame --> CompressAI --> first_decoded
GT last frame  --> CompressAI --> last_decoded

# CLIP conditioning: both frames
clip_fea = CLIP.visual([first[:, None, :, :], last[:, None, :, :]])

# VAE conditioning: first + zeros + last
video_cond = concat(first_resized, zeros(F-2), last_resized)  # (3, F, H, W)
y = VAE.encode(video_cond)  # (C, F_lat, H_lat, W_lat)

# Mask: boundary frames visible, interior masked
msk = ones(1, F, lat_h, lat_w)
msk[:, 1:-1] = 0   # first=1, middle=0, last=1

# VAE temporal compression alignment
msk = reshape_to_match_vae_stride(msk)  # (4, F_lat, H_lat, W_lat)
y = concat(msk, y)  # (C+4, F_lat, H_lat, W_lat)
```

The mask construction follows Wan2.1's native `WanFLF2V.generate()` exactly. The first frame is expanded to 4 temporal copies before VAE compression alignment, matching the VAE's 4x temporal stride.

### 4.3 Boundary Frame Sharing

The key architectural innovation in the FLF2V experiment is GOP chaining with shared boundary frames:

```
GOP 0: first=compress(GT[0]),   last=compress(GT[32])   frames [0..32]
GOP 1: first=compress(GT[32]),  last=compress(GT[64])   frames [32..64]
GOP 2: first=compress(GT[64]),  last=compress(GT[96])   frames [64..96]
```

The last frame of GOP N is reused as the first frame of GOP N+1. This means:
- **GOP 0** transmits both boundary frames: `first_bytes + last_bytes`
- **GOP k>0** transmits only the new last frame: `last_bytes` (first is reused)

**Concatenation:** `GOP0[0:33] + GOP1[1:33] + GOP2[1:33]` = `33 + 32*2 = 97` unique frames for 3 GOPs.

**Design rationale.** Without sharing, N GOPs would require 2N boundary frame compressions. With sharing, only N+1 are needed (a ~50% savings on boundary bytes for large N). More importantly, reusing the decoded boundary frame ensures temporal consistency at the splice point — both GOPs see exactly the same decoded reference, eliminating mismatch artifacts.

### 4.4 Custom Encode/Decode Functions

FLF2V cannot use the standard `pipe.encode()` / `pipe.decode()` because the pipeline's built-in encode calls `model.encode_image()` (I2V interface), whereas FLF2V needs `model.encode_first_last_frames()` (two-frame interface). Instead, `flf2v_encode()` and `flf2v_decode()` manually drive the SDE loop:

```python
def flf2v_encode(pipe, model, gop_frames, flf2v_cond, height, width):
    # 1. Encode prompt (empty)
    # 2. VAE-encode ground truth video
    # 3. Run SDE loop with flf2v_cond as i2v_cond
    # 4. At each SDE step: compute residual, select atoms, assemble noise
    # Returns: step_data, x0_true
```

The SDE loop logic is identical to `TurboDDCMWanPipeline.encode()` but uses the externally computed `flf2v_cond` instead of calling `model.encode_image()` internally.

**Design trade-off.** This duplication of the SDE loop could be avoided by refactoring the pipeline to accept pre-computed conditioning. We chose explicit duplication for clarity and to avoid breaking the I2V and T2V paths with FLF2V-specific abstractions.

### 4.5 Model Wrapper Design

`WanFLF2VWrapper` extends the pattern of `WanWrapper` with:

- `encode_first_last_frames(first, last, num_frames, height, width)` — CLIP encodes both frames as a pair; VAE encodes the first-zeros-last video; constructs the two-point mask
- `encode_image(image, ...)` — compatibility shim that delegates to `encode_first_last_frames(image, image, ...)` for pipeline interface compatibility
- Config: `flf2v-14B`, `flow_shift=5.0` (720p default)

### 4.6 Bitrate Accounting

```
Per-GOP bitrate:
  codebook: T_sde * F_lat * M * (ceil(log2(K)) + 1) bits
  GOP 0: first_frame_bytes + last_frame_bytes
  GOP k>0: last_frame_bytes only (first reused)

  unique_frames: 33 (GOP 0) or 32 (GOP k>0)
  BPP = total_gop_bits / (unique_frames * H * W * 3)
```

The per-sequence BPP is computed over all unique frames: `total_bytes * 8 / (total_unique_frames * H * W * 3)`.

---

## 5. Metrics Collection

All three experiments collect identical metrics per GOP, stored in `metrics.json`:

| Metric | Description | Computation |
|---|---|---|
| PSNR (dB) | Peak Signal-to-Noise Ratio | Per-frame MSE in [0,1] space, averaged |
| MS-SSIM | Multi-Scale Structural Similarity | `pytorch_msssim.ms_ssim`, batched by 4 |
| LPIPS | Learned Perceptual Image Patch Similarity | AlexNet backbone, batched by 4 |
| BPP | Bits Per Pixel | total_bits / (F * H * W * 3) |
| Bitrate (kbps) | Kilobits per second | total_bits / duration / 1000 |
| Per-frame PSNR | Frame-level quality curve | MSE per frame, useful for temporal analysis |
| Encode time (s) | Wall-clock encoding time | Includes model inference + atom selection |
| Decode time (s) | Wall-clock decoding time | Includes model inference + atom reconstruction + VAE decode |

A `summary.json` file aggregates per-sequence and overall averages across all GOPs and sequences.

---

## 6. Parameter Configuration Summary

| Parameter | 720p | 1080p | Rationale |
|---|---|---|---|
| M | 64 | 80 | All three experiments: higher resolution → more atoms |
| M_tail (I2V) | 128 | 128 | Fixed at 128 for both resolutions (AR tail frame quality) |
| K | 16384 | 16384 | Codebook size; beyond 16K, diminishing returns |
| steps | 20 | 20 | Total sampling steps; <15 insufficient quality |
| ddim_tail | 3 | 3 | Unified at 3 for all experiments |
| g_scale | 3.0 | 3.0 | SDE diffusion coefficient; sweet spot validated at 2.0-3.0 |
| guidance_scale | 1.0 | 1.0 | No CFG (empty/minimal prompt); saves one forward pass per step |
| flow_shift | 5.0 | 5.0 | Wan2.1 default for 720p+ |
| seed | 42 | 42 | Shared encoder/decoder seed for deterministic reproduction |
| num_frames | 33 | 33 | Frames per GOP; must be 4k+1 for VAE temporal alignment |
| tail_residual (I2V) | on | on | 8-bit quantized residual for last 1 latent frame; fixes AR chain quality |
| tail_residual_bits | 8 | 8 | Per-channel min/max quantization bits (4/8/16) |

The `4k+1` frame count constraint arises from Wan-VAE's temporal stride of 4: `F_lat = (F-1)/4 + 1`. For F=33, `F_lat = 9` latent frames.
