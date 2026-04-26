"""
sd3_image_ddcm.py — DDCM-Turbo Image Compression via SD3.5 (RF→SDE)

Uses SD3.5 Medium as a rectified-flow image model with RF→SDE conversion.
Same math as Wan video pipeline (sde_convert.py):
  - Score from velocity (Eq.8): ∇log p_t = -[(1-t)·v + x_t] / t
  - SDE drift (Eq.7): f_t = v - (g_t²/2)·score
  - Euler-Maruyama step (Eq.9): x_{t-Δt} = x_t - f_t·Δt + g_t·√Δt·z
  - Codebook noise replacement at each SDE step

Key advantage over SDXL DDIM approach:
  - g_t = scale·t² gives controllable, proven noise injection
  - RF velocity field gives clean x₀ estimates: x̂₀ = x_t - t·v
  - Same pipeline architecture as Wan video — unified codebase
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from PIL import Image
from dataclasses import dataclass

from .sde_convert import (
    velocity_to_score,
    diffusion_coeff,
    sde_drift,
    shifted_timesteps,
)
from .turbo_codebook import TurboPerFrameCodebook


@dataclass
class SD3DDCMConfig:
    K: int = 16384          # codebook size
    M: int = 100            # atoms per step
    num_steps: int = 28     # total sampling steps (SD3.5 default)
    num_ddim_tail: int = 3  # last N steps deterministic (ODE, no bits)
    g_scale: float = 3.0    # SDE diffusion coefficient: g_t = scale·t²
    flow_shift: float = 3.0 # SD3 timestep shift
    seed: int = 42
    height: int = 720
    width: int = 1280


class SD3ImageDDCM:
    """DDCM-Turbo image compression using SD3.5 with RF→SDE conversion."""

    def __init__(self, sd3_pipe, config: SD3DDCMConfig):
        self.config = config
        self.device = sd3_pipe.transformer.device

        # Extract SD3 components
        self.transformer = sd3_pipe.transformer
        self.vae = sd3_pipe.vae
        self.text_encoder = getattr(sd3_pipe, 'text_encoder', None)
        self.text_encoder_2 = getattr(sd3_pipe, 'text_encoder_2', None)
        self.text_encoder_3 = getattr(sd3_pipe, 'text_encoder_3', None)
        self.tokenizer = getattr(sd3_pipe, 'tokenizer', None)
        self.tokenizer_2 = getattr(sd3_pipe, 'tokenizer_2', None)
        self.tokenizer_3 = getattr(sd3_pipe, 'tokenizer_3', None)

        # VAE scaling factor
        self.vae_scale = self.vae.config.scaling_factor  # typically 1.5305
        self.vae_shift = getattr(self.vae.config, 'shift_factor', 0.0609)

        # Latent shape: SD3 VAE = 16 channels, 8x downsampling
        h_lat = config.height // 8
        w_lat = config.width // 8
        self.latent_shape = (16, h_lat, w_lat)

        # Timestep schedule (same as Wan: shifted schedule)
        self.num_sde_steps = config.num_steps - config.num_ddim_tail
        self.timesteps = shifted_timesteps(
            config.num_steps, shift=config.flow_shift, device=self.device
        )

        # Codebook
        self.codebook = TurboPerFrameCodebook(
            K=config.K, M=config.M,
            frame_shape=self.latent_shape,
            seed=config.seed, device=self.device,
        )

        # Null embeddings (computed once)
        self._null_embeds = None

        # Stats
        total_bits = self.num_sde_steps * self.codebook.bits_per_frame_step
        total_pixels = config.height * config.width * 3
        bpp = total_bits / total_pixels
        print(f"SD3.5 Image DDCM (RF→SDE):")
        print(f"  Image: {config.height}x{config.width}")
        print(f"  Latent: {self.latent_shape} (D={self.codebook.D})")
        print(f"  VAE scale={self.vae_scale}, shift={self.vae_shift}")
        print(f"  K={config.K}, M={config.M}, steps={config.num_steps}, "
              f"tail={config.num_ddim_tail}, g_scale={config.g_scale}")
        print(f"  SDE steps: {self.num_sde_steps}, Total: {total_bits} bits "
              f"({total_bits//8} bytes), BPP={bpp:.6f}")

    # ==============================================================
    # Text encoding (null prompt for compression)
    # ==============================================================

    @torch.no_grad()
    def _get_null_embeds(self):
        if self._null_embeds is not None:
            return self._null_embeds

        prompt = ""
        prompt_embeds_list = []
        pooled_list = []

        # CLIP-L (768d pooled)
        if self.text_encoder is not None and self.tokenizer is not None:
            tok = self.tokenizer(
                prompt, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            enc = self.text_encoder(tok, output_hidden_states=True)
            prompt_embeds_list.append(enc.hidden_states[-2])
            pooled_list.append(enc[0])  # (1, 768)

        # CLIP-G (1280d pooled)
        if self.text_encoder_2 is not None and self.tokenizer_2 is not None:
            tok2 = self.tokenizer_2(
                prompt, padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            enc2 = self.text_encoder_2(tok2, output_hidden_states=True)
            prompt_embeds_list.append(enc2.hidden_states[-2])
            pooled_list.append(enc2[0])  # (1, 1280)

        # T5 (may be None if not loaded)
        if self.text_encoder_3 is not None and self.tokenizer_3 is not None:
            tok3 = self.tokenizer_3(
                prompt, padding="max_length",
                max_length=256,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
            enc3 = self.text_encoder_3(tok3)[0]
            prompt_embeds_list.append(enc3)

        # Pooled: concat CLIP-L (768) + CLIP-G (1280) = 2048
        if pooled_list:
            pooled_prompt_embeds = torch.cat(pooled_list, dim=-1)  # (1, 2048)
        else:
            pooled_prompt_embeds = torch.zeros(1, 2048, device=self.device)

        # SD3 prompt_embeds layout:
        #   CLIP-L (768) + CLIP-G (1280) → concat features → (77, 2048)
        #   Zero-pad to T5 dim → (77, 4096)
        #   (If T5 present, concat along seq_len: (77+256, 4096))
        if len(prompt_embeds_list) >= 2:
            # CLIP-L + CLIP-G hidden states
            clip_embeds = torch.cat(prompt_embeds_list[:2], dim=-1)  # (1, 77, 2048)
            # Pad feature dim to 4096 (T5 channel width)
            clip_embeds = torch.nn.functional.pad(
                clip_embeds, (0, 4096 - clip_embeds.shape[-1])
            )  # (1, 77, 4096)
            if len(prompt_embeds_list) > 2:
                # T5 embeddings present
                t5_embeds = prompt_embeds_list[2]  # (1, 256, 4096)
                prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=-2)
            else:
                prompt_embeds = clip_embeds  # (1, 77, 4096)
        else:
            prompt_embeds = torch.zeros(1, 77, 4096, device=self.device)

        self._null_embeds = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
        return self._null_embeds

    # ==============================================================
    # VAE encode / decode
    # ==============================================================

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """PIL Image → SD3 latent (1, 16, H/8, W/8)."""
        image = image.resize((self.config.width, self.config.height), Image.LANCZOS)
        arr = np.array(image).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        t = (2.0 * t - 1.0).to(self.device, dtype=self.vae.dtype)

        latent = self.vae.encode(t).latent_dist.sample()
        # SD3 VAE scaling: (latent - shift) * scale
        return ((latent - self.vae_shift) * self.vae_scale).float()

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """SD3 latent (1, 16, H/8, W/8) → PIL Image."""
        # Inverse scaling
        latent = (latent.to(self.vae.dtype) / self.vae_scale) + self.vae_shift
        image = self.vae.decode(latent).sample
        image = (image / 2.0 + 0.5).clamp(0, 1)
        image = image[0].permute(1, 2, 0).cpu().float().numpy()
        return Image.fromarray((image * 255).astype(np.uint8))

    # ==============================================================
    # Velocity prediction
    # ==============================================================

    @torch.no_grad()
    def predict_velocity(self, x_t: torch.Tensor, t: float) -> torch.Tensor:
        """Predict velocity v_θ(x_t, t) using SD3 transformer.

        SD3 is a rectified-flow model: v = ε - x₀
        Linear interpolant: x_t = (1-t)·x₀ + t·ε

        Args:
            x_t: (1, 16, H/8, W/8) noisy latent
            t: timestep in [0, 1]

        Returns:
            v: (1, 16, H/8, W/8) velocity prediction
        """
        embeds = self._get_null_embeds()

        # SD3 transformer expects timestep in [0, 1000]
        timestep = torch.tensor([t * 1000.0], device=self.device, dtype=x_t.dtype)

        v_pred = self.transformer(
            hidden_states=x_t.to(self.transformer.dtype),
            timestep=timestep,
            encoder_hidden_states=embeds["prompt_embeds"].to(self.transformer.dtype),
            pooled_projections=embeds["pooled_prompt_embeds"].to(self.transformer.dtype),
        ).sample

        return v_pred.float()

    # ==============================================================
    # Encode: image → codebook indices (RF→SDE)
    # ==============================================================

    @torch.no_grad()
    def encode(self, image: Image.Image) -> List[Tuple[List[int], List[int]]]:
        """Encode image to DDCM bitstream using RF→SDE conversion.

        Same math as Wan video pipeline:
          1. Predict velocity v_t
          2. x̂₀ = x_t - t·v_t (MMSE estimate)
          3. Residual r = x₀_true - x̂₀
          4. Select top-M codebook atoms by |⟨z_i, r⟩|
          5. SDE step: x_{t-Δt} = x_t - f_t·Δt + g_t·√Δt·z_codebook

        Returns:
            step_data: [sde_step] = (indices, signs)
        """
        x0_true = self.encode_image(image)

        # Deterministic initial noise
        gen = torch.Generator(device="cpu").manual_seed(self.config.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device)

        step_data = []
        sde_idx = 0

        for i in range(self.config.num_steps):
            t_curr = self.timesteps[i].item()
            t_next = self.timesteps[i + 1].item()
            delta_t = t_curr - t_next

            v_t = self.predict_velocity(x_t, t_curr)

            # Final step → ODE to t=0
            if t_next < 1e-6:
                x_t = x_t - v_t * delta_t
                break

            # DDIM tail → ODE, no bits
            if i >= self.num_sde_steps:
                x_t = x_t - v_t * delta_t
                continue

            # --- DDCM SDE step (same as Wan video) ---
            # MMSE estimate: x̂₀ = x_t - t·v_t
            x0_hat = x_t - t_curr * v_t

            # Residual
            residual = (x0_true - x0_hat).squeeze(0)  # (16, H/8, W/8)

            # SDE components (Eq.7, 8)
            score = velocity_to_score(v_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, self.config.g_scale)
            f_t = sde_drift(v_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            # Select codebook atoms (single image = frame 0)
            idx, sgn, z_f = self.codebook.select_atoms(residual, sde_idx, 0)
            step_data.append((idx, sgn))

            # SDE step: x_{t-Δt} = x_t - f_t·Δt + g_t·√Δt·z
            z_4d = z_f.unsqueeze(0)  # (1, 16, H/8, W/8)
            x_t = x_t - f_t * delta_t + noise_coeff * z_4d

            sde_idx += 1

            if (i + 1) % 5 == 0 or i == 0:
                mse = ((x0_true - x0_hat) ** 2).mean().item()
                print(f"  Encode step {i+1}/{self.config.num_steps}: t={t_curr:.3f}, "
                      f"MSE={mse:.4f}, g_t={g_t:.4f}, noise_coeff={noise_coeff:.4f}")

        return step_data

    # ==============================================================
    # Decode: codebook indices → image (RF→SDE)
    # ==============================================================

    @torch.no_grad()
    def decode(self, step_data: List[Tuple[List[int], List[int]]]) -> Image.Image:
        """Decode image from DDCM bitstream using RF→SDE conversion."""
        gen = torch.Generator(device="cpu").manual_seed(self.config.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device)

        sde_idx = 0

        for i in range(self.config.num_steps):
            t_curr = self.timesteps[i].item()
            t_next = self.timesteps[i + 1].item()
            delta_t = t_curr - t_next

            v_t = self.predict_velocity(x_t, t_curr)

            if t_next < 1e-6:
                x_t = x_t - v_t * delta_t
                break

            if i >= self.num_sde_steps:
                x_t = x_t - v_t * delta_t
                continue

            # Reconstruct noise from codebook
            idx, sgn = step_data[sde_idx]
            z_f = self.codebook.reconstruct(idx, sgn, sde_idx, 0)
            z_4d = z_f.unsqueeze(0)

            # SDE components
            score = velocity_to_score(v_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, self.config.g_scale)
            f_t = sde_drift(v_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            x_t = x_t - f_t * delta_t + noise_coeff * z_4d

            sde_idx += 1

            if (i + 1) % 5 == 0:
                print(f"  Decode step {i+1}/{self.config.num_steps}")

        return self.decode_latent(x_t)

    # ==============================================================
    # Stats
    # ==============================================================

    @property
    def total_bits(self):
        return self.num_sde_steps * self.codebook.bits_per_frame_step

    @property
    def total_bytes(self):
        return self.total_bits // 8
