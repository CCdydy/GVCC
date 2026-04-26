"""
sdxl_image_ddcm.py — DDCM-Turbo Image Compression via SDXL

Compresses a single image using SDXL as the base diffusion model.
Uses stochastic DDIM (η > 0) with codebook noise replacement.

SDXL is VP (variance-preserving, ε-prediction), so we use:
  score = -ε_θ / σ_t
  DDIM step with η → noise injection point → codebook replacement

This replaces JPEG for first-frame compression in the I2V pipeline.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from PIL import Image
from dataclasses import dataclass
from diffusers import DDIMScheduler

from .turbo_codebook import TurboPerFrameCodebook


@dataclass
class SDXLDDCMConfig:
    K: int = 16384          # codebook size
    M: int = 32             # atoms per step
    num_steps: int = 20     # total DDIM steps
    num_ddim_tail: int = 3  # last N steps deterministic (no bits)
    eta: float = 1.0        # stochasticity (0=DDIM, 1=DDPM)
    seed: int = 42
    height: int = 480
    width: int = 832


class SDXLImageDDCM:
    """DDCM-Turbo image compression using SDXL."""

    def __init__(self, sdxl_pipe, config: SDXLDDCMConfig):
        self.config = config
        self.device = sdxl_pipe.unet.device

        # Extract SDXL components
        self.unet = sdxl_pipe.unet
        self.vae = sdxl_pipe.vae
        self.scheduler = sdxl_pipe.scheduler
        self.text_encoder = sdxl_pipe.text_encoder
        self.text_encoder_2 = sdxl_pipe.text_encoder_2
        self.tokenizer = sdxl_pipe.tokenizer
        self.tokenizer_2 = sdxl_pipe.tokenizer_2

        # CRITICAL: SDXL VAE is unstable in fp16 → must use fp32
        if self.vae.dtype == torch.float16:
            self.vae = self.vae.to(dtype=torch.float32)
            print("  VAE upcast: fp16 → fp32 (SDXL VAE NaN fix)")

        # VAE scaling factor
        self.vae_scale = self.vae.config.scaling_factor  # typically 0.13025

        # Latent shape: SDXL VAE = 4 channels, 8x downsampling
        h_lat = config.height // 8
        w_lat = config.width // 8
        self.latent_shape = (4, h_lat, w_lat)

        # CRITICAL: Use DDIMScheduler (not EulerDiscrete which is SDXL default)
        # DDCM requires DDIM stochastic sampling with η > 0
        self.scheduler = DDIMScheduler.from_config(sdxl_pipe.scheduler.config)
        self.scheduler.set_timesteps(config.num_steps, device=self.device)
        self.timesteps = self.scheduler.timesteps  # descending, e.g. [951, 901, ...]

        # Precompute alphas/sigmas for each timestep
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        # Codebook (single frame = single latent)
        self.num_sde_steps = config.num_steps - config.num_ddim_tail
        self.codebook = TurboPerFrameCodebook(
            K=config.K, M=config.M,
            frame_shape=self.latent_shape,
            seed=config.seed, device=self.device,
        )

        # Null text embeddings (computed once)
        self._null_embeds = None

        # Stats
        total_bits = self.num_sde_steps * self.codebook.bits_per_frame_step
        total_pixels = config.height * config.width * 3
        bpp = total_bits / total_pixels
        print(f"SDXL Image DDCM:")
        print(f"  Scheduler: {self.scheduler.__class__.__name__}")
        print(f"  Timesteps: {self.timesteps[:3].tolist()} ... {self.timesteps[-3:].tolist()}")
        print(f"  Image: {config.height}x{config.width}")
        print(f"  Latent: {self.latent_shape} (D={self.codebook.D})")
        print(f"  VAE scale: {self.vae_scale}")
        print(f"  K={config.K}, M={config.M}, steps={config.num_steps}, "
              f"tail={config.num_ddim_tail}, η={config.eta}")
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
        # CLIP-L
        tok1 = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)
        enc1 = self.text_encoder(tok1, output_hidden_states=True)
        hidden1 = enc1.hidden_states[-2]  # penultimate layer

        # CLIP-G
        tok2 = self.tokenizer_2(
            prompt, padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)
        enc2 = self.text_encoder_2(tok2, output_hidden_states=True)
        hidden2 = enc2.hidden_states[-2]
        pooled = enc2[0]  # pooled output

        prompt_embeds = torch.cat([hidden1, hidden2], dim=-1)

        self._null_embeds = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled,
        }
        return self._null_embeds

    # ==============================================================
    # VAE encode / decode
    # ==============================================================

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """PIL Image → SDXL latent (1, 4, H/8, W/8)."""
        image = image.resize((self.config.width, self.config.height), Image.LANCZOS)
        arr = np.array(image).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        t = (2.0 * t - 1.0).to(self.device, dtype=self.vae.dtype)

        latent = self.vae.encode(t).latent_dist.sample()
        return (latent * self.vae_scale).float()

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """SDXL latent (1, 4, H/8, W/8) → PIL Image."""
        latent = latent.to(self.vae.dtype) / self.vae_scale
        image = self.vae.decode(latent).sample
        image = (image / 2.0 + 0.5).clamp(0, 1)
        image = image[0].permute(1, 2, 0).cpu().float().numpy()
        return Image.fromarray((image * 255).astype(np.uint8))

    # ==============================================================
    # UNet noise prediction
    # ==============================================================

    @torch.no_grad()
    def predict_noise(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Predict noise ε_θ(x_t, t) using SDXL UNet.

        Args:
            x_t: (1, 4, H/8, W/8) noisy latent
            t: discrete timestep (integer, from scheduler)

        Returns:
            eps: (1, 4, H/8, W/8) predicted noise
        """
        embeds = self._get_null_embeds()
        timestep = torch.tensor([int(t)], device=self.device, dtype=torch.long)

        # SDXL UNet expects added_cond_kwargs
        added_cond = {
            "text_embeds": embeds["pooled_prompt_embeds"],
            "time_ids": self._get_time_ids(),
        }

        noise_pred = self.unet(
            x_t.to(self.unet.dtype),
            timestep,
            encoder_hidden_states=embeds["prompt_embeds"].to(self.unet.dtype),
            added_cond_kwargs={k: v.to(self.unet.dtype) for k, v in added_cond.items()},
        ).sample

        return noise_pred.float()

    def _get_time_ids(self):
        """SDXL micro-conditioning: (orig_h, orig_w, crop_top, crop_left, target_h, target_w)."""
        h, w = self.config.height, self.config.width
        return torch.tensor([[h, w, 0, 0, h, w]], device=self.device, dtype=torch.float32)

    # ==============================================================
    # Noise schedule helpers
    # ==============================================================

    def _get_alpha_sigma(self, t):
        """Get α_t and σ_t for discrete timestep t."""
        abar = self.alphas_cumprod[int(t)]
        alpha_t = torch.sqrt(abar)
        sigma_t = torch.sqrt(1.0 - abar)
        return alpha_t.item(), sigma_t.item()

    # ==============================================================
    # DDIM step with stochastic noise
    # ==============================================================

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        eps_theta: torch.Tensor,
        t_curr: int,
        t_next: int,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Stochastic DDIM step.

        x_{t-1} = α_{t-1} · x̂₀ + √(σ²_{t-1} - σ²_η) · ε_θ + σ_η · z

        Args:
            x_t: current noisy latent
            eps_theta: predicted noise
            t_curr: current timestep
            t_next: next timestep (or 0 for final)
            noise: codebook noise (or None for random)

        Returns:
            x_next, x0_hat, noise_coeff
        """
        alpha_t, sigma_t = self._get_alpha_sigma(t_curr)

        if t_next > 0:
            alpha_next, sigma_next = self._get_alpha_sigma(t_next)
        else:
            alpha_next, sigma_next = 1.0, 0.0

        # Predicted clean image (clip to prevent extreme values at high t)
        x0_hat = (x_t - sigma_t * eps_theta) / alpha_t
        x0_hat = x0_hat.clamp(-30.0, 30.0)

        # Noise coefficient (DDPM variance)
        if t_next > 0 and sigma_t > 0:
            # σ_η = η · √((σ²_{t-1}/σ²_t) · (1 - α²_t/α²_{t-1}))
            variance = (sigma_next ** 2 / sigma_t ** 2) * (1.0 - (alpha_t / alpha_next) ** 2)
            sigma_eta = self.config.eta * (max(variance, 0.0) ** 0.5)
        else:
            sigma_eta = 0.0

        # Direction pointing to x_t
        dir_coeff = (max(sigma_next ** 2 - sigma_eta ** 2, 0.0)) ** 0.5

        # DDIM step
        x_next = alpha_next * x0_hat + dir_coeff * eps_theta

        if sigma_eta > 0:
            if noise is None:
                noise = torch.randn_like(x_t)
            x_next = x_next + sigma_eta * noise

        return x_next, x0_hat, sigma_eta

    # ==============================================================
    # Encode: image → codebook indices
    # ==============================================================

    @torch.no_grad()
    def encode(self, image: Image.Image) -> List[Tuple[List[int], List[int]]]:
        """Encode image to DDCM bitstream.

        Returns:
            step_data: [sde_step] = (indices, signs)
        """
        x0_true = self.encode_image(image)

        # Deterministic initial noise
        gen = torch.Generator(device="cpu").manual_seed(self.config.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device)

        # Add noise: x_T = α_T · x_0 + σ_T · ε  (but at T=max, x_T ≈ ε)
        # Actually for DDIM starting from pure noise, x_t IS the noise

        step_data = []
        sde_idx = 0

        for i, t_curr in enumerate(self.timesteps):
            t_curr = t_curr.item()
            t_next = self.timesteps[i + 1].item() if i + 1 < len(self.timesteps) else 0

            eps_theta = self.predict_noise(x_t, t_curr)

            # Final step or DDIM tail → deterministic (η=0), no bits
            if t_next == 0 or i >= self.num_sde_steps:
                alpha_t, sigma_t = self._get_alpha_sigma(t_curr)
                if t_next > 0:
                    alpha_next, sigma_next = self._get_alpha_sigma(t_next)
                else:
                    alpha_next, sigma_next = 1.0, 0.0
                x0_hat = ((x_t - sigma_t * eps_theta) / alpha_t).clamp(-30, 30)
                x_t = alpha_next * x0_hat + sigma_next * eps_theta
                continue

            # --- DDCM SDE step ---
            alpha_t, sigma_t = self._get_alpha_sigma(t_curr)
            x0_hat = ((x_t - sigma_t * eps_theta) / alpha_t).clamp(-30, 30)

            # Residual
            residual = (x0_true - x0_hat).squeeze(0)  # (4, H/8, W/8)

            # Select codebook atoms (single frame = frame index 0)
            idx, sgn, z_f = self.codebook.select_atoms(residual, sde_idx, 0)
            step_data.append((idx, sgn))

            # Stochastic DDIM step with codebook noise
            z_4d = z_f.unsqueeze(0)  # (1, 4, H/8, W/8)
            x_t, _, noise_coeff = self._ddim_step(x_t, eps_theta, t_curr, t_next, noise=z_4d)

            sde_idx += 1

            if (i + 1) % 5 == 0 or i == 0:
                mse = ((x0_true - x0_hat) ** 2).mean().item()
                x0h_abs = x0_hat.abs().max().item()
                xt_abs = x_t.abs().max().item()
                print(f"  Encode step {i+1}/{len(self.timesteps)}: t={t_curr}, "
                      f"MSE={mse:.4f}, σ_η={noise_coeff:.4f}, "
                      f"|x̂₀|_max={x0h_abs:.2f}, |x_t|_max={xt_abs:.2f}")

        return step_data

    # ==============================================================
    # Decode: codebook indices → image
    # ==============================================================

    @torch.no_grad()
    def decode(self, step_data: List[Tuple[List[int], List[int]]]) -> Image.Image:
        """Decode image from DDCM bitstream."""
        gen = torch.Generator(device="cpu").manual_seed(self.config.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device)

        sde_idx = 0

        for i, t_curr in enumerate(self.timesteps):
            t_curr = t_curr.item()
            t_next = self.timesteps[i + 1].item() if i + 1 < len(self.timesteps) else 0

            eps_theta = self.predict_noise(x_t, t_curr)

            if t_next == 0 or i >= self.num_sde_steps:
                alpha_t, sigma_t = self._get_alpha_sigma(t_curr)
                if t_next > 0:
                    alpha_next, sigma_next = self._get_alpha_sigma(t_next)
                else:
                    alpha_next, sigma_next = 1.0, 0.0
                x0_hat = ((x_t - sigma_t * eps_theta) / alpha_t).clamp(-30, 30)
                x_t = alpha_next * x0_hat + sigma_next * eps_theta
                continue

            # Reconstruct noise from codebook
            idx, sgn = step_data[sde_idx]
            z_f = self.codebook.reconstruct(idx, sgn, sde_idx, 0)
            z_4d = z_f.unsqueeze(0)

            x_t, _, _ = self._ddim_step(x_t, eps_theta, t_curr, t_next, noise=z_4d)
            sde_idx += 1

            if (i + 1) % 5 == 0:
                print(f"  Decode step {i+1}/{len(self.timesteps)}")

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
