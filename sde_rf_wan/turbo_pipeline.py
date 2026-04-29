"""
turbo_pipeline.py — Turbo-DDCM Video Compression Pipeline for Wan2.1

Encode: ground-truth video → multi-atom codebook indices + signs
Decode: indices + signs → reconstructed video

Key features vs basic SDE-RF pipeline:
  1. Multi-atom selection (M atoms per frame per step) — much better quality
  2. Residual-guided selection: ⟨z_i, x₀ − x̂₀|t⟩ (Eq.13)
  3. DDIM tail: last N steps deterministic (no bits)
  4. Deterministic x_T from seed (no bits for initial noise)
"""

import torch
import time
from typing import List, Tuple
from PIL import Image

from .sde_convert import (
    velocity_to_score,
    diffusion_coeff,
    sde_drift,
    ode_sample_loop,
    shifted_timesteps,
)
from .turbo_codebook import TurboPerFrameCodebook, TurboBitstream
from .wan_wrapper import WanWrapper


class TurboDDCMWanPipeline:
    """Turbo-DDCM video compression pipeline for Wan2.1.

    Parameters:
        K: codebook size (atoms per step per frame), 16384+ recommended
        M: atoms selected per frame per step (quality knob)
        num_steps: total sampling steps
        num_ddim_tail: last N steps use ODE (no bits)
        g_scale: SDE diffusion coefficient scale
    """

    def __init__(
        self,
        model: WanWrapper,
        K: int = 16384,
        M: int = 32,
        num_steps: int = 20,
        num_ddim_tail: int = 3,
        guidance_scale: float = 5.0,
        g_scale: float = 3.0,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        seed: int = 42,
        M_tail: int = None,
        tail_latent_frames: int = 2,
    ):
        self.model = model
        self.K = K
        self.M = M
        self.M_tail = M_tail
        self.tail_latent_frames = tail_latent_frames
        self.num_steps = num_steps
        self.num_ddim_tail = num_ddim_tail
        self.num_sde_steps = num_steps - num_ddim_tail
        self.guidance_scale = guidance_scale
        self.g_scale = g_scale
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.seed = seed

        self.device = model.device
        self.latent_shape = model.get_latent_shape(num_frames, height, width)
        self.frame_shape = model.get_frame_shape(height, width)
        self.num_latent_frames = self.latent_shape[1]

        self.timesteps = shifted_timesteps(
            num_steps, shift=model.flow_shift, device=self.device
        )

        self.codebook = TurboPerFrameCodebook(
            K=K, M=M, frame_shape=self.frame_shape,
            seed=seed, device=self.device,
        )

        # Print stats
        bits_per_index = self.codebook.bits_per_index
        bits_normal = M * (bits_per_index + 1)
        if M_tail:
            bits_tail = M_tail * (bits_per_index + 1)
            n_tail = min(tail_latent_frames, self.num_latent_frames)
            n_normal = self.num_latent_frames - n_tail
            total_bits = self.num_sde_steps * (n_normal * bits_normal + n_tail * bits_tail)
        else:
            total_bits = self.num_sde_steps * self.num_latent_frames * bits_normal
        self._total_codebook_bits = total_bits
        total_pixels = num_frames * height * width
        bpp = total_bits / total_pixels
        duration_s = num_frames / 16.0
        bitrate_kbps = total_bits / duration_s / 1000.0

        print(f"Turbo-DDCM Pipeline:")
        print(f"  Video: {num_frames}f @ {height}x{width}")
        print(f"  Latent: {self.latent_shape}, frame: {self.frame_shape}")
        if M_tail:
            print(f"  K={K}, M={M}, M_tail={M_tail} (last {tail_latent_frames} lat frames)")
        else:
            print(f"  K={K}, M={M}")
        print(f"  T_sde={self.num_sde_steps}, DDIM_tail={num_ddim_tail}, gen_batch={self.codebook.gen_batch}")
        print(f"  Total: {total_bits} bits ({total_bits/8:.0f} bytes)")
        print(f"  BPP: {bpp:.6f}, Bitrate: {bitrate_kbps:.2f} kbps")

    def _get_M_for_frame(self, f: int) -> int:
        """Return M for a given latent frame index (tail frames may use M_tail)."""
        if self.M_tail and f >= self.num_latent_frames - self.tail_latent_frames:
            return self.M_tail
        return self.M

    def _model_fn(self, embeds: dict, i2v_cond=None):
        """Return CFG velocity callable."""
        def fn(x_t, t):
            return self.model.predict_velocity_cfg(
                x_t, t, embeds, self.guidance_scale, i2v_cond
            )
        return fn

    # ==================================================================
    # Encode: video → codebook indices + signs
    # ==================================================================

    @torch.no_grad()
    def encode(
        self,
        frames: List[Image.Image],
        prompt: str = "",
        ref_image: Image.Image = None,
    ) -> Tuple[List[List[Tuple[List[int], List[int]]]], torch.Tensor]:
        """Encode video to Turbo-DDCM bitstream.

        At each SDE step:
          1. Predict velocity u_t, compute MMSE estimate x̂₀|t = x_t − t·u_t
          2. Compute residual r = x₀ − x̂₀|t
          3. For each frame: select top-M atoms by |⟨z_i, r_f⟩|
          4. Combine atoms with sign coefficients, normalize
          5. Use as SDE noise for this step

        Args:
            frames: list of PIL Images (ground truth video)
            prompt: text description
            ref_image: reference image for I2V (first frame)

        Returns:
            step_data: [sde_step][frame] = (indices, signs)
            x0_final: final decoded latent
        """
        embeds = self.model.encode_prompt(prompt)

        # I2V conditioning (CLIP + VAE first frame)
        i2v_cond = None
        if self.model.is_i2v and ref_image is not None:
            i2v_cond = self.model.encode_image(
                ref_image, self.num_frames, self.height, self.width)

        # VAE encode ground-truth video
        x0_true = self.model.encode_video(frames, self.height, self.width)
        self._gt_latent = x0_true  # Store for residual computation

        model_fn = self._model_fn(embeds, i2v_cond)

        # Deterministic initial noise (shared between encoder/decoder)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device)

        step_data = []
        sde_idx = 0

        for i in range(self.num_steps):
            t_curr = self.timesteps[i].item()
            t_next = self.timesteps[i + 1].item()
            delta_t = t_curr - t_next

            u_t = model_fn(x_t, t_curr)

            # Final step → ODE to t=0
            if t_next < 1e-6:
                x_t = x_t - u_t * delta_t
                break

            # DDIM tail → ODE step, no bits
            if i >= self.num_sde_steps:
                x_t = x_t - u_t * delta_t
                continue

            # --- Turbo-DDCM SDE step ---
            # MMSE estimate: x̂₀|t = x_t − t · u_t
            x0_hat = x_t - t_curr * u_t  # (1, C, F, H, W)

            # Residual: what the model is missing
            residual = (x0_true - x0_hat).squeeze(0)  # (C, F, H, W)

            # SDE components
            score = velocity_to_score(u_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, self.g_scale)
            f_t = sde_drift(u_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            # Select atoms per temporal frame (with optional tail boost)
            frame_entries = []
            noise_frames = []

            for f in range(self.num_latent_frames):
                r_f = residual[:, f, :, :]  # (C, H, W)
                M_f = self._get_M_for_frame(f)
                idx, sgn, z_f = self.codebook.select_atoms(r_f, sde_idx, f, M_override=M_f)
                frame_entries.append((idx, sgn))
                noise_frames.append(z_f)

            step_data.append(frame_entries)

            # Assemble 3D noise and do SDE step
            noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)  # (1, C, F, H, W)
            x_t = x_t - f_t * delta_t + noise_coeff * noise_3d

            sde_idx += 1

            # Progress
            if (i + 1) % 5 == 0 or i == 0:
                mse = ((x0_true - x0_hat) ** 2).mean().item()
                print(f"  Encode step {i+1}/{self.num_steps}: "
                      f"residual_MSE={mse:.4f}, noise_coeff={noise_coeff:.4f}")

        return step_data, x_t

    # ==================================================================
    # Decode: codebook indices + signs → video
    # ==================================================================

    @torch.no_grad()
    def decode(
        self,
        step_data: List[List[Tuple[List[int], List[int]]]],
        prompt: str = "",
        ref_image: Image.Image = None,
        latent_correction: torch.Tensor = None,
    ) -> List[Image.Image]:
        """Decode video from Turbo-DDCM bitstream.

        Replays exact same SDE trajectory as encoder using stored noise.
        Deterministic: same indices + signs + seed + prompt → same video.
        For I2V: ref_image provides the first frame conditioning.

        Args:
            latent_correction: optional (1, C, F, H, W) tensor added to final
                latent before VAE decode (e.g., residual for tail frame correction).
        """
        embeds = self.model.encode_prompt(prompt)

        # I2V conditioning
        i2v_cond = None
        if self.model.is_i2v and ref_image is not None:
            i2v_cond = self.model.encode_image(
                ref_image, self.num_frames, self.height, self.width)

        model_fn = self._model_fn(embeds, i2v_cond)

        # Same deterministic initial noise
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device)

        sde_idx = 0

        for i in range(self.num_steps):
            t_curr = self.timesteps[i].item()
            t_next = self.timesteps[i + 1].item()
            delta_t = t_curr - t_next

            u_t = model_fn(x_t, t_curr)

            if t_next < 1e-6:
                x_t = x_t - u_t * delta_t
                break

            if i >= self.num_sde_steps:
                x_t = x_t - u_t * delta_t
                continue

            # Reconstruct noise from stored indices + signs
            score = velocity_to_score(u_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, self.g_scale)
            f_t = sde_drift(u_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            noise_frames = []
            for f in range(self.num_latent_frames):
                idx, sgn = step_data[sde_idx][f]
                z_f = self.codebook.reconstruct(idx, sgn, sde_idx, f)
                noise_frames.append(z_f)

            noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)
            x_t = x_t - f_t * delta_t + noise_coeff * noise_3d

            sde_idx += 1

            if (i + 1) % 5 == 0:
                print(f"  Decode step {i+1}/{self.num_steps}")

        # Apply latent correction before VAE decode (e.g., tail frame residual)
        if latent_correction is not None:
            x_t = x_t + latent_correction.to(x_t.device, dtype=x_t.dtype)

        frames = self.model.decode_latent(x_t)
        return frames

    # ==================================================================
    # ODE baseline (for quality comparison)
    # ==================================================================

    @torch.no_grad()
    def generate_ode(self, prompt: str, seed: int = 42, ref_image: Image.Image = None) -> List[Image.Image]:
        """Standard ODE generation (deterministic baseline)."""
        embeds = self.model.encode_prompt(prompt)

        i2v_cond = None
        if self.model.is_i2v and ref_image is not None:
            i2v_cond = self.model.encode_image(
                ref_image, self.num_frames, self.height, self.width)

        model_fn = self._model_fn(embeds, i2v_cond)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        x_init = torch.randn(1, *self.latent_shape, generator=gen).to(self.device)

        x_final = ode_sample_loop(model_fn, x_init, self.timesteps)
        frames = self.model.decode_latent(x_final)
        return frames

    # ==================================================================
    # Bitstream I/O
    # ==================================================================

    def save_compressed(self, step_data, filepath: str, prompt: str = ""):
        """Save step_data to binary file."""
        TurboBitstream.save(
            filepath, step_data,
            self.K, self.M,
            self.num_sde_steps, self.num_ddim_tail,
            self.num_latent_frames, self.seed,
            self.frame_shape, prompt,
            self.num_frames, self.height, self.width,
        )

    def load_and_decode(self, filepath: str) -> List[Image.Image]:
        """Load bitstream and decode to video."""
        data = TurboBitstream.load(filepath)
        return self.decode(data["step_data"], data["prompt"])
