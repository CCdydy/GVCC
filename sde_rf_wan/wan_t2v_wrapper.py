"""
wan_t2v_wrapper.py — Wan2.1 T2V Wrapper (Diffusers Format)

Loads Wan2.1 T2V model from diffusers-format checkpoint and exposes
the same interface as WanWrapper for the SDE-RF pipeline.

Key differences from WanWrapper (native format):
  - Uses diffusers WanTransformer3DModel instead of native WanModel
  - Uses transformers UMT5EncoderModel instead of native T5EncoderModel
  - Uses diffusers AutoencoderKLWan instead of native WanVAE
  - Handles latent normalization (mean/std) explicitly
  - No CLIP encoder (T2V only, no image conditioning)
"""

import os
import math
import torch
import numpy as np
from typing import Tuple, List, Optional
from PIL import Image


class WanT2VDiffusersWrapper:
    """Wrapper for Wan2.1 T2V in diffusers checkpoint format."""

    def __init__(
        self,
        checkpoint_dir: str = "./Wan2.1-T2V-1.3B-Diffusers",
        flow_shift: float = 3.0,
    ):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.flow_shift = flow_shift
        self.is_i2v = False
        self.device = None
        self.dtype = None

    def load(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """Load all components from diffusers-format checkpoint."""
        from diffusers import WanTransformer3DModel, AutoencoderKLWan
        from transformers import UMT5EncoderModel, AutoTokenizer

        self.device = torch.device(device)
        self.dtype = dtype

        print(f"Loading Wan2.1 T2V (diffusers) from {self.checkpoint_dir}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.checkpoint_dir, "tokenizer")
        )

        # Load T5 text encoder (keep on CPU, move to GPU only during encoding)
        print("  Loading T5 encoder...")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            os.path.join(self.checkpoint_dir, "text_encoder"),
            torch_dtype=dtype,
        )
        self.text_encoder.eval().requires_grad_(False)

        # Load VAE
        print("  Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            os.path.join(self.checkpoint_dir, "vae"),
            torch_dtype=torch.float32,
        )
        self.vae.eval().requires_grad_(False)
        self.vae.to(self.device)

        # Latent normalization constants (same values as native WanVAE)
        mean = torch.tensor(self.vae.config.latents_mean, dtype=torch.float32)
        std = torch.tensor(self.vae.config.latents_std, dtype=torch.float32)
        self.latent_mean = mean.view(1, -1, 1, 1, 1).to(self.device)
        self.latent_std = std.view(1, -1, 1, 1, 1).to(self.device)

        # Load DiT transformer
        print("  Loading DiT transformer...")
        self.model = WanTransformer3DModel.from_pretrained(
            os.path.join(self.checkpoint_dir, "transformer"),
            torch_dtype=dtype,
        )
        self.model.eval().requires_grad_(False)
        self.model.to(self.device)

        # VAE config (matching native)
        self.vae_stride = (4, 8, 8)
        self.vae_temporal_factor = self.vae_stride[0]
        self.vae_spatial_factor = self.vae_stride[1]
        self.latent_channels = self.vae.config.z_dim  # 16
        self.patch_size = tuple(self.model.config.patch_size)  # (1, 2, 2)
        self.sample_neg_prompt = (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )

        print(f"Wan2.1 T2V loaded. Device={device}, dtype={dtype}")
        print(f"  VAE compression: {self.vae_temporal_factor}x{self.vae_spatial_factor}x{self.vae_spatial_factor}")
        print(f"  Latent channels: {self.latent_channels}")

    # ================================================================
    # Latent shape helpers
    # ================================================================

    def get_latent_shape(
        self,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
    ) -> Tuple[int, ...]:
        """Compute latent shape: (C, F_lat, H_lat, W_lat).
        Spatial dims rounded to even for DiT patch_size=(1,2,2)."""
        f_lat = (num_frames - 1) // self.vae_temporal_factor + 1
        h_lat = height // self.vae_spatial_factor
        w_lat = width // self.vae_spatial_factor
        h_lat = h_lat + (h_lat % 2)
        w_lat = w_lat + (w_lat % 2)
        return (self.latent_channels, f_lat, h_lat, w_lat)

    def get_frame_shape(
        self,
        height: int = 480,
        width: int = 832,
    ) -> Tuple[int, ...]:
        """Shape of a single temporal frame in latent space: (C, H, W).
        Rounded to even for DiT compatibility."""
        h_lat = height // self.vae_spatial_factor
        w_lat = width // self.vae_spatial_factor
        h_lat = h_lat + (h_lat % 2)
        w_lat = w_lat + (w_lat % 2)
        return (self.latent_channels, h_lat, w_lat)

    # ================================================================
    # Text encoding
    # ================================================================

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
    ) -> dict:
        """Encode text prompt via T5.

        Returns dict with prompt_embeds and negative_prompt_embeds,
        matching the interface expected by TurboDDCMWanPipeline.
        """
        if not negative_prompt:
            negative_prompt = self.sample_neg_prompt

        max_length = 226

        self.text_encoder.to(self.device)

        def _encode_one(text):
            inputs = self.tokenizer(
                [text],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.to(self.device)
            mask = inputs.attention_mask.to(self.device)

            with torch.no_grad():
                out = self.text_encoder(input_ids, attention_mask=mask)
            hidden = out.last_hidden_state.to(dtype=self.dtype)

            # Trim to actual length, then re-pad (matches pipeline behavior)
            seq_len = mask.gt(0).sum(dim=1).long()
            trimmed = hidden[0, :seq_len[0]]
            padded = torch.cat([
                trimmed,
                trimmed.new_zeros(max_length - trimmed.size(0), trimmed.size(1))
            ]).unsqueeze(0)  # (1, L, D)
            return padded

        prompt_embeds = _encode_one(prompt)
        neg_embeds = _encode_one(negative_prompt)

        return {
            "prompt_embeds": prompt_embeds,          # (1, L, D)
            "negative_prompt_embeds": neg_embeds,    # (1, L, D)
        }

    # ================================================================
    # VAE encode / decode
    # ================================================================

    def _normalize_latent(self, raw_z: torch.Tensor) -> torch.Tensor:
        """Normalize raw VAE latent to model space: z_norm = (z_raw - mean) / std."""
        return (raw_z - self.latent_mean) / self.latent_std

    def _denormalize_latent(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize from model space to raw VAE space: z_raw = z_norm * std + mean."""
        return z_norm * self.latent_std + self.latent_mean

    @torch.no_grad()
    def encode_video(
        self,
        frames: List[Image.Image],
        height: int = 480,
        width: int = 832,
    ) -> torch.Tensor:
        """Encode video frames to normalized 3D latent.

        Args:
            frames: list of PIL Images
            height, width: target resolution

        Returns:
            latent: (1, C, F_lat, H_lat, W_lat) in normalized space
        """
        processed = []
        for frame in frames:
            frame = frame.resize((width, height), Image.LANCZOS)
            arr = np.array(frame).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
            processed.append(t)

        # (B, C, F, H, W) — diffusers VAE expects batch dim
        video_tensor = torch.stack(processed, dim=1).unsqueeze(0)  # (1, 3, F, H, W)
        video_tensor = 2.0 * video_tensor - 1.0  # normalize to [-1, 1]
        video_tensor = video_tensor.to(device=self.device, dtype=torch.float32)

        # Encode
        posterior = self.vae.encode(video_tensor).latent_dist
        raw_z = posterior.mode()  # (1, C, F_lat, H_lat, W_lat)

        # Normalize to model space
        z_norm = self._normalize_latent(raw_z)
        z_norm = z_norm.float()

        # Store raw dims for decode cropping, then pad to even for DiT
        _, _, _, h_lat, w_lat = z_norm.shape
        self._raw_latent_h = h_lat
        self._raw_latent_w = w_lat
        pad_h = h_lat % 2
        pad_w = w_lat % 2
        if pad_h or pad_w:
            z_norm = torch.nn.functional.pad(z_norm, (0, pad_w, 0, pad_h, 0, 0), mode='replicate')
        return z_norm

    @torch.no_grad()
    def decode_latent(
        self,
        latent: torch.Tensor,
    ) -> List[Image.Image]:
        """Decode normalized 3D latent to video frames.

        Args:
            latent: (1, C, F_lat, H_lat, W_lat) or (C, F_lat, H_lat, W_lat)
                    in normalized model space

        Returns:
            list of PIL Images
        """
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)

        # Crop any padding from encode (odd latent dims padded to even)
        raw_h = self._raw_latent_h if hasattr(self, '_raw_latent_h') else latent.shape[3]
        raw_w = self._raw_latent_w if hasattr(self, '_raw_latent_w') else latent.shape[4]
        latent = latent[:, :, :, :raw_h, :raw_w]

        # Denormalize to raw VAE space
        raw_z = self._denormalize_latent(latent.float().to(self.device))

        # Decode
        video = self.vae.decode(raw_z, return_dict=False)[0]  # (B, C, F, H, W)
        video = video.squeeze(0)  # (C, F, H, W)

        # Convert to PIL frames
        video = (video / 2.0 + 0.5).clamp(0, 1)
        video = video.permute(1, 2, 3, 0).cpu().float().numpy()  # (F, H, W, 3)

        frames = []
        for f in range(video.shape[0]):
            img = (video[f] * 255).astype(np.uint8)
            frames.append(Image.fromarray(img))
        return frames

    # ================================================================
    # Velocity prediction
    # ================================================================

    @torch.no_grad()
    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t: float,
        prompt_embeds,
        i2v_cond: Optional[dict] = None,
    ) -> torch.Tensor:
        """Predict velocity v_theta(x_t, t).

        Args:
            x_t: (1, C, F, H, W) or (C, F, H, W) current noisy latent
            t: timestep in [0, 1]
            prompt_embeds: (1, L, D) tensor or list of tensors
            i2v_cond: ignored (T2V only)

        Returns:
            v_theta: (1, C, F, H, W) velocity prediction
        """
        if x_t.dim() == 4:
            x_t = x_t.unsqueeze(0)

        x_in = x_t.to(self.dtype)

        # Handle prompt_embeds format
        if isinstance(prompt_embeds, list):
            # Native format: List[Tensor(L, D)] → (1, L, D)
            enc_hidden = prompt_embeds[0].unsqueeze(0) if prompt_embeds[0].dim() == 2 else prompt_embeds[0]
        else:
            enc_hidden = prompt_embeds
        enc_hidden = enc_hidden.to(self.device, self.dtype)

        # Timestep: 0-1000 scale (float, same as scheduler)
        timestep = torch.tensor([t * 1000.0], device=self.device, dtype=self.dtype)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            out = self.model(
                hidden_states=x_in,
                timestep=timestep,
                encoder_hidden_states=enc_hidden,
                return_dict=False,
            )[0]

        return out.float()

    @torch.no_grad()
    def predict_velocity_cfg(
        self,
        x_t: torch.Tensor,
        t: float,
        embeds: dict,
        guidance_scale: float = 5.0,
        i2v_cond: Optional[dict] = None,
    ) -> torch.Tensor:
        """Velocity prediction with Classifier-Free Guidance.

        v_cfg = v_uncond + scale * (v_cond - v_uncond)
        """
        if guidance_scale == 1.0:
            return self.predict_velocity(x_t, t, embeds["prompt_embeds"])

        v_cond = self.predict_velocity(x_t, t, embeds["prompt_embeds"])
        v_uncond = self.predict_velocity(x_t, t, embeds["negative_prompt_embeds"])
        return v_uncond + guidance_scale * (v_cond - v_uncond)
