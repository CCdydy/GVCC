"""
wan_flf2v_wrapper.py — Wan2.1 First-Last-Frame-to-Video (FLF2V) Model Wrapper

Extends WanWrapper for FLF2V conditioning: encodes BOTH first and last frames
via CLIP vision + VAE with a first-last mask.

Key method:
  encode_first_last_frames(first, last, num_frames, h, w)
    -> dict(clip_fea, y)  compatible with TurboDDCMWanPipeline's i2v_cond

CLIP conditioning:
  clip.visual([first[:, None, :, :], last[:, None, :, :]])
  (two B=1 tensors of shape (3, 1, H_clip, W_clip))

VAE conditioning:
  video_cond = concat(first_resized, zeros(F-2), last_resized)  -> (3, F, H, W)
  y = vae.encode([video_cond])
  mask = 1 at first and last frame, 0 in middle
  y = concat(mask, y) along channel dim -> (C+4, F_lat, H_lat, W_lat)
"""

import os
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from PIL import Image

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.clip import CLIPModel


class WanFLF2VWrapper:
    """Thin wrapper around native Wan2.1 modules for FLF2V (first-last-frame to video).

    Mirrors WanWrapper but uses config_name='flf2v-14B' and encodes
    two boundary frames instead of one.
    """

    def __init__(
        self,
        checkpoint_dir: str = "./Wan2.1-FLF2V-14B-720P",
        config_name: str = "flf2v-14B",
        flow_shift: float = 5.0,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.config_name = config_name
        self.flow_shift = flow_shift
        self.config = WAN_CONFIGS[config_name]
        # FLF2V always uses CLIP (like I2V)
        self.is_i2v = True
        self.device = None
        self.dtype = None

    def load(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """Load Wan2.1 FLF2V model components."""
        self.device = torch.device(device)
        self.dtype = dtype

        print(f"Loading Wan2.1 FLF2V ({self.config_name}) from {self.checkpoint_dir}...")

        # Load T5 text encoder
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=self.device,
            checkpoint_path=os.path.join(self.checkpoint_dir, self.config.t5_checkpoint),
            tokenizer_path=os.path.join(self.checkpoint_dir, self.config.t5_tokenizer),
        )

        # Load VAE
        self.vae = WanVAE(
            vae_pth=os.path.join(self.checkpoint_dir, self.config.vae_checkpoint),
            device=self.device,
        )

        # Load CLIP vision encoder
        self.clip = CLIPModel(
            dtype=self.config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(self.checkpoint_dir, self.config.clip_checkpoint),
            tokenizer_path=os.path.join(self.checkpoint_dir, self.config.clip_tokenizer),
        )

        # Load DiT transformer
        self.model = WanModel.from_pretrained(self.checkpoint_dir)
        self.model.eval().requires_grad_(False)
        self.model.to(dtype=dtype).to(self.device)

        # VAE config
        self.vae_stride = self.config.vae_stride  # (4, 8, 8)
        self.vae_temporal_factor = self.vae_stride[0]
        self.vae_spatial_factor = self.vae_stride[1]
        self.latent_channels = self.vae.model.z_dim  # 16
        self.patch_size = self.config.patch_size  # (1, 2, 2)
        self.sample_neg_prompt = self.config.sample_neg_prompt

        print(f"Wan2.1 FLF2V loaded. Device={device}, dtype={dtype}")
        print(f"  VAE compression: {self.vae_temporal_factor}x{self.vae_spatial_factor}x{self.vae_spatial_factor}")
        print(f"  Latent channels: {self.latent_channels}")

    # ================================================================
    # Internal: compute seq_len for model forward
    # ================================================================

    def _compute_seq_len(self, latent_shape: Tuple[int, ...]) -> int:
        """Compute sequence length for positional encoding.

        Matches WanFLF2V.generate() logic:
            seq_len = ceil(H_lat * W_lat / (patch_h * patch_w) * F_lat)
        """
        _, F_lat, H_lat, W_lat = latent_shape
        return math.ceil(
            (H_lat * W_lat) / (self.patch_size[1] * self.patch_size[2])
            * F_lat
        )

    # ================================================================
    # Text encoding
    # ================================================================

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
    ) -> dict:
        """Encode text prompt via UMT5.

        Returns dict with list-of-tensor format matching native Wan interface.
        """
        if not negative_prompt:
            negative_prompt = self.sample_neg_prompt

        context = self.text_encoder([prompt], self.device)
        context_null = self.text_encoder([negative_prompt], self.device)

        return {
            "prompt_embeds": context,           # List[Tensor(seq_len, 4096)]
            "negative_prompt_embeds": context_null,
        }

    # ================================================================
    # VAE encode / decode
    # ================================================================

    def get_latent_shape(
        self,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
    ) -> Tuple[int, ...]:
        """Compute latent shape: (C, F_latent, H_latent, W_latent).
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

    @torch.no_grad()
    def encode_video(
        self,
        frames,
        height: int = 480,
        width: int = 832,
    ) -> torch.Tensor:
        """Encode video frames to 3D latent.

        Args:
            frames: list of PIL Images (num_frames)
            height, width: target resolution

        Returns:
            latent: (1, C, F_lat, H_lat, W_lat)
        """
        processed = []
        for frame in frames:
            frame = frame.resize((width, height), Image.LANCZOS)
            arr = np.array(frame).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
            processed.append(t)

        # (3, F, H, W) -- native Wan VAE format
        video_tensor = torch.stack(processed, dim=1).to(
            device=self.device, dtype=torch.float32
        )
        # Normalize to [-1, 1]
        video_tensor = 2.0 * video_tensor - 1.0

        # Native VAE encode: List[(3, F, H, W)] -> List[(C, F_lat, H_lat, W_lat)]
        latent = self.vae.encode([video_tensor])[0]
        latent = latent.float().unsqueeze(0)  # (1, C, F_lat, H_lat, W_lat)

        # Store raw dims for decode cropping, then pad to even for DiT
        _, _, _, h_lat, w_lat = latent.shape
        self._raw_latent_h = h_lat
        self._raw_latent_w = w_lat
        pad_h = h_lat % 2
        pad_w = w_lat % 2
        if pad_h or pad_w:
            latent = torch.nn.functional.pad(latent, (0, pad_w, 0, pad_h, 0, 0), mode='replicate')
        return latent

    @torch.no_grad()
    def decode_latent(
        self,
        latent: torch.Tensor,
    ):
        """Decode 3D latent to video frames.

        Args:
            latent: (1, C, F_lat, H_lat, W_lat) or (C, F_lat, H_lat, W_lat)

        Returns:
            list of PIL Images
        """
        if latent.dim() == 5:
            latent = latent.squeeze(0)  # (C, F_lat, H_lat, W_lat)

        # Crop any padding from encode (odd latent dims padded to even)
        raw_h = self._raw_latent_h if hasattr(self, '_raw_latent_h') else latent.shape[2]
        raw_w = self._raw_latent_w if hasattr(self, '_raw_latent_w') else latent.shape[3]
        latent = latent[:, :, :raw_h, :raw_w]

        # Native VAE decode
        video = self.vae.decode([latent.float()])[0]  # (3, F, H, W) in [-1, 1]

        video = (video / 2.0 + 0.5).clamp(0, 1)
        video = video.permute(1, 2, 3, 0).cpu().float().numpy()  # (F, H, W, 3)

        frames = []
        for f_idx in range(video.shape[0]):
            img = (video[f_idx] * 255).astype(np.uint8)
            frames.append(Image.fromarray(img))

        return frames

    # ================================================================
    # FLF2V: first + last frame conditioning
    # ================================================================

    @torch.no_grad()
    def encode_first_last_frames(
        self,
        first_frame: Image.Image,
        last_frame: Image.Image,
        num_frames: int,
        height: int,
        width: int,
    ) -> dict:
        """Encode first and last frames for FLF2V conditioning.

        Follows the exact logic from wan/first_last_frame2video.py WanFLF2V.generate():
          1. CLIP: visual([first[:, None, :, :], last[:, None, :, :]])
          2. VAE: encode concat(first_resized, zeros(F-2), last_resized)
          3. Mask: 1 at first and last frame, 0 in middle
          4. y = concat(mask, vae_encoded) along channel dim

        Args:
            first_frame: PIL Image (first frame of GOP)
            last_frame: PIL Image (last frame of GOP)
            num_frames: number of video frames (e.g., 33)
            height, width: target resolution

        Returns:
            dict with:
                clip_fea: CLIP visual features of both frames
                y: (C+4, F_lat, H_lat, W_lat) VAE conditioning + mask
        """
        F_total = num_frames
        h, w = height, width
        lat_h = h // self.vae_spatial_factor
        lat_w = w // self.vae_spatial_factor
        F_lat = (F_total - 1) // self.vae_temporal_factor + 1

        # Prepare image tensors: (3, H, W) in [-1, 1]
        first_img = first_frame.resize((w, h), Image.LANCZOS)
        first_t = torch.from_numpy(
            np.array(first_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        first_t = 2.0 * first_t - 1.0  # [-1, 1]

        last_img = last_frame.resize((w, h), Image.LANCZOS)
        last_t = torch.from_numpy(
            np.array(last_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        last_t = 2.0 * last_t - 1.0  # [-1, 1]

        # ---- CLIP visual features (both frames) ----
        clip_fea = self.clip.visual([
            first_t.to(self.device)[:, None, :, :],
            last_t.to(self.device)[:, None, :, :],
        ])

        # ---- VAE conditioning: first + zeros + last ----
        # Resize via bicubic interpolation (matching native code exactly)
        first_resized = F.interpolate(
            first_t[None], size=(h, w), mode='bicubic'
        ).transpose(0, 1)  # (3, 1, H, W) -> keep (3, 1, H, W) via transpose

        last_resized = F.interpolate(
            last_t[None], size=(h, w), mode='bicubic'
        ).transpose(0, 1)

        video_cond = torch.cat([
            first_resized,                               # (3, 1, H, W)
            torch.zeros(3, F_total - 2, h, w),           # (3, F-2, H, W)
            last_resized,                                 # (3, 1, H, W)
        ], dim=1).to(self.device)                         # (3, F, H, W)

        y = self.vae.encode([video_cond])[0]  # (C, F_lat, H_lat, W_lat)

        # ---- Mask: first and last = 1, middle = 0 ----
        # Native code uses F_total=81 hardcoded in mask shape, but the actual
        # number of frames is F_total. We follow the native logic exactly:
        # msk = ones(1, F_total, lat_h, lat_w); msk[:, 1:-1] = 0
        msk = torch.ones(1, F_total, lat_h, lat_w, device=self.device)
        msk[:, 1:-1] = 0  # first and last = 1, middle = 0

        # Reshape mask to match VAE temporal compression (4x):
        #   First: expand frame 0 to 4 copies (VAE compresses first frame differently)
        #   concat(repeat(msk[:, 0:1], 4), msk[:, 1:])
        msk = torch.cat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)
        # Now msk has shape (1, F_total+3, lat_h, lat_w)
        # Reshape to (1, F_lat, 4, lat_h, lat_w) then transpose to (4, F_lat, lat_h, lat_w)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]  # (4, F_lat, H_lat, W_lat)

        # Concat mask + VAE encoding
        y = torch.cat([msk, y], dim=0)  # (C+4, F_lat, H_lat, W_lat)

        # Pad to even dims to match encode_video padding for DiT
        pad_h = lat_h % 2
        pad_w = lat_w % 2
        if pad_h or pad_w:
            y = torch.nn.functional.pad(y, (0, pad_w, 0, pad_h, 0, 0), mode='replicate')

        return {
            "clip_fea": clip_fea,  # Tensor (CLIP features for both frames)
            "y": y,                # (C+4, F_lat, H_lat_padded, W_lat_padded)
        }

    # ================================================================
    # I2V-compatible encode_image (for TurboDDCMWanPipeline compatibility)
    # ================================================================

    def encode_image(self, image: Image.Image, num_frames: int,
                     height: int, width: int) -> dict:
        """Encode a single reference image (I2V-style).

        Provided for interface compatibility but FLF2V should use
        encode_first_last_frames() instead. This encodes only the first frame
        with zeros for the rest (mask=1 at first frame only).
        """
        return self.encode_first_last_frames(
            image, image, num_frames, height, width
        )

    # ================================================================
    # Velocity prediction (the core interface for SDE-RF)
    # ================================================================

    @torch.no_grad()
    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t: float,
        prompt_embeds,
        i2v_cond: Optional[dict] = None,
    ) -> torch.Tensor:
        """Predict velocity v_theta(x_t, t) from Wan's DiT.

        Args:
            x_t: (1, C, F, H, W) or (C, F, H, W) current noisy latent
            t: timestep in [0, 1]
            prompt_embeds: List[Tensor(L, D)] from encode_prompt
            i2v_cond: dict with clip_fea and y (FLF2V conditioning)

        Returns:
            v_theta: (1, C, F, H, W) velocity prediction
        """
        from torch.cuda import amp

        # Prepare input: native model takes List[Tensor(C, F, H, W)]
        if x_t.dim() == 5:
            x_in = [x_t.squeeze(0).to(self.dtype)]
        else:
            x_in = [x_t.to(self.dtype)]

        # Context: List[Tensor(L, D)]
        if isinstance(prompt_embeds, list):
            context = [c.to(self.device, self.dtype) for c in prompt_embeds]
        else:
            context = [prompt_embeds.squeeze(0).to(self.device, self.dtype)]

        # Timestep: Wan model expects [0, 1000] scale
        timestep = torch.tensor([t * 1000.0], device=self.device, dtype=self.dtype)

        # Sequence length for positional encoding
        latent_shape = x_in[0].shape  # (C, F, H, W)
        seq_len = self._compute_seq_len(latent_shape)

        # FLF2V extra args (same interface as I2V)
        extra_kwargs = {}
        if i2v_cond is not None:
            extra_kwargs["clip_fea"] = i2v_cond["clip_fea"].to(self.device, self.dtype)
            extra_kwargs["y"] = [i2v_cond["y"].to(self.device, self.dtype)]

        with amp.autocast(dtype=self.dtype):
            out = self.model(
                x_in,
                t=timestep,
                context=context,
                seq_len=seq_len,
                **extra_kwargs,
            )[0]

        return out.float().unsqueeze(0)  # (1, C, F, H, W)

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
        For guidance=1.0, only runs one forward pass (v_cond).
        """
        if guidance_scale == 1.0:
            return self.predict_velocity(x_t, t, embeds["prompt_embeds"], i2v_cond)

        v_cond = self.predict_velocity(x_t, t, embeds["prompt_embeds"], i2v_cond)
        v_uncond = self.predict_velocity(x_t, t, embeds["negative_prompt_embeds"], i2v_cond)
        return v_uncond + guidance_scale * (v_cond - v_uncond)
