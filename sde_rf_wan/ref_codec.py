"""
ref_codec.py — Reference frame compression for GOP boundaries

Supports:
  - WebP (legacy, fast)
  - CompressAI cheng2020-attn (learned, much better at low bitrates)

Usage:
    from sde_rf_wan.ref_codec import compress_ref, decompress_ref

    # Compress
    data, nbytes = compress_ref(image, codec="compressai", quality=4)

    # Decompress
    decoded = decompress_ref(data, codec="compressai")
"""

import io
import struct
import torch
import numpy as np
from PIL import Image


# ==================================================================
# WebP
# ==================================================================

def compress_ref_webp(image: Image.Image, quality: int = 30):
    """Compress with WebP. Returns (decoded_image, raw_bytes, num_bytes)."""
    buf = io.BytesIO()
    image.save(buf, format="WebP", quality=quality)
    raw = buf.getvalue()
    nbytes = len(raw)
    decoded = Image.open(io.BytesIO(raw)).copy()
    return decoded, raw, nbytes


def decompress_ref_webp(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).copy()


# ==================================================================
# CompressAI (cheng2020-attn)
# ==================================================================

_compressai_cache = {}  # model cache to avoid reloading


def _get_compressai_model(model_name: str, quality: int, device: str = "cuda"):
    key = (model_name, quality)
    if key not in _compressai_cache:
        from compressai.zoo import models
        net = models[model_name](quality=quality, pretrained=True).eval().to(device)
        _compressai_cache[key] = net
    return _compressai_cache[key]


def compress_ref_compressai(
    image: Image.Image,
    model_name: str = "cheng2020-attn",
    quality: int = 4,
    device: str = "cuda",
):
    """Compress with CompressAI learned codec.

    Returns (decoded_image, raw_bytes, num_bytes).
    raw_bytes is a self-contained binary blob for decompression.
    """
    net = _get_compressai_model(model_name, quality, device)

    x = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)

    _, _, h, w = x.shape
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        out = net.compress(x)
        rec = net.decompress(out["strings"], out["shape"])

    # Pack into self-contained binary format
    # Header: h(u16) w(u16) pad_h(u8) pad_w(u8) n_strings(u8)
    # Then for each string list: n_items(u8), then for each item: len(u32) + data
    blob = bytearray()
    blob.extend(struct.pack('<HH BB B', h, w, pad_h, pad_w, len(out["strings"])))
    # Store shape
    blob.extend(struct.pack('<B', len(out["shape"])))
    for s in out["shape"]:
        blob.extend(struct.pack('<I', s))
    # Store strings
    for string_list in out["strings"]:
        blob.extend(struct.pack('<B', len(string_list)))
        for s in string_list:
            blob.extend(struct.pack('<I', len(s)))
            blob.extend(s)

    nbytes = len(blob)

    # Decode
    x_hat = rec["x_hat"][:, :, :h, :w].clamp(0, 1)
    decoded_np = (x_hat[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    decoded = Image.fromarray(decoded_np)

    return decoded, bytes(blob), nbytes


def decompress_ref_compressai(
    data: bytes,
    model_name: str = "cheng2020-attn",
    quality: int = 4,
    device: str = "cuda",
) -> Image.Image:
    """Decompress CompressAI blob back to PIL image."""
    net = _get_compressai_model(model_name, quality, device)

    off = 0
    h, w, pad_h, pad_w, n_string_lists = struct.unpack_from('<HH BB B', data, off)
    off += 7

    n_shape = struct.unpack_from('<B', data, off)[0]; off += 1
    shape = []
    for _ in range(n_shape):
        shape.append(struct.unpack_from('<I', data, off)[0]); off += 4

    strings = []
    for _ in range(n_string_lists):
        n_items = struct.unpack_from('<B', data, off)[0]; off += 1
        items = []
        for _ in range(n_items):
            slen = struct.unpack_from('<I', data, off)[0]; off += 4
            items.append(data[off:off+slen])
            off += slen
        strings.append(items)

    with torch.no_grad():
        rec = net.decompress(strings, shape)

    x_hat = rec["x_hat"][:, :, :h, :w].clamp(0, 1)
    decoded_np = (x_hat[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(decoded_np)


# ==================================================================
# Unified interface
# ==================================================================

def compress_ref(image: Image.Image, codec: str = "compressai", **kwargs):
    """Compress reference frame.

    Args:
        image: PIL Image
        codec: "webp" or "compressai"
        **kwargs: codec-specific params
            webp: quality (int, default 30)
            compressai: model_name (str), quality (int, default 4), device (str)

    Returns:
        decoded: PIL Image (decoded reference)
        data: bytes (compressed data for transmission)
        nbytes: int (total bytes)
    """
    if codec == "webp":
        quality = kwargs.get("quality", 30)
        return compress_ref_webp(image, quality)
    elif codec == "compressai":
        model_name = kwargs.get("model_name", "cheng2020-attn")
        quality = kwargs.get("quality", 4)
        device = kwargs.get("device", "cuda")
        return compress_ref_compressai(image, model_name, quality, device)
    else:
        raise ValueError(f"Unknown codec: {codec}")


def decompress_ref(data: bytes, codec: str = "compressai", **kwargs) -> Image.Image:
    """Decompress reference frame."""
    if codec == "webp":
        return decompress_ref_webp(data)
    elif codec == "compressai":
        model_name = kwargs.get("model_name", "cheng2020-attn")
        quality = kwargs.get("quality", 4)
        device = kwargs.get("device", "cuda")
        return decompress_ref_compressai(data, model_name, quality, device)
    else:
        raise ValueError(f"Unknown codec: {codec}")
