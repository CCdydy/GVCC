"""
turbo_codebook.py — Turbo-DDCM Multi-Atom Thresholding Codebook for Video

Implements Turbo-DDCM (arXiv:2511.06424) adapted for Wan2.1 video latents:
  - Per-step, per-frame Gaussian codebook (K atoms of shape C×H×W)
  - Thresholding: top-M atoms by |⟨z_i, residual⟩| (Eq.13)
  - Signed combination + unit variance normalization (Eq.10)
  - Bitstream: M indices + M sign bits per frame per SDE step

Bitrate example (K=1024, M=32, T_sde=17, F=21):
  17 × 21 × (32 × 10 + 32) = 125,664 bits ≈ 15.7 KB

IMPORTANT: Atoms are generated one-at-a-time via randn(D) to ensure the
random sequence is deterministic regardless of batch size. This guarantees
encoder-decoder consistency even across machines with different VRAM.
PyTorch's randn(N, D) produces different sequences for different N due to
internal RNG block alignment — generating per-atom avoids this.
"""

import torch
import math
import struct
from typing import Tuple, List, Dict
from pathlib import Path


class TurboPerFrameCodebook:
    """Multi-atom thresholding codebook for per-frame video latents.

    At each SDE step, for each temporal frame:
      1. Generate K i.i.d. Gaussian atoms (deterministic via seed)
      2. Compute inner products with residual r = x₀ − x̂₀|t
      3. Select top-M by |inner product| (Eq.13)
      4. Combine with sign(ip) coefficients, normalize to unit std (Eq.10)
    """

    def __init__(
        self,
        K: int = 16384,
        M: int = 32,
        frame_shape: Tuple[int, ...] = (16, 60, 104),
        seed: int = 42,
        device: torch.device = torch.device("cuda"),
        gen_batch: int = 0,  # 0 = auto-tune based on D and available VRAM
    ):
        self.K = K
        self.M = M
        assert M <= K, f"M ({M}) must be <= K ({K})"
        self.frame_shape = frame_shape
        self.D = math.prod(frame_shape)
        self.seed = seed
        self.device = device

        # gen_batch controls how many atoms are held in memory simultaneously
        # for the batched inner product computation. Does NOT affect the
        # random sequence (atoms are always generated one at a time).
        if gen_batch <= 0:
            mem_budget = 64 * 1024 * 1024  # 64 MB — safe for 720p + 14B model
            atom_bytes = self.D * 4  # float32
            self.gen_batch = max(32, min(512, mem_budget // atom_bytes))
        else:
            self.gen_batch = gen_batch

        self.bits_per_index = math.ceil(math.log2(K)) if K > 1 else 1
        self.bits_per_frame_step = M * (self.bits_per_index + 1)  # +1 for sign bit

    def _sf_seed(self, step: int, frame: int) -> int:
        """Deterministic seed for (step, frame) pair."""
        return self.seed * 100003 + step * 10007 + frame

    def _generate_atoms_batch(
        self, gen: torch.Generator, count: int
    ) -> torch.Tensor:
        """Generate `count` atoms one-at-a-time for RNG portability.

        Each atom is generated via randn(D) — the sequence is identical
        regardless of how many atoms are batched for inner products later.

        Returns: (count, D) tensor of atoms.
        """
        atoms = torch.empty(count, self.D, device=self.device)
        for i in range(count):
            atoms[i] = torch.randn(self.D, generator=gen, device=self.device)
        return atoms

    def select_atoms(
        self,
        residual: torch.Tensor,
        step_idx: int,
        frame_idx: int,
        M_override: int = None,
    ) -> Tuple[List[int], List[int], torch.Tensor]:
        """Turbo-DDCM Eq.13: select top-M atoms by |⟨z_i, r⟩|.

        Args:
            residual: (C, H, W) residual = x₀ − x̂₀|t for one temporal frame
            step_idx: SDE step index (0-based, only counting SDE steps)
            frame_idx: temporal frame index in latent
            M_override: use this M instead of self.M (for frame-adaptive allocation)

        Returns:
            indices: sorted list of M atom indices in [0, K)
            signs: list of M signs (+1 or −1)
            combined: (C, H, W) unit-variance combined noise
        """
        M = M_override if M_override is not None else self.M
        r = residual.detach().contiguous().reshape(-1).to(self.device)
        seed_sf = self._sf_seed(step_idx, frame_idx)

        # --- Pass 1: compute all K inner products ---
        gen = torch.Generator(device=self.device).manual_seed(seed_sf)
        all_ips = torch.empty(self.K, device=self.device)

        for b0 in range(0, self.K, self.gen_batch):
            bc = min(self.gen_batch, self.K - b0)
            atoms = self._generate_atoms_batch(gen, bc)
            all_ips[b0:b0 + bc] = atoms @ r
            del atoms

        # Top-M by absolute inner product
        _, top_idx = all_ips.abs().topk(M)
        top_idx_sorted = top_idx.sort().values
        signs_t = torch.sign(all_ips[top_idx_sorted])
        signs_t[signs_t == 0] = 1.0

        idx_list = top_idx_sorted.cpu().tolist()
        sign_list = signs_t.cpu().int().tolist()

        # --- Pass 2: regenerate selected atoms and combine ---
        combined = self._regenerate_and_combine(seed_sf, idx_list, signs_t)
        return idx_list, sign_list, combined

    def reconstruct(
        self,
        indices: List[int],
        signs: List[int],
        step_idx: int,
        frame_idx: int,
    ) -> torch.Tensor:
        """Reconstruct combined noise from indices + signs (decoder side).

        Returns: (C, H, W) unit-variance combined noise.
        """
        seed_sf = self._sf_seed(step_idx, frame_idx)
        signs_t = torch.tensor(signs, device=self.device, dtype=torch.float32)
        return self._regenerate_and_combine(seed_sf, indices, signs_t)

    def _regenerate_and_combine(
        self,
        seed_sf: int,
        indices: List[int],
        signs_t: torch.Tensor,
    ) -> torch.Tensor:
        """Regenerate selected atoms from seed, combine with signs, normalize."""
        gen = torch.Generator(device=self.device).manual_seed(seed_sf)
        target = set(indices)
        found: Dict[int, torch.Tensor] = {}

        for gi in range(self.K):
            atom = torch.randn(self.D, generator=gen, device=self.device)
            if gi in target:
                found[gi] = atom
            del atom
            if len(found) >= len(indices):
                break

        # Stack in index order and combine
        stacked = torch.stack([found[i] for i in indices])   # (M, D)
        combined = (signs_t.unsqueeze(1) * stacked).sum(0)    # (D,)

        # Normalize to unit variance (Eq.10)
        std = combined.std()
        if std > 1e-8:
            combined = combined / std

        return combined.view(self.frame_shape)


# ==================================================================
# Bitstream I/O for Turbo-DDCM
# ==================================================================

class TurboBitstream:
    """Binary I/O for Turbo-DDCM multi-atom indices + signs.

    Format:
      Header: magic(4) + 10×uint32 + prompt_len(uint32) + frame_shape + prompt
      Data:   for each SDE step, for each frame:
                M indices (uint16 if K≤65536, else uint32)
                ceil(M/8) bytes of sign bits
    """

    @staticmethod
    def save(
        filepath: str,
        step_data: List[List[Tuple[List[int], List[int]]]],
        K: int, M: int,
        num_sde_steps: int, num_ddim_steps: int,
        num_latent_frames: int, seed: int,
        frame_shape: Tuple[int, ...],
        prompt: str = "",
        num_frames: int = 81, height: int = 480, width: int = 832,
    ):
        """Save compressed Turbo-DDCM bitstream.

        Args:
            step_data: [sde_step][frame] = (indices, signs)
        """
        data = bytearray(b"TDCM")  # magic
        prompt_b = prompt.encode("utf-8")

        # Header: 10 uint32 values
        data.extend(struct.pack("<10I",
            K, M, num_sde_steps, num_ddim_steps,
            num_latent_frames, seed, len(frame_shape),
            num_frames, height, width))
        data.extend(struct.pack("<I", len(prompt_b)))
        for d in frame_shape:
            data.extend(struct.pack("<I", d))
        data.extend(prompt_b)

        # Index format
        idx_fmt = "<H" if K <= 65536 else "<I"

        # Data: indices + sign bits per frame per step
        # Each frame entry stores its own M_actual (uint16) to support M_tail
        for step in step_data:
            for indices, signs in step:
                m_actual = len(indices)
                data.extend(struct.pack("<H", m_actual))
                for idx in indices:
                    data.extend(struct.pack(idx_fmt, idx))
                sb = bytearray((m_actual + 7) // 8)
                for j, s in enumerate(signs):
                    if s > 0:
                        sb[j // 8] |= (1 << (j % 8))
                data.extend(sb)

        Path(filepath).write_bytes(bytes(data))
        print(f"Turbo-DDCM: {len(data)} bytes -> {filepath}")

    @staticmethod
    def load(filepath: str) -> dict:
        """Load Turbo-DDCM bitstream."""
        raw = Path(filepath).read_bytes()
        assert raw[:4] == b"TDCM", f"Invalid magic: {raw[:4]}"
        off = 4

        vals = struct.unpack_from("<10I", raw, off)
        K, M, n_sde, n_ddim, n_lat, seed, ndim, n_fr, h, w = vals
        off += 40

        plen = struct.unpack_from("<I", raw, off)[0]
        off += 4

        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack_from("<I", raw, off)[0])
            off += 4

        prompt = raw[off:off + plen].decode("utf-8")
        off += plen

        idx_fmt = "<H" if K <= 65536 else "<I"
        idx_sz = struct.calcsize(idx_fmt)

        step_data = []
        for _ in range(n_sde):
            frame_data = []
            for _ in range(n_lat):
                m_actual = struct.unpack_from("<H", raw, off)[0]
                off += 2
                indices = []
                for _ in range(m_actual):
                    indices.append(struct.unpack_from(idx_fmt, raw, off)[0])
                    off += idx_sz
                sign_bytes_len = (m_actual + 7) // 8
                sr = raw[off:off + sign_bytes_len]
                off += sign_bytes_len
                signs = []
                for j in range(m_actual):
                    bit = (sr[j // 8] >> (j % 8)) & 1
                    signs.append(1 if bit else -1)
                frame_data.append((indices, signs))
            step_data.append(frame_data)

        return dict(
            K=K, M=M,
            num_sde_steps=n_sde, num_ddim_steps=n_ddim,
            num_latent_frames=n_lat, seed=seed,
            frame_shape=tuple(shape), prompt=prompt,
            num_frames=n_fr, height=h, width=w,
            step_data=step_data,
        )
