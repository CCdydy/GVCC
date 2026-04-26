"""
Shared UVG dataset discovery helpers.

Supported layouts (all recursive):
1) data/uvg/<Sequence>_..._RAW/<Sequence>_... .yuv
2) data/uvg/*.yuv
3) data/UVG/*.yuv
4) data/*.yuv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


UVG_ORDER = [
    "Beauty",
    "Bosphorus",
    "HoneyBee",
    "Jockey",
    "ReadySteadyGo",
    "ShakeNDry",
    "YachtRide",
]

_ALIASES = {
    "beauty": "Beauty",
    "bosphorus": "Bosphorus",
    "honeybee": "HoneyBee",
    "jockey": "Jockey",
    "readysteadygo": "ReadySteadyGo",
    "readysetgo": "ReadySteadyGo",
    "shakendry": "ShakeNDry",
    "yachtride": "YachtRide",
}


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _canonical_sequence_name(name: str) -> str:
    key = _norm_token(name)
    return _ALIASES.get(key, name)


def _sequence_name_from_path(yuv_path: Path) -> str:
    candidates = [yuv_path.stem, yuv_path.parent.name, yuv_path.parent.parent.name]
    for raw in candidates:
        token = _norm_token(raw)
        if not token:
            continue
        for alias_key, canonical in _ALIASES.items():
            if alias_key in token:
                return canonical

    # Fallback to filename prefix before first underscore.
    return _canonical_sequence_name(yuv_path.stem.split("_")[0])


def _candidate_roots(data_dir: str) -> List[Path]:
    base = Path(data_dir).expanduser()
    roots = [base, base / "uvg", base / "UVG"]

    unique: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _discover_yuv_files(data_dir: str) -> List[Path]:
    found: Dict[str, Path] = {}
    for root in _candidate_roots(data_dir):
        if root.is_file() and root.suffix.lower() == ".yuv":
            key = str(root.resolve(strict=False))
            found[key] = root
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*.yuv"):
            key = str(path.resolve(strict=False))
            found[key] = path
    return sorted(found.values(), key=lambda p: str(p))


def find_uvg_sequences(data_dir: str) -> List[Tuple[str, str]]:
    """Return discovered UVG sequences as (sequence_name, yuv_path)."""
    seq_to_path: Dict[str, str] = {}

    for yuv_path in _discover_yuv_files(data_dir):
        seq = _sequence_name_from_path(yuv_path)
        # Keep the first discovered file per sequence to avoid duplicates.
        if seq not in seq_to_path:
            seq_to_path[seq] = str(yuv_path)

    ordered: List[Tuple[str, str]] = []
    for seq in UVG_ORDER:
        if seq in seq_to_path:
            ordered.append((seq, seq_to_path.pop(seq)))

    for seq in sorted(seq_to_path.keys()):
        ordered.append((seq, seq_to_path[seq]))

    return ordered


def find_uvg_sequence(data_dir: str, sequence: str) -> Optional[str]:
    """Return path for one sequence name, or None if not found."""
    target = _canonical_sequence_name(sequence)
    for seq, path in find_uvg_sequences(data_dir):
        if _canonical_sequence_name(seq) == target:
            return path
    return None

