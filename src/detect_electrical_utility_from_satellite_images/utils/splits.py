"""Utilities for deterministic dataset splits.

The project uses preprocessed patch filenames that encode the source tile and
patch coordinates. These helpers support tile-level splits (no leakage across
patches from the same tile) and optional persistence to disk.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_PATCH_STEM_RE = re.compile(r"^(?P<tile_id>.+)_(?P<patch_idx>\d{6})_x(?P<x>\d+)_y(?P<y>\d+)$")


def patch_path_to_tile_id(patch_path: Path) -> str:
    """Extract tile id from a patch filename.

    Args:
        patch_path: Path like `.../images/NZ_Dunedin_1_000123_x500_y0.png`.

    Returns:
        Tile id like `NZ_Dunedin_1`.

    Raises:
        ValueError: If filename does not match the expected pattern.
    """
    stem = patch_path.stem
    match = _PATCH_STEM_RE.match(stem)
    if match is None:
        raise ValueError(
            f"Patch filename does not match expected pattern: {patch_path.name}"
        )
    return match.group("tile_id")


@dataclass(frozen=True)
class TileSplit:
    """A deterministic tile-level split."""

    train_tiles: list[str]
    val_tiles: list[str]


def _default_split_path(patches_dir: Path, split_file_name: str) -> Path:
    return patches_dir / split_file_name


def load_or_create_tile_split(
    *,
    patches_dir: Path,
    image_files: Iterable[Path],
    seed: int,
    val_split: float,
    persist: bool,
    split_file_name: str,
    min_val_tiles: int,
) -> TileSplit:
    """Load a persisted tile split or create a new one.

    Args:
        patches_dir: Region directory like `preprocessed_data/patches/NZ_Dunedin`.
        image_files: Patch image paths.
        seed: RNG seed.
        val_split: Fraction of tiles to reserve for validation.
        persist: Whether to persist split to disk.
        split_file_name: File name under `patches_dir`.
        min_val_tiles: Minimum number of validation tiles (when possible).

    Returns:
        TileSplit with train/val tile ids.

    Raises:
        ValueError: If there are no tiles found.
    """
    patches_dir = Path(patches_dir)
    split_path = _default_split_path(patches_dir, split_file_name)

    if persist and split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        train_tiles = list(payload.get("train_tiles", []))
        val_tiles = list(payload.get("val_tiles", []))
        if train_tiles or val_tiles:
            return TileSplit(train_tiles=train_tiles, val_tiles=val_tiles)

    tile_ids: list[str] = []
    seen: set[str] = set()
    for img_path in image_files:
        tile_id = patch_path_to_tile_id(Path(img_path))
        if tile_id not in seen:
            seen.add(tile_id)
            tile_ids.append(tile_id)

    if not tile_ids:
        raise ValueError(f"No tiles found under: {patches_dir}")

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(tile_ids)

    n_tiles = len(tile_ids)
    if n_tiles < 2:
        # Caller should fallback to patch-level splitting.
        return TileSplit(train_tiles=tile_ids, val_tiles=[])

    n_val_tiles = max(min_val_tiles, int(math.ceil(n_tiles * val_split)))
    n_val_tiles = min(n_val_tiles, n_tiles - 1)  # keep at least one train tile

    val_tiles = sorted(tile_ids[:n_val_tiles])
    train_tiles = sorted(tile_ids[n_val_tiles:])

    split = TileSplit(train_tiles=train_tiles, val_tiles=val_tiles)

    if persist:
        split_path.write_text(
            json.dumps(
                {
                    "split_level": "tile",
                    "seed": seed,
                    "val_split": val_split,
                    "train_tiles": split.train_tiles,
                    "val_tiles": split.val_tiles,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return split


def indices_for_tiles(image_files: list[Path], tiles: set[str]) -> list[int]:
    """Map tile ids to dataset indices (based on image file list)."""
    indices: list[int] = []
    for idx, img_path in enumerate(image_files):
        tile_id = patch_path_to_tile_id(img_path)
        if tile_id in tiles:
            indices.append(idx)
    return indices
