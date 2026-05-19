"""Per-tile seed derivation aligned with ComfyUI ``LTXVLoopingSampler``.

Mirror of ``looping_sampler.py::_calculate_tile_seed``::

    tile_seed = base_seed
              + start_index * (vertical_tiles * horizontal_tiles)
              + v * horizontal_tiles
              + h
              + per_tile_offset

where ``start_index`` is the pixel-frame index where the temporal tile
begins (NOT the temporal tile ordinal), ``v`` and ``h`` are the spatial
tile coordinates.
"""

from __future__ import annotations


def calc_tile_seed(
    base_seed: int,
    *,
    start_index: int,
    vertical_tiles: int,
    horizontal_tiles: int,
    v: int = 0,
    h: int = 0,
    per_tile_offset: int = 0,
) -> int:
    """Return the seed for a specific (temporal, spatial) tile.

    Args:
        base_seed: global seed from YAML ``common.seed``.
        start_index: pixel-frame start of the temporal tile.
        vertical_tiles, horizontal_tiles: spatial grid size (>=1).
        v, h: spatial tile coordinates (0-indexed).
        per_tile_offset: optional per-temporal-tile offset (from
            ``per_tile_seed_offsets`` YAML list); 0 disables.
    """
    if vertical_tiles < 1 or horizontal_tiles < 1:
        raise ValueError("vertical_tiles and horizontal_tiles must be >= 1")
    if not (0 <= v < vertical_tiles):
        raise ValueError(f"v ({v}) out of range [0, {vertical_tiles})")
    if not (0 <= h < horizontal_tiles):
        raise ValueError(f"h ({h}) out of range [0, {horizontal_tiles})")
    return (
        base_seed
        + start_index * (vertical_tiles * horizontal_tiles)
        + v * horizontal_tiles
        + h
        + per_tile_offset
    )


def per_tile_offset_for(
    offsets: list[int] | None, temporal_tile_index: int
) -> int:
    """Pick the offset for a temporal tile, repeating the last entry.

    Matches ComfyUI's ``_get_per_tile_value`` behaviour.
    """
    if not offsets:
        return 0
    return offsets[min(temporal_tile_index, len(offsets) - 1)]
