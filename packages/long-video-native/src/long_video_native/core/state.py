"""Per-(v, h) tile memory cache for spatial × temporal looping.

Used by ``SpatialTiledLoopingPipeline`` to maintain a separate
"previous-segment tail latent" for every spatial tile so that each tile's
temporal seam is stitched against its own past — never another tile's.

For temporal-only generation, the caller can use a single-cell instance
``TileMemory(vertical_tiles=1, horizontal_tiles=1)`` or skip this module
entirely and store the tail latent directly.
"""

from __future__ import annotations

import torch


class TileMemory:
    """Dictionary of ``[B, C, overlap, H_tile, W_tile]`` tensors keyed by
    ``(v_idx, h_idx)``.

    The cache also tracks an optional **anchor** latent per tile — typically
    a slice from the first segment — used as the negative-index latent for
    long-term memory.
    """

    def __init__(self, vertical_tiles: int, horizontal_tiles: int) -> None:
        if vertical_tiles < 1 or horizontal_tiles < 1:
            raise ValueError("tile counts must be >= 1")
        self.vertical_tiles = vertical_tiles
        self.horizontal_tiles = horizontal_tiles
        self._tail: dict[tuple[int, int], torch.Tensor] = {}
        self._anchor: dict[tuple[int, int], torch.Tensor] = {}

    # --- tail (short-term, used for overlap blending) ----------------------

    def set_tail(self, v: int, h: int, latent: torch.Tensor) -> None:
        self._tail[(v, h)] = latent.detach().clone()

    def get_tail(self, v: int, h: int) -> torch.Tensor | None:
        return self._tail.get((v, h))

    # --- anchor (long-term, used for negative-index latent) ----------------

    def set_anchor(self, v: int, h: int, latent: torch.Tensor) -> None:
        """Set the long-term anchor for a tile (typically called once after
        the first segment finishes)."""
        self._anchor[(v, h)] = latent.detach().clone()

    def has_anchor(self, v: int, h: int) -> bool:
        return (v, h) in self._anchor

    def get_anchor(self, v: int, h: int) -> torch.Tensor | None:
        return self._anchor.get((v, h))

    # --- introspection -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TileMemory(grid={self.vertical_tiles}x{self.horizontal_tiles}, "
            f"tails={len(self._tail)}, anchors={len(self._anchor)})"
        )
