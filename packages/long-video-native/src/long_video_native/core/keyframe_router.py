"""Global keyframe → per-temporal-tile routing.

ComfyUI's ``LTXVLoopingSampler`` accepts a flat list of pixel-frame indices
(``optional_cond_image_indices``) and works out which temporal tile each
keyframe lands in and what its in-tile index is. We replicate that mapping
here so users can specify keyframes in **global pixel-frame coordinates**
instead of having to bucket them by segment.

Algorithm — direct port of
``looping_sampler.py::_calculate_keyframe_per_tile_indices`` (lines
~580-680). All units are pixel frames unless suffixed ``_latent``.

Notes
-----
* The first temporal tile covers pixel frames ``[0, tile_size - 8]``
  (size = ``tile_size - 7``).
* Subsequent tiles start at ``n * (tile_size - overlap) - 7`` and end at
  ``tile_size + n * (tile_size - overlap) - 1 - 7`` — the ``-7`` reflects
  the fact that the first latent of each extension tile is re-interpreted
  as 1 pixel frame inside the tile (see ComfyUI source comment).
* If a keyframe lands in the *overlap region* of tile ``n > 0`` (i.e.
  ``in_tile_index < temporal_overlap``), it is reassigned to tile ``n-1``
  with the corresponding in-tile index.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeyframeRoute:
    """A single keyframe routed to a specific temporal tile.

    Attributes:
        temporal_tile_index: which temporal tile (0-indexed) this keyframe
            belongs to.
        in_tile_pixel_index: pixel-frame index *within* that tile, ready to
            pass to ``ImageConditioningInput.frame_idx``.
        global_pixel_index: the original global pixel-frame index (kept for
            logging / debugging).
        meta: pass-through opaque data (e.g. a path to the image, strength).
    """

    temporal_tile_index: int
    in_tile_pixel_index: int
    global_pixel_index: int
    meta: object = None


def route_keyframes(
    keyframe_pixel_indices: list[int],
    *,
    tile_size_pixel: int,
    overlap_pixel: int,
    total_pixel_frames: int,
    metas: list[object] | None = None,
) -> list[KeyframeRoute]:
    """Route a flat list of pixel-frame keyframe indices to per-tile positions.

    Args:
        keyframe_pixel_indices: global pixel-frame indices, e.g. ``[0, 120, 480]``.
        tile_size_pixel: ``temporal_tile_size`` in pixel frames.
        overlap_pixel: ``temporal_overlap`` in pixel frames.
        total_pixel_frames: total pixel frames in the assembled video (used
            for bounds checks).
        metas: optional list aligned with ``keyframe_pixel_indices`` whose
            entries are attached to each :class:`KeyframeRoute`.

    Returns:
        One :class:`KeyframeRoute` per *valid* keyframe (out-of-range ones
        are dropped with a print-like warning).
    """
    if tile_size_pixel <= 0:
        raise ValueError("tile_size_pixel must be > 0")
    if overlap_pixel < 0 or overlap_pixel >= tile_size_pixel:
        raise ValueError(
            f"overlap_pixel ({overlap_pixel}) must be in [0, tile_size_pixel)"
        )
    if metas is None:
        metas = [None] * len(keyframe_pixel_indices)
    if len(metas) != len(keyframe_pixel_indices):
        raise ValueError("metas length must match keyframe_pixel_indices length")

    tile_step = tile_size_pixel - overlap_pixel
    routes: list[KeyframeRoute] = []

    for kf_idx, meta in zip(keyframe_pixel_indices, metas, strict=True):
        if kf_idx >= total_pixel_frames or kf_idx < 0:
            # Mirror ComfyUI's silent-skip with a print; we just skip.
            continue

        # First tile (index 0): covers pixel frames [0, tile_size - 8]
        if kf_idx < tile_size_pixel - 7:
            routes.append(
                KeyframeRoute(
                    temporal_tile_index=0,
                    in_tile_pixel_index=kf_idx,
                    global_pixel_index=kf_idx,
                    meta=meta,
                )
            )
            continue

        # Subsequent tiles: walk forward until the keyframe is bounded.
        tile_index = 1
        placed = False
        # Hard cap to avoid pathological infinite loops if inputs are weird.
        max_iters = max(1, total_pixel_frames // max(tile_step, 1) + 2)
        for _ in range(max_iters):
            tile_start = tile_index * tile_step - 7
            tile_end = tile_size_pixel + tile_index * tile_step - 1 - 7

            if kf_idx <= tile_end:
                in_tile_index = kf_idx - tile_start - 7

                # If the keyframe is in the overlap region of this tile,
                # ComfyUI re-attributes it to the previous tile.
                if in_tile_index < overlap_pixel and tile_index >= 1:
                    prev_index = tile_index - 1
                    if prev_index == 0:
                        in_tile_index = kf_idx
                    else:
                        prev_start = tile_start - tile_step
                        in_tile_index = kf_idx - prev_start - 7
                    tile_index = prev_index

                routes.append(
                    KeyframeRoute(
                        temporal_tile_index=tile_index,
                        in_tile_pixel_index=in_tile_index,
                        global_pixel_index=kf_idx,
                        meta=meta,
                    )
                )
                placed = True
                break
            tile_index += 1

        if not placed:
            # Shouldn't happen if bounds checks above hold, but be defensive.
            continue

    return routes


def total_pixel_frames(
    n_segments: int, tile_size_pixel: int, overlap_pixel: int
) -> int:
    """Compute total pixel frames of the assembled video.

    ``T = tile + (N-1) * (tile - overlap)``
    """
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    return tile_size_pixel + (n_segments - 1) * (tile_size_pixel - overlap_pixel)


def keyframes_for_tile(
    routes: list[KeyframeRoute], temporal_tile_index: int
) -> list[KeyframeRoute]:
    """Filter routes to those belonging to ``temporal_tile_index``."""
    return [r for r in routes if r.temporal_tile_index == temporal_tile_index]
