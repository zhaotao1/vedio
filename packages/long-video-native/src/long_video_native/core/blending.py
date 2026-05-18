"""Blend-weight builders for temporal stitching and 2-D spatial tiling.

Temporal: linear ramp over the overlap region between consecutive chunks.
Spatial: separable 1-D linear ramps on the four sides of each tile that
border another tile, multiplied together to form a 2-D blend mask.

Both mirror ComfyUI's ``_create_spatial_weights`` (looping_sampler.py L526)
but operate on latent tensors and stay shape-agnostic.
"""

from __future__ import annotations

import torch


def temporal_overlap_blend(
    prev: torch.Tensor,
    curr: torch.Tensor,
    overlap: int,
) -> torch.Tensor:
    """Blend the last ``overlap`` latent frames of ``prev`` with the first
    ``overlap`` latent frames of ``curr`` using a linear ramp.

    Both tensors are 5-D ``[B, C, T, H, W]`` and must share ``[B, C, H, W]``.

    Returns a single tensor of shape
    ``[B, C, prev.T + curr.T - overlap, H, W]``.
    """
    if overlap <= 0:
        return torch.cat([prev, curr], dim=2)
    if overlap > min(prev.shape[2], curr.shape[2]):
        raise ValueError(
            f"overlap ({overlap}) exceeds tensor length "
            f"(prev={prev.shape[2]}, curr={curr.shape[2]})"
        )

    ramp = torch.linspace(0.0, 1.0, overlap, device=prev.device, dtype=prev.dtype)
    ramp = ramp.view(1, 1, -1, 1, 1)

    prev_tail = prev[:, :, -overlap:, :, :]
    curr_head = curr[:, :, :overlap, :, :]
    blended = prev_tail * (1.0 - ramp) + curr_head * ramp

    return torch.cat(
        [prev[:, :, :-overlap, :, :], blended, curr[:, :, overlap:, :, :]],
        dim=2,
    )


def spatial_tile_weights(
    tile_shape: tuple[int, int, int, int, int],
    v_idx: int,
    h_idx: int,
    vertical_tiles: int,
    horizontal_tiles: int,
    spatial_overlap: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build per-tile 2-D blend weights.

    Direct port of ComfyUI ``LTXVLoopingSampler._create_spatial_weights``,
    operating in **latent** space. ``spatial_overlap`` is in latent units;
    callers convert from pixel units via ``// VIDEO_SCALE_FACTORS.height``.

    Returns a tensor of shape ``tile_shape`` with values in ``[0, 1]``.
    Centre stays at 1.0; sides that border another tile ramp from 0 → 1
    (or 1 → 0) over ``spatial_overlap``. Corners are the product of two
    ramps and therefore approach 0 quadratically.

    The accumulation pattern in the caller is:
        accumulator += tile_out * weights
        weight_sum  += weights
        final = accumulator / weight_sum.clamp_min(eps)
    """
    weights = torch.ones(tile_shape, device=device, dtype=dtype)
    if spatial_overlap <= 0:
        return weights

    # Horizontal (W axis)
    if h_idx > 0:
        h_blend = torch.linspace(0.0, 1.0, spatial_overlap, device=device, dtype=dtype)
        weights[:, :, :, :, :spatial_overlap] *= h_blend.view(1, 1, 1, 1, -1)
    if h_idx < horizontal_tiles - 1:
        h_blend = torch.linspace(1.0, 0.0, spatial_overlap, device=device, dtype=dtype)
        weights[:, :, :, :, -spatial_overlap:] *= h_blend.view(1, 1, 1, 1, -1)

    # Vertical (H axis)
    if v_idx > 0:
        v_blend = torch.linspace(0.0, 1.0, spatial_overlap, device=device, dtype=dtype)
        weights[:, :, :, :spatial_overlap, :] *= v_blend.view(1, 1, 1, -1, 1)
    if v_idx < vertical_tiles - 1:
        v_blend = torch.linspace(1.0, 0.0, spatial_overlap, device=device, dtype=dtype)
        weights[:, :, :, -spatial_overlap:, :] *= v_blend.view(1, 1, 1, -1, 1)

    return weights


def plan_tile_grid(
    full_size_px: int,
    tile_size_px: int,
    overlap_px: int,
) -> list[tuple[int, int]]:
    """Plan tile starts/ends in pixel units along one axis.

    Returns a list of ``(start, end)`` pairs (end exclusive). The last tile
    is clamped to ``full_size_px`` and may be shorter than ``tile_size_px``
    (which is acceptable for ltx-pipelines as long as 32-divisibility holds
    — the runner is responsible for choosing values that satisfy that).
    """
    if tile_size_px >= full_size_px:
        return [(0, full_size_px)]
    if overlap_px >= tile_size_px:
        raise ValueError(
            f"overlap_px ({overlap_px}) must be < tile_size_px ({tile_size_px})"
        )

    step = tile_size_px - overlap_px
    starts: list[int] = []
    cursor = 0
    while cursor + tile_size_px < full_size_px:
        starts.append(cursor)
        cursor += step
    starts.append(full_size_px - tile_size_px)  # always cover the right edge
    # Deduplicate while preserving order.
    seen: set[int] = set()
    unique_starts = [s for s in starts if not (s in seen or seen.add(s))]
    return [(s, s + tile_size_px) for s in unique_starts]
