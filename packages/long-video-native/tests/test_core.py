"""Unit tests for core utilities — no model, no GPU required."""

from __future__ import annotations

import pytest
import torch

from long_video_native.core.adain import adain_match
from long_video_native.core.blending import (
    plan_tile_grid,
    spatial_tile_weights,
    temporal_overlap_blend,
)
from long_video_native.core.conditioning_builder import (
    NegativeIndexSpec,
    OverlapSpec,
    build_video_conditionings,
    slice_anchor,
    slice_overlap_tail,
)
from long_video_native.core.state import TileMemory


# ---------------------------------------------------------------- adain

def test_adain_match_preserves_shape():
    target = torch.randn(1, 128, 4, 22, 38)
    ref = torch.randn(1, 128, 4, 22, 38) * 3.0 + 1.0
    out = adain_match(target, ref, factor=1.0)
    assert out.shape == target.shape
    # per-channel mean/std should now match the reference
    mean_diff = (out.mean(dim=(2, 3, 4)) - ref.mean(dim=(2, 3, 4))).abs().max()
    assert mean_diff < 1e-4


def test_adain_factor_zero_is_identity():
    target = torch.randn(1, 8, 2, 4, 4)
    ref = torch.randn(1, 8, 2, 4, 4)
    out = adain_match(target, ref, factor=0.0)
    torch.testing.assert_close(out, target)


# ------------------------------------------------------------ blending

def test_temporal_overlap_blend_length():
    a = torch.randn(1, 4, 10, 8, 8)
    b = torch.randn(1, 4, 10, 8, 8)
    out = temporal_overlap_blend(a, b, overlap=3)
    assert out.shape[2] == 10 + 10 - 3


def test_temporal_overlap_zero_is_concat():
    a = torch.randn(1, 4, 5, 4, 4)
    b = torch.randn(1, 4, 5, 4, 4)
    out = temporal_overlap_blend(a, b, overlap=0)
    torch.testing.assert_close(out, torch.cat([a, b], dim=2))


def test_plan_tile_grid_single_tile():
    tiles = plan_tile_grid(total_px=704, tile_px=704, overlap_px=128)
    assert tiles == [(0, 704)]


def test_plan_tile_grid_multi_tile():
    tiles = plan_tile_grid(total_px=1408, tile_px=704, overlap_px=128)
    assert len(tiles) >= 2
    # covers the full range
    assert tiles[0][0] == 0
    assert tiles[-1][1] == 1408


def test_spatial_tile_weights_partition_of_unity():
    """When tiles are accumulated they should sum to ~1 in the interior."""
    h, w = 22, 38
    overlap = 4
    n_v, n_h = 2, 2
    accumulator = torch.zeros(1, 1, 1, h * 2 - overlap, w * 2 - overlap)
    for v in range(n_v):
        for h_i in range(n_h):
            shape = (1, 1, 1, h, w)
            wts = spatial_tile_weights(
                shape, v_idx=v, h_idx=h_i,
                vertical_tiles=n_v, horizontal_tiles=n_h,
                spatial_overlap=overlap,
                device=torch.device("cpu"), dtype=torch.float32,
            )
            v_off = v * (h - overlap)
            h_off = h_i * (w - overlap)
            accumulator[:, :, :, v_off:v_off + h, h_off:h_off + w] += wts
    # interior cells should sum to ~1; corners may differ slightly
    interior = accumulator[:, :, :, overlap:-overlap, overlap:-overlap]
    assert (interior - 1.0).abs().max() < 1e-3


# ---------------------------------------------------- conditioning_builder

def test_build_conditionings_overlap_only():
    tail = torch.randn(1, 128, 3, 22, 38)
    out = build_video_conditionings(
        overlap=OverlapSpec(tail_latent=tail, strength=0.5)
    )
    assert len(out) == 1
    assert out[0].frame_idx == 0
    assert out[0].strength == 0.5


def test_build_conditionings_negative_index():
    tail = torch.randn(1, 128, 3, 22, 38)
    anchor = torch.randn(1, 128, 2, 22, 38)
    out = build_video_conditionings(
        overlap=OverlapSpec(tail_latent=tail, strength=0.5),
        negative_index=NegativeIndexSpec(
            anchor_latent=anchor, negative_frame_idx=-16, strength=0.3
        ),
    )
    assert len(out) == 2
    assert out[1].frame_idx == -16


def test_negative_frame_idx_must_be_negative():
    anchor = torch.randn(1, 128, 2, 22, 38)
    with pytest.raises(ValueError, match="negative"):
        build_video_conditionings(
            negative_index=NegativeIndexSpec(
                anchor_latent=anchor, negative_frame_idx=4, strength=0.3
            )
        )


def test_slice_overlap_tail():
    v = torch.arange(16).view(1, 1, 16, 1, 1).float()
    tail = slice_overlap_tail(v, 3)
    assert tail.shape[2] == 3
    torch.testing.assert_close(tail.flatten(), torch.tensor([13.0, 14.0, 15.0]))


def test_slice_anchor_from_start():
    v = torch.arange(16).view(1, 1, 16, 1, 1).float()
    a = slice_anchor(v, 2, from_start=True)
    torch.testing.assert_close(a.flatten(), torch.tensor([0.0, 1.0]))


# ----------------------------------------------------------------- state

def test_tile_memory_roundtrip():
    mem = TileMemory(vertical_tiles=2, horizontal_tiles=3)
    t = torch.randn(1, 4, 3, 8, 8)
    mem.set_tail(1, 2, t)
    torch.testing.assert_close(mem.get_tail(1, 2), t)
    assert mem.get_tail(0, 0) is None
