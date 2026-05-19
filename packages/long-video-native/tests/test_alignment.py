"""Tests for the new ComfyUI-aligned helpers (units, seeding, keyframe_router,
prompt_cache, extend_conditioning, dynamic_conditioning, latent_io)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from long_video_native.core.dynamic_conditioning import (
    DynamicConditioningConfig,
    maybe_wrap_denoiser,
)
from long_video_native.core.extend_conditioning import (
    ExtendPrefixSpec,
    build_extend_conditioning,
)
from long_video_native.core.keyframe_router import (
    KeyframeRoute,
    keyframes_for_tile,
    route_keyframes,
    total_pixel_frames,
)
from long_video_native.core.latent_io import load_reference_latent
from long_video_native.core.seeding import calc_tile_seed, per_tile_offset_for
from long_video_native.core.units import (
    VAE_TEMPORAL_SCALE,
    latent_to_pixel_frames,
    pixel_to_latent_frames,
    snap_pixel_frames,
    snap_pixel_keyframe_idx,
    snap_pixel_overlap,
)


# --------------------------------------------------------------------- units


def test_pixel_latent_round_trip():
    # Valid LTX-2 video lengths satisfy (n - 1) % 8 == 0.
    for n_pix in (1, 9, 17, 121, 201):
        n_lat = pixel_to_latent_frames(n_pix)
        assert latent_to_pixel_frames(n_lat) == n_pix


def test_pixel_to_latent_rounds_down():
    # 122 pixel frames → ((122-1)//8)+1 = 16 latent frames
    assert pixel_to_latent_frames(122) == 16


def test_snap_pixel_frames():
    assert snap_pixel_frames(1) == 1
    assert snap_pixel_frames(9) == 9
    assert snap_pixel_frames(15) == 9
    assert snap_pixel_frames(17) == 17


def test_snap_pixel_overlap_step_8():
    assert snap_pixel_overlap(0) == 0
    assert snap_pixel_overlap(23) == 16
    assert snap_pixel_overlap(24) == 24
    with pytest.raises(ValueError):
        snap_pixel_overlap(-1)


def test_snap_pixel_keyframe_idx():
    assert snap_pixel_keyframe_idx(0) == 0
    assert snap_pixel_keyframe_idx(1) == 1
    assert snap_pixel_keyframe_idx(9) == 9
    # 8 falls between 1 and 9 → rounds down to 1.
    assert snap_pixel_keyframe_idx(8) == 1
    # negative indices pass through unchanged (RoPE long-memory anchors)
    assert snap_pixel_keyframe_idx(-16) == -16


def test_vae_temporal_scale_constant():
    assert VAE_TEMPORAL_SCALE == 8


# ------------------------------------------------------------------ seeding


def test_calc_tile_seed_matches_comfyui_formula():
    # base + start*(V*H) + v*H + h + offset
    seed = calc_tile_seed(
        base_seed=100,
        start_index=80,
        vertical_tiles=2,
        horizontal_tiles=3,
        v=1,
        h=2,
        per_tile_offset=5,
    )
    assert seed == 100 + 80 * (2 * 3) + 1 * 3 + 2 + 5


def test_per_tile_offset_for_repeats_last():
    assert per_tile_offset_for(None, 5) == 0
    assert per_tile_offset_for([], 5) == 0
    assert per_tile_offset_for([10, 20], 0) == 10
    assert per_tile_offset_for([10, 20], 1) == 20
    assert per_tile_offset_for([10, 20], 5) == 20


def test_calc_tile_seed_out_of_range_raises():
    with pytest.raises(ValueError):
        calc_tile_seed(
            base_seed=0,
            start_index=0,
            vertical_tiles=1,
            horizontal_tiles=1,
            v=1,
            h=0,
        )


# --------------------------------------------------------- keyframe router


def test_route_keyframes_first_tile():
    # tile_size=80, overlap=24, total=80 + 2*(80-24) = 192
    routes = route_keyframes(
        [0, 32, 50],
        tile_size_pixel=80,
        overlap_pixel=24,
        total_pixel_frames=192,
    )
    assert all(r.temporal_tile_index == 0 for r in routes)
    assert [r.in_tile_pixel_index for r in routes] == [0, 32, 50]


def test_route_keyframes_drops_out_of_range():
    routes = route_keyframes(
        [0, 1000],
        tile_size_pixel=80,
        overlap_pixel=24,
        total_pixel_frames=192,
    )
    assert len(routes) == 1
    assert routes[0].global_pixel_index == 0


def test_route_keyframes_subsequent_tile():
    # tile 1 starts at 1*(80-24) - 7 = 49, ends at 80 + 56 - 1 - 7 = 128
    # keyframe at global 120 → in_tile = 120 - 49 - 7 = 64
    routes = route_keyframes(
        [120],
        tile_size_pixel=80,
        overlap_pixel=24,
        total_pixel_frames=192,
    )
    assert len(routes) == 1
    assert routes[0].temporal_tile_index == 1
    assert routes[0].in_tile_pixel_index == 64


def test_route_keyframes_overlap_reattributed():
    # If a keyframe lands in tile-1's overlap region (in_tile < overlap=24),
    # it should be reassigned to tile 0.
    # tile 1 starts at pixel 49, overlap region is in_tile [0, 24) →
    # global pixel [49+7, 49+7+24) = [56, 80) → e.g. global 60 → in_tile=4
    # which is < 24, so it's pushed back to tile 0 with in_tile=60.
    routes = route_keyframes(
        [60],
        tile_size_pixel=80,
        overlap_pixel=24,
        total_pixel_frames=192,
    )
    assert len(routes) == 1
    assert routes[0].temporal_tile_index == 0
    assert routes[0].in_tile_pixel_index == 60


def test_keyframes_for_tile_filters():
    routes = [
        KeyframeRoute(0, 0, 0),
        KeyframeRoute(1, 10, 60),
        KeyframeRoute(1, 30, 80),
    ]
    assert len(keyframes_for_tile(routes, 0)) == 1
    assert len(keyframes_for_tile(routes, 1)) == 2
    assert keyframes_for_tile(routes, 2) == []


def test_total_pixel_frames_formula():
    # 14 segments × 121 frames, overlap 32 → 121 + 13*(121-32) = 1278
    assert total_pixel_frames(14, 121, 32) == 1278


# ---------------------------------------------------------- extend conditioning


def test_build_extend_conditioning_returns_empty_when_none():
    assert build_extend_conditioning(None) == []


def test_build_extend_conditioning_basic():
    prefix = torch.randn(1, 128, 4, 22, 38)
    spec = ExtendPrefixSpec(prefix_latent=prefix, strength=1.0, latent_idx=0)
    out = build_extend_conditioning(spec)
    assert len(out) == 1
    assert out[0].strength == 1.0
    assert out[0].latent_idx == 0
    assert out[0].latent.shape == prefix.shape


def test_build_extend_conditioning_validates_shape():
    bad = torch.randn(1, 128, 4, 22)  # only 4-D
    with pytest.raises(ValueError, match="5-D"):
        build_extend_conditioning(
            ExtendPrefixSpec(prefix_latent=bad, strength=1.0, latent_idx=0)
        )


def test_build_extend_conditioning_validates_strength():
    prefix = torch.randn(1, 128, 4, 22, 38)
    with pytest.raises(ValueError, match=r"strength"):
        build_extend_conditioning(
            ExtendPrefixSpec(prefix_latent=prefix, strength=1.5, latent_idx=0)
        )


# -------------------------------------------------------- dynamic conditioning


def test_dynamic_conditioning_disabled_returns_inner():
    sentinel = object()
    assert (
        maybe_wrap_denoiser(sentinel, DynamicConditioningConfig(enabled=False))
        is sentinel
    )


def test_dynamic_conditioning_invalid_power():
    with pytest.raises(ValueError):
        DynamicConditioningConfig(enabled=True, power=0.0)


def test_dynamic_conditioning_wraps_when_enabled():
    cfg = DynamicConditioningConfig(enabled=True, power=1.3)

    class _FakeState:
        def __init__(self):
            self.denoise_mask = torch.full((1, 4, 1), 0.5)

    calls: list[int] = []

    def inner(transformer, video_state, audio_state, sigmas, step_idx):
        calls.append(step_idx)
        return (None, None)

    state = _FakeState()
    wrapped = maybe_wrap_denoiser(inner, cfg)
    wrapped(None, state, None, torch.zeros(2), 0)
    wrapped(None, state, None, torch.zeros(2), 1)
    # Mask after two calls: 0.5 ** (1.3 ** 2) ≈ 0.5 ** 1.69
    expected = 0.5 ** (1.3 ** 2)
    assert torch.allclose(state.denoise_mask, torch.full_like(state.denoise_mask, expected), atol=1e-5)
    assert calls == [0, 1]


# ---------------------------------------------------------------- latent_io


def test_load_reference_latent_pt(tmp_path: Path):
    p = tmp_path / "ref.pt"
    t = torch.randn(1, 128, 4, 22, 38)
    torch.save(t, p)
    loaded = load_reference_latent(p)
    torch.testing.assert_close(loaded, t)


def test_load_reference_latent_pt_dict(tmp_path: Path):
    p = tmp_path / "ref.pt"
    t = torch.randn(1, 128, 4, 22, 38)
    torch.save({"latent": t}, p)
    loaded = load_reference_latent(p)
    torch.testing.assert_close(loaded, t)


def test_load_reference_latent_rejects_wrong_dim(tmp_path: Path):
    p = tmp_path / "ref.pt"
    torch.save(torch.randn(1, 128, 4), p)
    with pytest.raises(ValueError, match="5-D"):
        load_reference_latent(p)


def test_load_reference_latent_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_reference_latent(tmp_path / "does-not-exist.pt")


def test_load_reference_latent_unsupported_format(tmp_path: Path):
    p = tmp_path / "ref.bin"
    p.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="unsupported"):
        load_reference_latent(p)
