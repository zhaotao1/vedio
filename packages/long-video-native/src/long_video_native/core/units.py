"""Pixel ↔ latent frame unit conversion for LTX-2.

ComfyUI ``LTXVLoopingSampler`` exposes all temporal sizes in *pixel frames*
(``temporal_tile_size``, ``temporal_overlap``, ``cond_image_indices``).
LTX-2's VAE has a temporal compression factor of 8: ``T_pixel = (T_latent - 1)
* 8 + 1`` for ``T_latent >= 1``.

This module centralises the conversion so the new ComfyUI-aligned config
fields and the legacy ``*_latent_frames`` ones interoperate cleanly.
"""

from __future__ import annotations

VAE_TEMPORAL_SCALE = 8
"""LTX-2 VAE temporal compression factor (1 latent frame ↔ 8 pixel frames)."""


def pixel_to_latent_frames(n_pixel: int) -> int:
    """Convert a *count* of pixel frames to latent frames.

    Mirrors ``(num_frames - 1) // 8 + 1`` from ltx-core/ltx-pipelines.
    Valid inputs satisfy ``(n_pixel - 1) % 8 == 0`` but we accept any
    positive int (rounding down to the nearest 8K+1).
    """
    if n_pixel < 1:
        raise ValueError(f"n_pixel must be >= 1, got {n_pixel}")
    return (n_pixel - 1) // VAE_TEMPORAL_SCALE + 1


def latent_to_pixel_frames(n_latent: int) -> int:
    """Convert a *count* of latent frames back to pixel frames."""
    if n_latent < 1:
        raise ValueError(f"n_latent must be >= 1, got {n_latent}")
    return (n_latent - 1) * VAE_TEMPORAL_SCALE + 1


def snap_pixel_frames(n_pixel: int) -> int:
    """Round *n_pixel* down to the nearest valid LTX-2 video length (8K+1)."""
    if n_pixel < 1:
        return 1
    return ((n_pixel - 1) // VAE_TEMPORAL_SCALE) * VAE_TEMPORAL_SCALE + 1


def snap_pixel_overlap(n_pixel: int) -> int:
    """Round *n_pixel* down to the nearest multiple of 8.

    ComfyUI ``temporal_overlap`` is in pixel frames with step 8.
    """
    if n_pixel < 0:
        raise ValueError(f"overlap must be >= 0, got {n_pixel}")
    return (n_pixel // VAE_TEMPORAL_SCALE) * VAE_TEMPORAL_SCALE


def snap_pixel_keyframe_idx(idx: int) -> int:
    """Round *idx* down to the nearest ``1 mod 8`` (ComfyUI keyframe rule).

    Negative indices are passed through unchanged (RoPE-only conditioning).
    """
    if idx < 0:
        return idx
    return ((idx - 1) // VAE_TEMPORAL_SCALE) * VAE_TEMPORAL_SCALE + 1 if idx >= 1 else 0
