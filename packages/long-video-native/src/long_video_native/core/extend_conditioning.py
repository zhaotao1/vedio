"""Clean-prefix conditioning for the Extend-style sampler (P4).

ComfyUI's ``LTXVExtendSampler`` reuses the last ``frame_overlap`` pixel
frames of the previous tile as a *clean prefix*: the underlying sampler
runs over the full ``tile_size`` frames but the prefix tokens are pinned
to their reference values (``denoise_mask = 0``) so only the new
``tile_size - frame_overlap`` frames are denoised.

ltx-core supports this natively via :class:`VideoConditionByLatentIndex`,
which:

* replaces the token grid at a given ``latent_idx`` with the supplied
  latent;
* sets the denoise mask at those positions to ``1.0 - strength`` —
  ``strength=1.0`` therefore gives a fully clean prefix (mask=0).

This module materialises an :class:`ExtendPrefixSpec` from the previous
tile's tail latent and returns the corresponding conditioning item ready
to be appended to ``ModalitySpec.conditionings``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ltx_core.conditioning import VideoConditionByLatentIndex


@dataclass(frozen=True)
class ExtendPrefixSpec:
    """Spec for the clean-prefix portion of an extension tile.

    Attributes:
        prefix_latent: ``[B, C, T_prefix_latent, H_latent, W_latent]`` — the
            last ``overlap_latent_frames`` of the previous tile's latent,
            already at the *current* tile's spatial resolution.
        strength: how strongly the prefix is pinned. ``1.0`` = fully
            clean (mask = 0); ``0.5`` = half-noised; ``0.0`` = no
            constraint (equivalent to no prefix). ComfyUI uses
            ``temporal_overlap_cond_strength`` here; ``0.5`` default.
        latent_idx: latent-frame index at which to inject the prefix.
            For the standard "extend at the start of the new tile" case
            this is ``0`` — the prefix replaces the first
            ``T_prefix_latent`` frames of the new tile's latent grid.
    """

    prefix_latent: torch.Tensor
    strength: float
    latent_idx: int = 0


def build_extend_conditioning(spec: ExtendPrefixSpec | None):
    """Materialise a :class:`VideoConditionByLatentIndex` for the prefix.

    Returns an empty list when ``spec`` is ``None`` so callers can simply
    do ``conditionings = stage_image_conds + build_extend_conditioning(spec)``.
    """
    if spec is None:
        return []
    if spec.prefix_latent.dim() != 5:
        raise ValueError(
            f"prefix_latent must be 5-D [B, C, T, H, W]; got "
            f"{tuple(spec.prefix_latent.shape)}"
        )
    if not (0.0 <= spec.strength <= 1.0):
        raise ValueError(f"strength must be in [0, 1]; got {spec.strength}")
    if spec.latent_idx < 0:
        raise ValueError(f"latent_idx must be >= 0; got {spec.latent_idx}")
    return [
        VideoConditionByLatentIndex(
            latent=spec.prefix_latent,
            strength=spec.strength,
            latent_idx=spec.latent_idx,
        )
    ]
