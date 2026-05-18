"""Builders for ``VideoConditionByKeyframeIndex`` sets used by long-video
looping.

Three kinds of conditions are produced:

* **Overlap conditioning** — previous segment's tail latent injected at
  ``frame_idx = 0`` of the new segment with ``strength = overlap_strength``.
  Implements ComfyUI's ``temporal_overlap_cond_strength`` ⇔ latent half-noise
  transition.
* **Negative-index conditioning** — long-term memory anchor latent injected
  with a **negative** ``frame_idx`` so the model sees it as a "virtual past"
  via RoPE position encoding. Direct port of ComfyUI's
  ``optional_negative_index_latents``. ltx-core supports this natively because
  ``VideoConditionByKeyframeIndex.apply_to`` does
  ``positions[:, 0, ...] += self.frame_idx`` without any sign check.
* **Keyframe conditioning** — user-supplied image latents at arbitrary
  positive ``frame_idx``.

All three reduce to a list of ``VideoConditionByKeyframeIndex`` instances —
ltx-pipelines' ``DiffusionStage`` consumes them via ``ModalitySpec.conditionings``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ltx_core.conditioning import VideoConditionByKeyframeIndex
from ltx_core.types import VideoLatentShape


@dataclass(frozen=True)
class OverlapSpec:
    """Spec for injecting the previous segment's tail latent at ``frame_idx=0``."""

    tail_latent: torch.Tensor  # [B, C, overlap_T_latent, H_latent, W_latent]
    strength: float            # ComfyUI's temporal_overlap_cond_strength


@dataclass(frozen=True)
class NegativeIndexSpec:
    """Spec for long-term memory anchor injected at a *negative* frame index."""

    anchor_latent: torch.Tensor  # [B, C, anchor_T_latent, H_latent, W_latent]
    negative_frame_idx: int      # < 0 in pixel-frame units
    strength: float
    num_pixel_frames: int = 1    # how many pixel frames the anchor encodes


@dataclass(frozen=True)
class KeyframeSpec:
    """Spec for a user-supplied keyframe (pre-encoded to latent)."""

    keyframe_latent: torch.Tensor  # [B, C, 1, H_latent, W_latent]
    frame_idx: int                 # >= 0 pixel-frame units
    strength: float
    num_pixel_frames: int = 1


def build_video_conditionings(
    *,
    overlap: OverlapSpec | None = None,
    negative_index: NegativeIndexSpec | None = None,
    keyframes: list[KeyframeSpec] | None = None,
) -> list[VideoConditionByKeyframeIndex]:
    """Materialise all three condition kinds into a flat list ready for
    ``ModalitySpec(conditionings=...)``.

    Order matters only mildly: ltx-core appends each item's tokens to the
    latent state in sequence, but the underlying transformer attention is
    permutation-invariant within "extra tokens". We emit overlap first
    (semantically the most important for visual continuity), then negative-
    index, then keyframes.
    """
    conds: list[VideoConditionByKeyframeIndex] = []

    if overlap is not None:
        # The overlap latent acts as a clean prefix for frame_idx = 0.
        # ``num_pixel_frames`` is set to a token count > 1 so RoPE positions
        # span the full overlap range instead of being collapsed to a single
        # frame (the ``== 1`` branch of VideoConditionByKeyframeIndex).
        overlap_T_latent = overlap.tail_latent.shape[2]
        # 1 latent frame = 1 + scale_factors.time*(T-1) pixel frames for T>1,
        # but for conditioning purposes the only thing that matters is
        # "span more than one pixel frame" so the collapse branch is skipped.
        # We pass 8*T_latent as a faithful proxy.
        num_pixel = max(8 * overlap_T_latent, 2)
        conds.append(
            VideoConditionByKeyframeIndex(
                keyframes=overlap.tail_latent,
                frame_idx=0,
                strength=overlap.strength,
                num_pixel_frames=num_pixel,
            )
        )

    if negative_index is not None:
        if negative_index.negative_frame_idx >= 0:
            raise ValueError(
                "negative_index.negative_frame_idx must be negative; "
                f"got {negative_index.negative_frame_idx}"
            )
        conds.append(
            VideoConditionByKeyframeIndex(
                keyframes=negative_index.anchor_latent,
                frame_idx=negative_index.negative_frame_idx,
                strength=negative_index.strength,
                num_pixel_frames=negative_index.num_pixel_frames,
            )
        )

    if keyframes:
        for kf in keyframes:
            conds.append(
                VideoConditionByKeyframeIndex(
                    keyframes=kf.keyframe_latent,
                    frame_idx=kf.frame_idx,
                    strength=kf.strength,
                    num_pixel_frames=kf.num_pixel_frames,
                )
            )

    return conds


def slice_overlap_tail(
    video_latent: torch.Tensor, overlap_latent_frames: int
) -> torch.Tensor:
    """Return the last ``overlap_latent_frames`` along the temporal axis.

    Operates on 5-D ``[B, C, T, H, W]``. Detached and cloned so callers can
    free the source tensor.
    """
    if overlap_latent_frames <= 0:
        raise ValueError("overlap_latent_frames must be >= 1")
    if overlap_latent_frames > video_latent.shape[2]:
        raise ValueError(
            f"overlap_latent_frames ({overlap_latent_frames}) > T "
            f"({video_latent.shape[2]})"
        )
    return video_latent[:, :, -overlap_latent_frames:, :, :].detach().clone()


def slice_anchor(
    video_latent: torch.Tensor,
    anchor_latent_frames: int,
    from_start: bool = True,
) -> torch.Tensor:
    """Slice an anchor for long-term memory.

    By default we take frames from the start of the first segment (the
    "establishing shot"), matching ComfyUI's typical usage where the user
    supplies a hand-picked anchor latent. Pass ``from_start=False`` to take
    from the end instead.
    """
    if anchor_latent_frames <= 0:
        raise ValueError("anchor_latent_frames must be >= 1")
    if anchor_latent_frames > video_latent.shape[2]:
        anchor_latent_frames = video_latent.shape[2]
    if from_start:
        return video_latent[:, :, :anchor_latent_frames, :, :].detach().clone()
    return video_latent[:, :, -anchor_latent_frames:, :, :].detach().clone()


def expected_latent_shape(
    height: int,
    width: int,
    num_frames: int,
    latent_channels: int = 128,
) -> VideoLatentShape:
    """Return the ltx-core ``VideoLatentShape`` for a target pixel grid.

    Useful for runtime sanity checks before slicing tails / anchors.
    """
    return VideoLatentShape(
        batch=1,
        channels=latent_channels,
        frames=(num_frames - 1) // 8 + 1,
        height=height // 32,
        width=width // 32,
    )
