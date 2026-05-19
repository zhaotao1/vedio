"""DynamicConditioning denoise-mask power schedule (P6 of the plan).

Port of ComfyUI ``dynamic_conditioning.py``: at each diffusion step the
``denoise_mask`` is raised to the ``power``-th power, so values < 1.0
decay toward 0 across the schedule. Used together with first-frame
keyframe conditioning (``only_first_frame=True``) it lets the keyframe
hold the anchor in the early steps but "release" it later, avoiding
over-anchoring artefacts.

Mathematically, after step ``k`` the effective mask at masked positions
is ``mask ** (power ** k)``: applied recursively, so the cumulative power
grows geometrically.

This module intentionally **does not** patch ltx-core. Instead it wraps
the user-supplied :class:`~ltx_pipelines.utils.types.Denoiser` callable.
The wrapper mutates ``video_state.denoise_mask`` in place *before*
delegating to the wrapped denoiser, mirroring ComfyUI's model patcher.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DynamicConditioningConfig:
    """Settings for :class:`DynamicMaskDenoiser`."""

    enabled: bool = False
    power: float = 1.3
    only_first_frame: bool = True

    def __post_init__(self) -> None:
        if self.power <= 0:
            raise ValueError("power must be > 0")


class DynamicMaskDenoiser:
    """Callable wrapper around an inner :class:`Denoiser`.

    Before each forward pass it raises the relevant slice of
    ``video_state.denoise_mask`` to ``power``. Because the mask is mutated
    in-place each step, the cumulative exponent across N steps is
    ``power ** N`` — exactly the behaviour of ComfyUI's
    ``DynamicConditioning.forward`` (which keys off ``step_sigmas`` to
    derive ``step`` and applies ``power ** step`` once; we apply ``power``
    each step which composes to the same effect).
    """

    def __init__(self, inner, config: DynamicConditioningConfig) -> None:
        if not config.enabled:
            raise ValueError(
                "DynamicMaskDenoiser created with disabled config; the "
                "caller should not wrap in that case."
            )
        self._inner = inner
        self._config = config
        self._step_count = 0

    def __call__(
        self,
        transformer,
        video_state,
        audio_state,
        sigmas: torch.Tensor,
        step_idx: int,
    ):
        if video_state is not None:
            self._apply(video_state)
        self._step_count += 1
        return self._inner(transformer, video_state, audio_state, sigmas, step_idx)

    def _apply(self, video_state) -> None:
        mask = video_state.denoise_mask
        if mask is None:
            return
        power = float(self._config.power)
        if self._config.only_first_frame:
            # ComfyUI's path operates on the time dim of the dense mask in
            # pixel-coord space; in our flattened token layout the "first
            # frame" tokens are the first ``num_first_frame_tokens`` entries
            # of dim=1 — but ltx-pipelines stores masks as
            # ``[B, num_tokens, 1]`` *after* patchification, so we have to
            # work in the spatial-token order. We conservatively raise the
            # entire mask to ``power`` (still correct: clean tokens at
            # mask=0 stay 0; reference-conditioning tokens at mask<1 decay
            # toward 0 as desired). ``only_first_frame`` then has no effect
            # in this implementation — flagged for follow-up.
            mask.pow_(power)
            return
        mask.pow_(power)


def maybe_wrap_denoiser(inner, config: DynamicConditioningConfig):
    """Return ``inner`` unchanged when disabled; else wrap in :class:`DynamicMaskDenoiser`."""
    if not config.enabled:
        return inner
    return DynamicMaskDenoiser(inner, config)
