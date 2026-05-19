"""Per-tile prompt context cache for shared-guider operation (P3).

ComfyUI's ``MultiPromptProvider`` + ``_prepare_guider_for_chunk`` keeps
the same guider instance across temporal tiles and only swaps the
``positive`` conditioning. Our equivalent here:

* Encode the **negative prompt + all unique positive prompts** in a single
  ``prompt_encoder`` call.
* Cache the negative encoding (it's the same constant for all segments).
* Expose ``positive_for(tile_index)`` that picks the per-tile encoding,
  repeating the last entry if not enough prompts were supplied (matching
  ComfyUI's behaviour).

The cache is built once per ``generate_long`` call and shared across all
temporal + spatial tiles. This **does not** require the underlying guider
to be shared — but it does remove the per-segment cost of re-encoding
the same negative prompt, and (importantly) eliminates encoding-noise
drift between segments that share a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class _EncodedContext:
    video: torch.Tensor
    audio: torch.Tensor


class PromptContextCache:
    """Caches encoded (video, audio) contexts for negative + per-tile positive prompts."""

    def __init__(
        self,
        prompt_encoder,
        positive_prompts: Sequence[str],
        negative_prompt: str,
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image=None,
        enhance_prompt_seed: int = 0,
    ) -> None:
        if not positive_prompts:
            raise ValueError("positive_prompts must be non-empty")

        # Encode unique positives + the negative in one batched call so the
        # encoder only loads once and gradient state is shared.
        unique_positives: list[str] = []
        seen: dict[str, int] = {}
        for p in positive_prompts:
            if p not in seen:
                seen[p] = len(unique_positives)
                unique_positives.append(p)
        self._unique_to_idx = seen

        batch = [*unique_positives, negative_prompt]
        encoded = prompt_encoder(
            batch,
            enhance_first_prompt=enhance_first_prompt,
            enhance_prompt_image=enhance_prompt_image,
            enhance_prompt_seed=enhance_prompt_seed,
        )
        # ``prompt_encoder`` returns one ctx per input prompt (in order).
        contexts = list(encoded)
        if len(contexts) != len(batch):
            raise RuntimeError(
                f"prompt_encoder returned {len(contexts)} contexts, expected {len(batch)}"
            )
        self._unique_contexts: list[_EncodedContext] = [
            _EncodedContext(video=c.video_encoding, audio=c.audio_encoding)
            for c in contexts[:-1]
        ]
        neg = contexts[-1]
        self._neg = _EncodedContext(video=neg.video_encoding, audio=neg.audio_encoding)
        self._positive_prompts = list(positive_prompts)

    def positive_for(self, tile_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(video_context, audio_context)`` for a temporal tile.

        If the tile index exceeds the prompt list length, the LAST prompt
        is repeated (matches ComfyUI's MultiPromptProvider behaviour).
        """
        clamped = min(tile_index, len(self._positive_prompts) - 1)
        prompt = self._positive_prompts[clamped]
        ctx = self._unique_contexts[self._unique_to_idx[prompt]]
        return ctx.video, ctx.audio

    @property
    def negative(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._neg.video, self._neg.audio
