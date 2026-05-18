"""Temporal looping pipeline built on top of ltx-pipelines' ``DistilledPipeline``.

This is the MVP that replicates ComfyUI ``LTXVLoopingSampler``'s temporal
behaviour:

* generate per-segment latents through the distilled 2-stage path;
* stitch consecutive segments by injecting the previous segment's tail
  latent as a ``VideoConditionByKeyframeIndex`` at ``frame_idx = 0`` with
  ``strength = overlap_strength`` (the "latent half-noise transition");
* optionally inject a long-term **negative-index** anchor latent for global
  consistency across many segments;
* apply latent-space **AdaIN** to suppress colour drift;
* assemble the full latent timeline with overlap blending and decode once.

The transformer is built once via ``DiffusionStage.model_context()`` and
reused across all segments, so the per-segment overhead is just one stage-1
+ one stage-2 denoising loop.

Spatial tiling is handled by :mod:`long_video_native.pipeline.spatial_tiled`
which wraps this class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.video_vae import TilingConfig
from ltx_core.types import Audio
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
)
from ltx_pipelines.utils.types import ModalitySpec

from long_video_native.core.adain import adain_match
from long_video_native.core.blending import temporal_overlap_blend
from long_video_native.core.conditioning_builder import (
    NegativeIndexSpec,
    OverlapSpec,
    build_video_conditionings,
    slice_anchor,
    slice_overlap_tail,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopingConfig:
    """Configuration for ``LoopingPipeline.generate_long``.

    All "latent frame" units are in VAE temporal space (8 pixel frames per
    latent frame). ``chunk_num_frames`` is in pixel frames and must satisfy
    ``(chunk_num_frames - 1) % 8 == 0``.
    """

    # Per-segment generation
    chunk_num_frames: int = 121          # pixel frames per chunk; must be 8K+1
    overlap_latent_frames: int = 3       # latent-frame overlap between chunks
    overlap_strength: float = 0.5        # = ComfyUI temporal_overlap_cond_strength

    # Long-term memory (negative-index latent)
    enable_negative_index: bool = False
    negative_index_anchor_latent_frames: int = 2  # how many latent frames to keep as anchor
    negative_index_strength: float = 0.3
    negative_frame_offset: int = -16     # negative pixel-frame offset for RoPE

    # Colour drift correction
    adain_factor: float = 0.5            # 0 disables; 1 fully matches first segment

    # Stage sigmas (override only for advanced experimentation)
    stage_1_sigmas: torch.Tensor = field(default_factory=lambda: DISTILLED_SIGMAS)
    stage_2_sigmas: torch.Tensor = field(default_factory=lambda: STAGE_2_DISTILLED_SIGMAS)

    def __post_init__(self) -> None:
        if (self.chunk_num_frames - 1) % 8 != 0:
            raise ValueError(
                f"chunk_num_frames must be 8K+1, got {self.chunk_num_frames}"
            )
        if self.overlap_latent_frames < 0:
            raise ValueError("overlap_latent_frames must be >= 0")
        if not 0.0 <= self.overlap_strength <= 1.0:
            raise ValueError("overlap_strength must be in [0, 1]")
        if not 0.0 <= self.adain_factor <= 1.0:
            raise ValueError("adain_factor must be in [0, 1]")
        if self.enable_negative_index and self.negative_frame_offset >= 0:
            raise ValueError("negative_frame_offset must be < 0")


class LoopingPipeline(DistilledPipeline):
    """Multi-segment temporal looping wrapper around the distilled pipeline.

    Public entry points:

    * :meth:`generate_long` — top-level: generate N segments, stitch, decode.
    * :meth:`generate_one_segment` — internal helper for the spatial tiling
      pipeline; produces a single segment's hi-res latent given an optional
      previous-tail and anchor latent. Does NOT decode.
    """

    # ---------------------------------------------------------------- public

    # NOTE: we deliberately use ``@torch.no_grad()`` rather than
    # ``@torch.inference_mode()`` here because ``self.video_decoder(...)``
    # returns a *lazy* iterator that is consumed later by ``encode_video``
    # in the runner — i.e. after this function (and therefore the
    # inference-mode context) has already exited. Inference-mode tensors
    # produced inside this scope would then be illegal to feed into
    # autograd-enabled ops (the VAE decoder weights have
    # ``requires_grad=True``), triggering:
    #   RuntimeError: Inference tensors cannot be saved for backward.
    # ``torch.no_grad()`` gives us the same gradient suppression without
    # marking tensors as inference-only, so deferred decoding works.
    @torch.no_grad()
    def generate_long(
        self,
        prompts: list[str],
        *,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        keyframes_per_segment: list[list] | None = None,
        config: LoopingConfig | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[torch.Tensor, Audio]:
        """Generate a long video from a list of per-segment prompts.

        Args:
            prompts: one prompt per segment. Length determines the number of
                segments; total pixel frames =
                ``len(prompts) * (chunk_num_frames - overlap_pixel_frames)
                + overlap_pixel_frames``.
            seed: master seed; each segment derives ``seed + segment_idx``.
            height, width: full output resolution (pixel). Must satisfy
                two-stage divisibility (height/width % 64 == 0).
            frame_rate: target fps.
            keyframes_per_segment: optional list aligned with ``prompts``;
                each element is a list of
                ``ImageConditioningInput``-compatible tuples that ltx-pipelines
                accepts (see :func:`combined_image_conditionings`).
            config: :class:`LoopingConfig` instance (defaults applied if None).
            tiling_config: forwarded to VAE decode.
            enhance_prompt: pass-through to ltx-pipelines prompt encoder.

        Returns:
            ``(decoded_video_chunks_iterator, decoded_audio)`` — same shape as
            ``DistilledPipeline.__call__`` to be drop-in compatible with the
            existing ``encode_video`` writer.
        """
        if not prompts:
            raise ValueError("prompts must be non-empty")
        config = config or LoopingConfig()
        assert_resolution(height=height, width=width, is_two_stage=True)
        keyframes_per_segment = keyframes_per_segment or [[] for _ in prompts]
        if len(keyframes_per_segment) != len(prompts):
            raise ValueError(
                "keyframes_per_segment length must match prompts length"
            )

        # Build the transformer ONCE for both stages — every segment reuses
        # the same context to amortise the load cost.
        # NOTE: distilled pipeline calls the same stage for both stage-1 and
        # stage-2; we follow that pattern and build the context per stage.
        all_video_latents: list[torch.Tensor] = []
        all_audio_latents: list[torch.Tensor] = []

        # Anchor latent for negative-index memory is captured after segment 0.
        anchor_latent: torch.Tensor | None = None

        for seg_idx, prompt in enumerate(prompts):
            prev_tail = (
                all_video_latents[-1][:, :, -config.overlap_latent_frames:, :, :]
                .detach()
                .clone()
                if seg_idx > 0 and config.overlap_latent_frames > 0
                else None
            )
            logger.info(
                "[long-video] segment %d/%d prompt=%r",
                seg_idx + 1, len(prompts), prompt[:60],
            )

            video_latent, audio_latent = self.generate_one_segment(
                prompt=prompt,
                seed=seed + seg_idx,
                height=height,
                width=width,
                num_frames=config.chunk_num_frames,
                frame_rate=frame_rate,
                images=keyframes_per_segment[seg_idx],
                prev_tail_latent=prev_tail,
                anchor_latent=(
                    anchor_latent if config.enable_negative_index else None
                ),
                config=config,
                enhance_prompt=enhance_prompt and seg_idx == 0,  # only first
            )

            # AdaIN against the first segment to halt colour drift.
            if seg_idx > 0 and config.adain_factor > 0:
                video_latent = adain_match(
                    video_latent, all_video_latents[0], factor=config.adain_factor
                )

            all_video_latents.append(video_latent)
            all_audio_latents.append(audio_latent)

            # Snapshot the anchor from segment 0 (start of the establishing shot).
            if seg_idx == 0 and config.enable_negative_index:
                anchor_latent = slice_anchor(
                    video_latent,
                    config.negative_index_anchor_latent_frames,
                    from_start=True,
                )

        # --- assemble full latent timeline with overlap blending -----------
        full_video_latent = all_video_latents[0]
        for nxt in all_video_latents[1:]:
            full_video_latent = temporal_overlap_blend(
                full_video_latent, nxt, overlap=config.overlap_latent_frames
            )
        # Audio: simple concat along the temporal frame axis. LTX-2 audio
        # latents have shape (batch, channels, frames, mel_bins); we
        # concatenate on the frame dim (-2), NOT mel_bins (-1), otherwise
        # downstream per-channel statistics buffers shape-mismatch.
        # No cross-segment overlap mechanism exists in ltx-core for audio;
        # if continuity artefacts appear, callers can post-process externally.
        full_audio_latent = torch.cat(all_audio_latents, dim=-2)

        # --- decode --------------------------------------------------------
        logger.info(
            "[long-video] assembled latent shape video=%s audio=%s; decoding...",
            tuple(full_video_latent.shape), tuple(full_audio_latent.shape),
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        decoded_video = self.video_decoder(
            full_video_latent, tiling_config, generator
        )
        decoded_audio = self.audio_decoder(full_audio_latent)
        return decoded_video, decoded_audio

    # ------------------------------------------------------ single-segment

    # See note on ``generate_long``: ``no_grad`` instead of
    # ``inference_mode`` so the returned latents can be safely fed into
    # the lazy video-decoder iterator after this function returns.
    @torch.no_grad()
    def generate_one_segment(
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list,
        prev_tail_latent: torch.Tensor | None,
        anchor_latent: torch.Tensor | None,
        config: LoopingConfig,
        enhance_prompt: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the distilled 2-stage pipeline for a single segment.

        Returns ``(video_latent, audio_latent)`` — unpatchified 5-D / 3-D
        tensors, NOT decoded.

        This method mirrors ``DistilledPipeline.__call__`` (distilled.py
        L87) but injects extra conditionings for the overlap and the
        negative-index anchor. It is the unit of work consumed by the
        spatial-tiling wrapper.
        """
        dtype = torch.bfloat16
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        # ----------------- Stage 1: low-res generation ---------------------
        stage_1_sigmas = config.stage_1_sigmas.to(
            dtype=torch.float32, device=self.device
        )
        stage_1_h, stage_1_w = height // 2, width // 2

        # Image-keyframe conditionings (low-res spatial)
        stage_1_image_conds = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_h,
                width=stage_1_w,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        # Extra continuity conditionings (overlap + negative-index)
        # These are also low-res spatial — the caller must hand us latents at
        # the right scale. For stage 1 we downsample if needed.
        extra_conds_s1 = self._build_continuity_conditionings(
            prev_tail_latent=_maybe_resize_latent(
                prev_tail_latent, target_h=stage_1_h // 32, target_w=stage_1_w // 32
            ),
            anchor_latent=_maybe_resize_latent(
                anchor_latent, target_h=stage_1_h // 32, target_w=stage_1_w // 32
            ),
            config=config,
        )

        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_w,
            height=stage_1_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_1_image_conds + extra_conds_s1,
            ),
            audio=ModalitySpec(context=audio_context),
        )

        # ----------------- Stage 2: upsample + refine ----------------------
        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        stage_2_sigmas = config.stage_2_sigmas.to(
            dtype=torch.float32, device=self.device
        )

        stage_2_image_conds = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        extra_conds_s2 = self._build_continuity_conditionings(
            prev_tail_latent=prev_tail_latent,  # already hi-res
            anchor_latent=anchor_latent,
            config=config,
        )

        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_image_conds + extra_conds_s2,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )

        return video_state.latent, audio_state.latent

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _build_continuity_conditionings(
        *,
        prev_tail_latent: torch.Tensor | None,
        anchor_latent: torch.Tensor | None,
        config: LoopingConfig,
    ) -> list:
        overlap: OverlapSpec | None = None
        if prev_tail_latent is not None and config.overlap_latent_frames > 0:
            overlap = OverlapSpec(
                tail_latent=prev_tail_latent,
                strength=config.overlap_strength,
            )
        negative: NegativeIndexSpec | None = None
        if anchor_latent is not None and config.enable_negative_index:
            negative = NegativeIndexSpec(
                anchor_latent=anchor_latent,
                negative_frame_idx=config.negative_frame_offset,
                strength=config.negative_index_strength,
                num_pixel_frames=max(8 * anchor_latent.shape[2], 2),
            )
        return build_video_conditionings(overlap=overlap, negative_index=negative)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _maybe_resize_latent(
    latent: torch.Tensor | None, target_h: int, target_w: int
) -> torch.Tensor | None:
    """Resize a latent's spatial dims to ``(target_h, target_w)`` if needed.

    Latents are bf16/float ``[B, C, T, H, W]``; ``F.interpolate`` over the
    last two dims with area mode preserves low-frequency content reasonably
    for conditioning purposes (the model only sees these tokens via attention,
    not as a target to reconstruct).
    """
    if latent is None:
        return None
    if latent.shape[-2] == target_h and latent.shape[-1] == target_w:
        return latent
    b, c, t, _, _ = latent.shape
    flat = latent.reshape(b * c, 1, t, latent.shape[-2], latent.shape[-1])
    resized = torch.nn.functional.interpolate(
        flat.float(),
        size=(t, target_h, target_w),
        mode="trilinear",
        align_corners=False,
    ).to(latent.dtype)
    return resized.reshape(b, c, t, target_h, target_w)
