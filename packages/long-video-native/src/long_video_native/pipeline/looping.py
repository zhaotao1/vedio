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

import gc
import logging
from collections.abc import Iterator
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

    # --- Streaming decode (Level-2 OOM mitigation) ------------------------
    # When enabled:
    #   * each segment's latent is moved to CPU immediately after generation
    #     so GPU memory is bounded by a single segment, not the whole timeline
    #   * the assembled full latent stays on CPU
    #   * VAE decode runs macro-chunk by macro-chunk: each chunk is shipped
    #     to GPU, decoded (using the inner tiling_config), then dropped
    #   * consecutive macro-chunks share ``streaming_overlap_latent_frames``
    #     latent frames which are linearly blended in pixel space to hide
    #     the seam (8 pixel frames per latent frame, VAE temporal scale)
    streaming_decode: bool = False
    streaming_chunk_latent_frames: int = 32   # latent frames per decode macro-chunk
    streaming_overlap_latent_frames: int = 4  # latent-frame overlap between macro-chunks

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
        if self.streaming_decode:
            if self.streaming_chunk_latent_frames <= self.streaming_overlap_latent_frames:
                raise ValueError(
                    "streaming_chunk_latent_frames must exceed streaming_overlap_latent_frames"
                )
            if self.streaming_overlap_latent_frames < 0:
                raise ValueError("streaming_overlap_latent_frames must be >= 0")


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
        #
        # Memory note: when ``config.streaming_decode`` is True we move each
        # finished segment latent to CPU immediately. This caps GPU residency
        # to a single segment instead of growing linearly with the timeline,
        # which is the key change that lets us scale past ~60s on 80GB.
        all_video_latents: list[torch.Tensor] = []
        all_audio_latents: list[torch.Tensor] = []

        # Anchor latent for negative-index memory is captured after segment 0.
        # Kept on GPU because it is re-fed into the transformer each segment.
        anchor_latent: torch.Tensor | None = None

        for seg_idx, prompt in enumerate(prompts):
            prev_tail = None
            if seg_idx > 0 and config.overlap_latent_frames > 0:
                # Tail latent must be on the active device for the next
                # transformer pass; ship it back from CPU only when needed.
                prev_full = all_video_latents[-1]
                prev_tail = (
                    prev_full[:, :, -config.overlap_latent_frames:, :, :]
                    .to(self.device, non_blocking=False)
                    .detach()
                    .clone()
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
            if prev_tail is not None:
                del prev_tail

            # AdaIN against the first segment to halt colour drift. Pull the
            # reference back to ``video_latent``'s device for the match; the
            # result inherits ``video_latent.device``.
            if seg_idx > 0 and config.adain_factor > 0:
                ref = all_video_latents[0].to(video_latent.device, non_blocking=False)
                video_latent = adain_match(video_latent, ref, factor=config.adain_factor)
                del ref

            # Snapshot the anchor from segment 0 BEFORE moving to CPU. Anchor
            # is small (a couple latent frames) and lives on GPU between
            # segments.
            if seg_idx == 0 and config.enable_negative_index:
                anchor_latent = slice_anchor(
                    video_latent,
                    config.negative_index_anchor_latent_frames,
                    from_start=True,
                )

            if config.streaming_decode:
                video_latent = video_latent.detach().to("cpu", non_blocking=False)
                audio_latent = audio_latent.detach().to("cpu", non_blocking=False)
                torch.cuda.empty_cache()
                gc.collect()

            all_video_latents.append(video_latent)
            all_audio_latents.append(audio_latent)

        # --- assemble full latent timeline with overlap blending -----------
        # Blending stays on whichever device the latents live on (CPU when
        # streaming, GPU otherwise). ``temporal_overlap_blend`` is
        # device-agnostic.
        full_video_latent = all_video_latents[0]
        for nxt in all_video_latents[1:]:
            full_video_latent = temporal_overlap_blend(
                full_video_latent, nxt, overlap=config.overlap_latent_frames
            )
        # Free the per-segment list once the assembled latent is built.
        del all_video_latents

        # Audio: simple concat along the temporal frame axis. LTX-2 audio
        # latents have shape (batch, channels, frames, mel_bins); we
        # concatenate on the frame dim (-2), NOT mel_bins (-1), otherwise
        # downstream per-channel statistics buffers shape-mismatch.
        # No cross-segment overlap mechanism exists in ltx-core for audio;
        # if continuity artefacts appear, callers can post-process externally.
        full_audio_latent = torch.cat(all_audio_latents, dim=-2)
        del all_audio_latents

        # --- decode --------------------------------------------------------
        # Critical for streaming: force a hard cleanup barrier before we
        # touch the VAE decoder. After 14 segments of stage-1 + stage-2 the
        # CUDA caching allocator can be holding tens of GB of *allocated*
        # blocks (not just reserved) — observed at 78 GB on 80 GB A100 even
        # with offload=cpu — which leaves the streaming decoder no room to
        # build its own weights + tile workspace. Drop unused tensors,
        # finalize all pending CUDA work, then drain the cache twice.
        if config.streaming_decode and anchor_latent is not None:
            # ``anchor_latent`` was kept on GPU between segments for the
            # negative-index path. Once generation is done it is dead weight.
            anchor_latent = None
        if full_video_latent.device.type == "cpu":
            # ``full_audio_latent`` is small (~5MB); ok to ship to GPU later.
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(
                "[long-video] post-generation cleanup: cuda mem allocated=%.2f GB reserved=%.2f GB",
                torch.cuda.memory_allocated() / 1024**3,
                torch.cuda.memory_reserved() / 1024**3,
            )

        logger.info(
            "[long-video] assembled latent shape video=%s audio=%s "
            "(device=%s); decoding (streaming=%s)...",
            tuple(full_video_latent.shape),
            tuple(full_audio_latent.shape),
            full_video_latent.device,
            config.streaming_decode,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Decode audio BEFORE entering the video-decode persistent context.
        # The audio path builds + frees its own decoder + vocoder via
        # ``gpu_model``; doing it first means by the time the video VAE
        # comes up, those weights are guaranteed to be on the ``meta``
        # device. We also pull the resulting audio tensor to CPU so it does
        # not occupy GPU memory during the long video decode.
        if full_audio_latent.device.type != self.device.type:
            full_audio_latent = full_audio_latent.to(self.device, non_blocking=False)
        decoded_audio = self.audio_decoder(full_audio_latent)
        del full_audio_latent
        if hasattr(decoded_audio, "data") and hasattr(decoded_audio.data, "device") and decoded_audio.data.device.type != "cpu":
            decoded_audio = decoded_audio._replace(data=decoded_audio.data.to("cpu")) if hasattr(decoded_audio, "_replace") else decoded_audio
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(
            "[long-video] post-audio-decode cleanup: cuda mem allocated=%.2f GB reserved=%.2f GB",
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )

        if config.streaming_decode:
            decoded_video = _streaming_video_decode(
                self.video_decoder,
                full_video_latent,
                tiling_config=tiling_config,
                chunk_latent_frames=config.streaming_chunk_latent_frames,
                overlap_latent_frames=config.streaming_overlap_latent_frames,
                device=self.device,
                generator=generator,
            )
        else:
            decoded_video = self.video_decoder(
                full_video_latent, tiling_config, generator
            )
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


# ---------------------------------------------------------------------------
# Streaming decode (Level-2 OOM mitigation)
# ---------------------------------------------------------------------------

# VAE temporal compression factor: 1 latent frame ↔ 8 pixel frames.
_VAE_TEMPORAL_SCALE = 8


def _streaming_video_decode(
    video_decoder,  # ltx_pipelines.utils.blocks.VideoDecoder (avoid hard import)
    full_latent_cpu: torch.Tensor,
    *,
    tiling_config: TilingConfig | None,
    chunk_latent_frames: int,
    overlap_latent_frames: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> Iterator[torch.Tensor]:
    """Stream-decode a CPU-resident full latent in temporal macro-chunks.

    The VAE decoder is built **once** and reused across all chunks via
    ``VideoDecoder.persistent()``. Each macro-chunk is moved to GPU
    just-in-time, decoded (with the inner ``tiling_config`` controlling
    per-tile workspace), then dropped before the next chunk lands.

    Consecutive macro-chunks share ``overlap_latent_frames`` latent frames
    (``overlap_latent_frames * 8`` pixel frames). The overlapping pixel
    region is linearly cross-faded so the seam is not visible.

    Yields pixel-frame tensors in the same ``(f, h, w, c)`` float-in-[0,1]
    layout as ``VideoDecoder.__call__``, suitable for direct consumption
    by ``encode_video``.

    Memory profile: GPU residency is bounded by
        max(latent macro-chunk, single tile decode workspace)
    independent of the total video length, which is the whole point of
    this code path.
    """
    if full_latent_cpu.device.type != "cpu":
        # We rely on the caller to have moved the assembled latent to CPU
        # before streaming; otherwise the OOM mitigation is moot.
        logger.warning(
            "[streaming-decode] full_latent_cpu lives on %s, not cpu; "
            "expected memory savings will not materialise.",
            full_latent_cpu.device,
        )

    t_lat = full_latent_cpu.shape[2]
    if chunk_latent_frames >= t_lat:
        # Whole timeline fits in one macro-chunk — just ship and decode.
        # We still use the persistent context so memory cleanup is uniform.
        logger.info(
            "[streaming-decode] full timeline (%d latent frames) fits in "
            "one macro-chunk; using single-shot decode.",
            t_lat,
        )
        with video_decoder.persistent() as decoder:
            chunk_gpu = full_latent_cpu.to(device, non_blocking=False)
            for frames in decoder.decode_video(chunk_gpu, tiling_config, generator):
                # Move to CPU before yielding so the next decoder tile has
                # room. ``encode_video`` consumes CPU/GPU equally well.
                yield frames.detach().to("cpu", non_blocking=False)
            del chunk_gpu
            torch.cuda.empty_cache()
        return

    stride = chunk_latent_frames - overlap_latent_frames
    overlap_px = overlap_latent_frames * _VAE_TEMPORAL_SCALE

    # Build the (start, end) ranges in latent space.
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < t_lat:
        end = min(start + chunk_latent_frames, t_lat)
        ranges.append((start, end))
        if end == t_lat:
            break
        start += stride

    logger.info(
        "[streaming-decode] %d macro-chunks (chunk=%d, overlap=%d latent frames, "
        "%d total latent frames)",
        len(ranges), chunk_latent_frames, overlap_latent_frames, t_lat,
    )

    with video_decoder.persistent() as decoder:
        logger.info(
            "[streaming-decode] decoder built on GPU: allocated=%.2f GB reserved=%.2f GB",
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )
        prev_tail_pix: torch.Tensor | None = None  # CPU, shape [overlap_px, H, W, C]

        for i, (lstart, lend) in enumerate(ranges):
            # 1. Slice + ship to GPU.
            chunk_cpu = full_latent_cpu[:, :, lstart:lend, :, :].contiguous()
            chunk_gpu = chunk_cpu.to(device, non_blocking=False)
            del chunk_cpu
            logger.info(
                "[streaming-decode] chunk %d/%d on GPU shape=%s: allocated=%.2f GB",
                i + 1, len(ranges), tuple(chunk_gpu.shape),
                torch.cuda.memory_allocated() / 1024**3,
            )

            # 2. Decode through the existing tiled-decode path. The inner
            #    ``tiling_config`` already chops further into VAE tiles; the
            #    macro-chunk just bounds how much latent is GPU-resident.
            decoded_parts = [
                frames.detach().to("cpu", non_blocking=False)
                for frames in decoder.decode_video(chunk_gpu, tiling_config, generator)
            ]
            del chunk_gpu
            torch.cuda.empty_cache()

            decoded_cpu = torch.cat(decoded_parts, dim=0)  # (f, h, w, c)
            del decoded_parts

            is_last = i == len(ranges) - 1

            # 3. Stitch with previous macro-chunk's tail via linear cross-fade
            #    over ``overlap_px`` pixel frames.
            if prev_tail_pix is None:
                if is_last or overlap_px == 0:
                    logger.info(
                        "[streaming-decode] yielded macro-chunk %d/%d (%d pixel frames)",
                        i + 1, len(ranges), decoded_cpu.shape[0],
                    )
                    yield decoded_cpu
                    prev_tail_pix = None
                else:
                    head_emit = decoded_cpu[:-overlap_px]
                    prev_tail_pix = decoded_cpu[-overlap_px:].clone()
                    logger.info(
                        "[streaming-decode] yielded macro-chunk %d/%d head (%d pixel frames)",
                        i + 1, len(ranges), head_emit.shape[0],
                    )
                    yield head_emit
            else:
                # Cross-fade the head of this chunk with prev_tail_pix.
                head = decoded_cpu[:overlap_px]
                ramp = torch.linspace(
                    0.0, 1.0, overlap_px, dtype=decoded_cpu.dtype
                ).view(-1, 1, 1, 1)
                blended = prev_tail_pix * (1.0 - ramp) + head * ramp
                logger.info(
                    "[streaming-decode] cross-fade %d/%d (%d pixel frames)",
                    i + 1, len(ranges), blended.shape[0],
                )
                yield blended
                del head, ramp, blended

                if is_last:
                    rest = decoded_cpu[overlap_px:]
                    logger.info(
                        "[streaming-decode] yielded final tail %d/%d (%d pixel frames)",
                        i + 1, len(ranges), rest.shape[0],
                    )
                    yield rest
                    prev_tail_pix = None
                else:
                    middle = decoded_cpu[overlap_px:-overlap_px]
                    prev_tail_pix = decoded_cpu[-overlap_px:].clone()
                    logger.info(
                        "[streaming-decode] yielded macro-chunk %d/%d middle (%d pixel frames)",
                        i + 1, len(ranges), middle.shape[0],
                    )
                    yield middle

            del decoded_cpu
