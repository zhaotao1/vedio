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

from ltx_core.components.diffusion_steps import Res2sDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuider,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.video_vae import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    DEFAULT_NEGATIVE_PROMPT,
    DISTILLED_SIGMAS,
    LTX_2_3_PARAMS,
    STAGE_2_DISTILLED_SIGMAS,
)
from ltx_pipelines.utils.denoisers import (
    FactoryGuidedDenoiser,
    GuidedDenoiser,
    SimpleDenoiser,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
)
from ltx_pipelines.utils.samplers import res2s_audio_video_denoising_loop
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

from long_video_native.core.adain import adain_match
from long_video_native.core.blending import temporal_overlap_blend
from long_video_native.core.conditioning_builder import (
    NegativeIndexSpec,
    OverlapSpec,
    build_video_conditionings,
    slice_anchor,
)
from long_video_native.core.dynamic_conditioning import (
    DynamicConditioningConfig,
    maybe_wrap_denoiser,
)
from long_video_native.core.extend_conditioning import (
    ExtendPrefixSpec,
    build_extend_conditioning,
)
from long_video_native.core.keyframe_router import (
    route_keyframes,
    total_pixel_frames,
)
from long_video_native.core.prompt_cache import PromptContextCache
from long_video_native.core.seeding import calc_tile_seed, per_tile_offset_for
from long_video_native.core.units import VAE_TEMPORAL_SCALE

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

    # --- Grid-tiled VAE decode (ComfyUI-style spatial tiling) -------------
    # When ``vae_grid_horizontal_tiles`` or ``vae_grid_vertical_tiles`` > 1
    # the streaming decoder splits each temporal macro-chunk into an
    # H × W grid in latent space, decodes each grid tile *independently*
    # with the un-tiled VAE (i.e. ``tiling_config=None``), and blends the
    # tiles back together with linear ramps over the overlap region.
    #
    # This bypasses ltx-core's ``tiled_decode`` entirely (which has a
    # workspace-allocation bug at small spatial tile sizes) and gives a
    # predictable GPU-memory budget per tile.
    #
    # ``vae_grid_overlap_latent`` is the per-side overlap *in latent
    # pixels* (1 latent pixel = 32 pixel pixels on H/W). Set to 0 to
    # disable blending (raw tile stitching — may show seams).
    vae_grid_horizontal_tiles: int = 1
    vae_grid_vertical_tiles: int = 1
    vae_grid_overlap_latent: int = 2

    # Stage sigmas (override only for advanced experimentation)
    stage_1_sigmas: torch.Tensor = field(default_factory=lambda: DISTILLED_SIGMAS)
    stage_2_sigmas: torch.Tensor = field(default_factory=lambda: STAGE_2_DISTILLED_SIGMAS)

    # Non-distilled two-stage guidance (ignored by the distilled pipeline).
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    num_inference_steps: int = LTX_2_3_PARAMS.num_inference_steps
    video_guider_params: MultiModalGuiderParams = field(
        default_factory=lambda: LTX_2_3_PARAMS.video_guider_params
    )
    audio_guider_params: MultiModalGuiderParams = field(
        default_factory=lambda: LTX_2_3_PARAMS.audio_guider_params
    )
    max_batch_size: int = 1

    # === ComfyUI LTXVLoopingSampler-aligned options (P0-P8) ===============
    # All default-off / default to current behaviour so existing YAMLs run
    # unchanged. See packages/long-video-native/README.md (TODO) for the
    # full alignment story.

    # P3 — Share a single PromptContextCache across all temporal+spatial
    # tiles. Negative prompt encoded once, positives in batch. Saves time
    # and removes encoding-noise drift between segments.
    share_prompt_encoding: bool = False

    # P4 — Use clean-prefix conditioning (``VideoConditionByLatentIndex``
    # at strength=overlap_strength) instead of the legacy pixel-space
    # ``temporal_overlap_blend``. Mirrors ComfyUI ``LTXVExtendSampler``.
    # The previous tile's tail latent is pinned as a clean prefix at
    # latent_idx=0 of the new tile, so only the remaining frames are
    # denoised.
    use_extend_sampler: bool = False

    # P6 — DynamicConditioning: raise denoise_mask to ``power`` each step.
    dynamic_conditioning_enabled: bool = False
    dynamic_conditioning_power: float = 1.3
    dynamic_conditioning_only_first_frame: bool = True

    # P2 — Per-tile seed scheme.
    #   "per_segment_legacy" — seed = base_seed + segment_idx (current default).
    #   "comfyui"            — seed = base + start*(V*H) + v*H + h + offset.
    seeding_mode: str = "per_segment_legacy"
    per_tile_seed_offsets: tuple[int, ...] = ()

    # P5 — Optional external reference latents (CPU tensors).
    # When set, they override the internally captured anchor / first-tile
    # statistics.
    external_normalizing_latent: torch.Tensor | None = None
    external_negative_index_latent: torch.Tensor | None = None

    # P8 — IC-LoRA guiding-latents placeholder. Not yet wired into the
    # transformer; populated by the runner so future revisions can
    # consume it without breaking the YAML schema.
    guiding_latents: torch.Tensor | None = None
    guiding_strength: float = 1.0
    guiding_start_step: int = 0
    guiding_end_step: int = 1000
    cond_image_strength: float = 1.0

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
        if self.vae_grid_horizontal_tiles < 1 or self.vae_grid_vertical_tiles < 1:
            raise ValueError("vae_grid_*_tiles must be >= 1")
        if self.vae_grid_overlap_latent < 0:
            raise ValueError("vae_grid_overlap_latent must be >= 0")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if self.seeding_mode not in ("per_segment_legacy", "comfyui"):
            raise ValueError(
                f"seeding_mode must be 'per_segment_legacy' or 'comfyui'; "
                f"got {self.seeding_mode!r}"
            )
        if self.dynamic_conditioning_enabled and self.dynamic_conditioning_power <= 0:
            raise ValueError("dynamic_conditioning_power must be > 0")
        if not 0.0 <= self.guiding_strength <= 1.0:
            raise ValueError("guiding_strength must be in [0, 1]")
        if not 0.0 <= self.cond_image_strength <= 1.0:
            raise ValueError("cond_image_strength must be in [0, 1]")
        if self.guiding_latents is not None:
            # P8 placeholder: the YAML wiring lands now so configs stay
            # forward-compatible, but the IC-LoRA branch in the transformer
            # is not yet hooked up. Fail loudly rather than silently
            # ignoring user intent.
            raise NotImplementedError(
                "guiding_latents (IC-LoRA) is reserved for a future release; "
                "leave references.guiding_latents_path unset for now"
            )


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
        cond_images: list[tuple] | None = None,
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
            seed: master seed; each segment derives ``seed + segment_idx``
                (legacy mode) or the ComfyUI tile-seed formula
                (``seeding_mode='comfyui'``).
            height, width: full output resolution (pixel). Must satisfy
                two-stage divisibility (height/width % 64 == 0).
            frame_rate: target fps.
            keyframes_per_segment: optional list aligned with ``prompts``;
                each element is a list of
                ``ImageConditioningInput``-compatible tuples that ltx-pipelines
                accepts (see :func:`combined_image_conditionings`).
            cond_images: optional GLOBAL keyframes — list of
                ``(image_input, global_pixel_frame_idx, strength)`` tuples
                where ``image_input`` is anything accepted by
                ``combined_image_conditionings``. When set, they are
                routed to per-segment positions via :mod:`keyframe_router`
                and **merged with** ``keyframes_per_segment``. Mirrors
                ComfyUI ``optional_cond_images`` + ``cond_image_indices``.
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

        # P1 — global keyframes → per-segment routing.
        # ComfyUI ``LTXVLoopingSampler`` accepts a flat list of pixel-frame
        # keyframe indices and computes which temporal tile each one lands
        # in; we do the same and merge with any per-segment keyframes the
        # caller already supplied.
        if cond_images:
            overlap_pixel = config.overlap_latent_frames * VAE_TEMPORAL_SCALE
            total_pixel = total_pixel_frames(
                len(prompts), config.chunk_num_frames, overlap_pixel
            )
            routes = route_keyframes(
                [int(ci[1]) for ci in cond_images],
                tile_size_pixel=config.chunk_num_frames,
                overlap_pixel=overlap_pixel,
                total_pixel_frames=total_pixel,
                metas=list(cond_images),
            )
            for r in routes:
                meta = r.meta
                if meta is None:
                    continue
                img_input, _, *rest = meta
                strength = float(rest[0]) if rest else config.cond_image_strength
                keyframes_per_segment[r.temporal_tile_index].append(
                    (img_input, r.in_tile_pixel_index, strength)
                )

        # P3 — shared PromptContextCache (skip when share_prompt_encoding=False
        # to preserve the legacy per-segment encode behaviour).
        prompt_cache: PromptContextCache | None = None
        if config.share_prompt_encoding:
            prompt_cache = PromptContextCache(
                self.prompt_encoder,
                positive_prompts=prompts,
                negative_prompt=config.negative_prompt,
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=(
                    keyframes_per_segment[0][0][0]
                    if keyframes_per_segment[0]
                    else None
                ),
                enhance_prompt_seed=seed,
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

        # Anchor latent for negative-index memory is captured after segment 0,
        # OR taken from ``config.external_negative_index_latent`` if provided.
        anchor_latent: torch.Tensor | None = None
        if config.external_negative_index_latent is not None:
            anchor_latent = config.external_negative_index_latent.to(
                self.device, non_blocking=False
            )

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

            # P2 — per-tile seed derivation.
            if config.seeding_mode == "comfyui":
                offset = per_tile_offset_for(
                    list(config.per_tile_seed_offsets), seg_idx
                )
                # Pixel-frame start of this temporal tile.
                start_pix = seg_idx * (
                    config.chunk_num_frames
                    - config.overlap_latent_frames * VAE_TEMPORAL_SCALE
                )
                seg_seed = calc_tile_seed(
                    base_seed=seed,
                    start_index=start_pix,
                    vertical_tiles=1,
                    horizontal_tiles=1,
                    v=0,
                    h=0,
                    per_tile_offset=offset,
                )
            else:
                seg_seed = seed + seg_idx

            logger.info(
                "[long-video] segment %d/%d seed=%d prompt=%r",
                seg_idx + 1, len(prompts), seg_seed, prompt[:60],
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                logger.info(
                    "[mem] seg %d/%d BEFORE generate: allocated=%.2f GB reserved=%.2f GB",
                    seg_idx + 1, len(prompts),
                    torch.cuda.memory_allocated() / 1024**3,
                    torch.cuda.memory_reserved() / 1024**3,
                )

            video_latent, audio_latent = self.generate_one_segment(
                prompt=prompt,
                seed=seg_seed,
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
                prompt_contexts=(
                    (prompt_cache.positive_for(seg_idx), prompt_cache.negative)
                    if prompt_cache is not None
                    else None
                ),
                tile_index=seg_idx,
            )
            if prev_tail is not None:
                del prev_tail

            # P5 — AdaIN: prefer external normalizing latent if provided,
            # otherwise fall back to the first-segment reference.
            if seg_idx > 0 and config.adain_factor > 0:
                if config.external_normalizing_latent is not None:
                    ref = config.external_normalizing_latent.to(
                        video_latent.device, non_blocking=False
                    )
                else:
                    ref = all_video_latents[0].to(
                        video_latent.device, non_blocking=False
                    )
                video_latent = adain_match(video_latent, ref, factor=config.adain_factor)
                del ref

            # Snapshot the anchor from segment 0 BEFORE moving to CPU. Anchor
            # is small (a couple latent frames) and lives on GPU between
            # segments. Skip if user supplied an external one already.
            if (
                seg_idx == 0
                and config.enable_negative_index
                and config.external_negative_index_latent is None
            ):
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

            # --- diagnostics: per-segment latent statistics ---------------
            # Drift detector. If ``std`` collapses toward 0 or ``mean``
            # drifts hard between segments, the negative-index anchor /
            # AdaIN / overlap blend is over-constraining. Look for:
            #   * std monotonically shrinking across segments → collapse
            #   * mean drifting toward the first segment's mean → AdaIN/anchor pull
            #   * tail_std << full_std → end-of-segment fade-to-black
            with torch.no_grad():
                vl = video_latent.float()
                tail_n = min(
                    config.overlap_latent_frames if config.overlap_latent_frames > 0 else 1,
                    vl.shape[2],
                )
                tail = vl[:, :, -tail_n:, :, :]
                head = vl[:, :, :tail_n, :, :]
                logger.info(
                    "[long-video] segment %d/%d stats: "
                    "shape=%s mean=%.4f std=%.4f | head(mean=%.4f std=%.4f) "
                    "tail(mean=%.4f std=%.4f) abs_max=%.3f",
                    seg_idx + 1, len(prompts),
                    tuple(vl.shape),
                    vl.mean().item(), vl.std().item(),
                    head.mean().item(), head.std().item(),
                    tail.mean().item(), tail.std().item(),
                    vl.abs().max().item(),
                )

            all_video_latents.append(video_latent)
            all_audio_latents.append(audio_latent)

            if torch.cuda.is_available():
                logger.info(
                    "[mem] seg %d/%d AFTER  generate: allocated=%.2f GB reserved=%.2f GB "
                    "peak_allocated=%.2f GB peak_reserved=%.2f GB",
                    seg_idx + 1, len(prompts),
                    torch.cuda.memory_allocated() / 1024**3,
                    torch.cuda.memory_reserved() / 1024**3,
                    torch.cuda.max_memory_allocated() / 1024**3,
                    torch.cuda.max_memory_reserved() / 1024**3,
                )

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
        # Per-quarter latent-frame stats on the assembled timeline. If the
        # last quarter's std collapses (≪ first quarter's std), generation
        # itself faded out before decode. If std is healthy here but the
        # rendered video is still black at the tail, the leak is in decode
        # / encode (look at streaming-decode chunk logs).
        with torch.no_grad():
            fvl = full_video_latent.float()
            t = fvl.shape[2]
            if t >= 4:
                q = t // 4
                for qi, (s, e) in enumerate([(0, q), (q, 2 * q), (2 * q, 3 * q), (3 * q, t)]):
                    seg = fvl[:, :, s:e, :, :]
                    logger.info(
                        "[long-video] timeline quarter %d/4 [latent %d:%d] "
                        "mean=%.4f std=%.4f abs_max=%.3f",
                        qi + 1, s, e,
                        seg.mean().item(), seg.std().item(), seg.abs().max().item(),
                    )
            del fvl
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
                grid_horizontal_tiles=config.vae_grid_horizontal_tiles,
                grid_vertical_tiles=config.vae_grid_vertical_tiles,
                grid_overlap_latent=config.vae_grid_overlap_latent,
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
        prompt_contexts: tuple[
            tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
        tile_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the distilled 2-stage pipeline for a single segment.

        Returns ``(video_latent, audio_latent)`` — unpatchified 5-D / 3-D
        tensors, NOT decoded.

        This method mirrors ``DistilledPipeline.__call__`` (distilled.py
        L87) but injects extra conditionings for the overlap and the
        negative-index anchor. It is the unit of work consumed by the
        spatial-tiling wrapper.

        Args:
            prompt_contexts: optional pre-encoded ``((pos_v, pos_a),
                (neg_v, neg_a))`` from a :class:`PromptContextCache`.
                When supplied, the per-segment ``prompt_encoder`` call is
                skipped (P3 — shared prompt encoding).
            tile_index: this segment's temporal tile index, for logging /
                forward-compat hooks.
        """
        dtype = torch.bfloat16
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        if prompt_contexts is not None:
            (video_context, audio_context), _neg_unused = prompt_contexts
        else:
            (ctx_p,) = self.prompt_encoder(
                [prompt],
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            )
            video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        # P6 — optionally wrap the denoiser to apply DynamicConditioning
        # power schedule each step (mutates ``video_state.denoise_mask``).
        dyn_cfg = DynamicConditioningConfig(
            enabled=config.dynamic_conditioning_enabled,
            power=config.dynamic_conditioning_power,
            only_first_frame=config.dynamic_conditioning_only_first_frame,
        )
        denoiser_s1 = maybe_wrap_denoiser(
            SimpleDenoiser(video_context, audio_context), dyn_cfg
        )

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
            denoiser=denoiser_s1,
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
            denoiser=maybe_wrap_denoiser(
                SimpleDenoiser(video_context, audio_context), dyn_cfg
            ),
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
        # P4 — ``use_extend_sampler``: switch overlap conditioning from
        # ``VideoConditionByKeyframeIndex`` (frame-level RoPE injection) to
        # ``VideoConditionByLatentIndex`` (clean prefix at latent_idx=0).
        # The latter pins the prefix tokens via denoise_mask=0 so only the
        # remaining tile frames are denoised — the mechanism ComfyUI uses
        # in ``LTXVExtendSampler``.
        if (
            config.use_extend_sampler
            and prev_tail_latent is not None
            and config.overlap_latent_frames > 0
        ):
            extend_conds = build_extend_conditioning(
                ExtendPrefixSpec(
                    prefix_latent=prev_tail_latent,
                    strength=config.overlap_strength,
                    latent_idx=0,
                )
            )
            negative: NegativeIndexSpec | None = None
            if anchor_latent is not None and config.enable_negative_index:
                negative = NegativeIndexSpec(
                    anchor_latent=anchor_latent,
                    negative_frame_idx=config.negative_frame_offset,
                    strength=config.negative_index_strength,
                    num_pixel_frames=max(8 * anchor_latent.shape[2], 2),
                )
            return extend_conds + build_video_conditionings(negative_index=negative)

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


class TwoStagesLoopingPipeline(LoopingPipeline):
    """Long-video wrapper around the non-distilled two-stage pipeline.

    The long-video stitching/streaming decode path is inherited from
    :class:`LoopingPipeline`; this class only swaps the per-segment latent
    generation to the full/dev checkpoint + distilled LoRA recipe.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
        offload_mode: OffloadMode = OffloadMode.NONE,
    ) -> None:
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self._scheduler = LTX2Scheduler()

        self.prompt_encoder = PromptEncoder(
            checkpoint_path, gemma_root, self.dtype, self.device,
            registry=registry, offload_mode=offload_mode,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path, self.dtype, self.device, registry=registry
        )
        self.upsampler = VideoUpsampler(
            checkpoint_path, spatial_upsampler_path, self.dtype, self.device,
            registry=registry,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path, self.dtype, self.device, registry=registry
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path, self.dtype, self.device, registry=registry
        )

        self.stage_1 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
            offload_mode=offload_mode,
        )
        self.stage_2 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=(*tuple(loras), *distilled_lora),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
            offload_mode=offload_mode,
        )

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
        prompt_contexts: tuple[
            tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
        tile_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        if prompt_contexts is not None:
            (v_context_p, a_context_p), (v_context_n, a_context_n) = prompt_contexts
        else:
            ctx_p, ctx_n = self.prompt_encoder(
                [prompt, config.negative_prompt],
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
                enhance_prompt_seed=seed,
            )
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        dyn_cfg = DynamicConditioningConfig(
            enabled=config.dynamic_conditioning_enabled,
            power=config.dynamic_conditioning_power,
            only_first_frame=config.dynamic_conditioning_only_first_frame,
        )

        stage_1_h, stage_1_w = height // 2, width // 2
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
        extra_conds_s1 = self._build_continuity_conditionings(
            prev_tail_latent=_maybe_resize_latent(
                prev_tail_latent, target_h=stage_1_h // 32, target_w=stage_1_w // 32
            ),
            anchor_latent=_maybe_resize_latent(
                anchor_latent, target_h=stage_1_h // 32, target_w=stage_1_w // 32
            ),
            config=config,
        )

        stage_1_sigmas = self._scheduler.execute(
            steps=config.num_inference_steps
        ).to(dtype=torch.float32, device=self.device)
        video_state, audio_state = self.stage_1(
            denoiser=maybe_wrap_denoiser(FactoryGuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider_factory=create_multimodal_guider_factory(
                    params=config.video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider_factory=create_multimodal_guider_factory(
                    params=config.audio_guider_params,
                    negative_context=a_context_n,
                ),
            ), dyn_cfg),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_w,
            height=stage_1_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_1_image_conds + extra_conds_s1,
            ),
            audio=ModalitySpec(context=a_context_p),
            max_batch_size=config.max_batch_size,
        )

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
            prev_tail_latent=prev_tail_latent,
            anchor_latent=anchor_latent,
            config=config,
        )

        video_state, audio_state = self.stage_2(
            denoiser=maybe_wrap_denoiser(
                SimpleDenoiser(v_context=v_context_p, a_context=a_context_p), dyn_cfg
            ),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_image_conds + extra_conds_s2,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )

        return video_state.latent, audio_state.latent


class TwoStagesHQLoopingPipeline(TwoStagesLoopingPipeline):
    """Long-video wrapper around the HQ Res2S two-stage pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        distilled_lora_strength_stage_1: float = 0.25,
        distilled_lora_strength_stage_2: float = 0.5,
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
        offload_mode: OffloadMode = OffloadMode.NONE,
    ) -> None:
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self._scheduler = LTX2Scheduler()

        distilled_lora_stage_1 = LoraPathStrengthAndSDOps(
            path=distilled_lora[0].path,
            strength=distilled_lora_strength_stage_1,
            sd_ops=distilled_lora[0].sd_ops,
        )
        distilled_lora_stage_2 = LoraPathStrengthAndSDOps(
            path=distilled_lora[0].path,
            strength=distilled_lora_strength_stage_2,
            sd_ops=distilled_lora[0].sd_ops,
        )

        self.prompt_encoder = PromptEncoder(
            checkpoint_path, gemma_root, self.dtype, self.device,
            registry=registry, offload_mode=offload_mode,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path, self.dtype, self.device, registry=registry
        )
        self.upsampler = VideoUpsampler(
            checkpoint_path, spatial_upsampler_path, self.dtype, self.device,
            registry=registry,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path, self.dtype, self.device, registry=registry
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path, self.dtype, self.device, registry=registry
        )
        self.stage_1 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=(*tuple(loras), distilled_lora_stage_1),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
            offload_mode=offload_mode,
        )
        self.stage_2 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=(*tuple(loras), distilled_lora_stage_2),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
            offload_mode=offload_mode,
        )

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
        prompt_contexts: tuple[
            tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
        tile_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        if prompt_contexts is not None:
            (v_context_p, a_context_p), (v_context_n, a_context_n) = prompt_contexts
        else:
            ctx_p, ctx_n = self.prompt_encoder(
                [prompt, config.negative_prompt],
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
                enhance_prompt_seed=seed,
            )
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        dyn_cfg = DynamicConditioningConfig(
            enabled=config.dynamic_conditioning_enabled,
            power=config.dynamic_conditioning_power,
            only_first_frame=config.dynamic_conditioning_only_first_frame,
        )

        stage_1_h, stage_1_w = height // 2, width // 2
        stage_1_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=stage_1_w, height=stage_1_h,
            fps=frame_rate,
        )
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
        extra_conds_s1 = self._build_continuity_conditionings(
            prev_tail_latent=_maybe_resize_latent(
                prev_tail_latent, target_h=stage_1_h // 32, target_w=stage_1_w // 32
            ),
            anchor_latent=_maybe_resize_latent(
                anchor_latent, target_h=stage_1_h // 32, target_w=stage_1_w // 32
            ),
            config=config,
        )

        empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_shape).to_torch_shape())
        stage_1_sigmas = self._scheduler.execute(
            latent=empty_latent, steps=config.num_inference_steps
        ).to(dtype=torch.float32, device=self.device)
        stepper = Res2sDiffusionStep()
        video_state, audio_state = self.stage_1(
            denoiser=maybe_wrap_denoiser(GuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider=MultiModalGuider(
                    params=config.video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider=MultiModalGuider(
                    params=config.audio_guider_params,
                    negative_context=a_context_n,
                ),
            ), dyn_cfg),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            stepper=stepper,
            width=stage_1_w,
            height=stage_1_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_1_image_conds + extra_conds_s1,
            ),
            audio=ModalitySpec(context=a_context_p),
            loop=res2s_audio_video_denoising_loop,
            max_batch_size=config.max_batch_size,
        )

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
            prev_tail_latent=prev_tail_latent,
            anchor_latent=anchor_latent,
            config=config,
        )

        video_state, audio_state = self.stage_2(
            denoiser=maybe_wrap_denoiser(
                SimpleDenoiser(v_context=v_context_p, a_context=a_context_p), dyn_cfg
            ),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            stepper=stepper,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_image_conds + extra_conds_s2,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
            loop=res2s_audio_video_denoising_loop,
        )

        return video_state.latent, audio_state.latent


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
# VAE spatial compression factor: 1 latent pixel ↔ 32 image pixels (H/W).
_VAE_SPATIAL_SCALE = 32


def _grid_tiled_decode_chunk(
    decoder,
    chunk_gpu: torch.Tensor,        # (B, C, T_lat, H_lat, W_lat) on GPU
    *,
    horizontal_tiles: int,
    vertical_tiles: int,
    overlap_latent: int,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    """Decode one temporal macro-chunk using ComfyUI-style spatial grid tiling.

    Splits ``chunk_gpu`` into a ``vertical_tiles × horizontal_tiles`` grid
    in latent H/W, decodes each tile *independently* via the un-tiled VAE,
    and accumulates the per-tile pixel output into a single CPU tensor
    using a linear blend ramp over the per-side ``overlap_latent`` border.

    Returns: CPU tensor of shape ``(F_px, H_px, W_px, C)`` (float, [0, 1]).
    """
    b, c, t_lat, h_lat, w_lat = chunk_gpu.shape
    assert b == 1, f"grid tiled decode expects batch=1, got {b}"

    t_px = (t_lat - 1) * _VAE_TEMPORAL_SCALE + 1  # LTX VAE temporal output = (T-1)*8+1
    h_px = h_lat * _VAE_SPATIAL_SCALE
    w_px = w_lat * _VAE_SPATIAL_SCALE
    overlap_px = overlap_latent * _VAE_SPATIAL_SCALE

    # Latent-grid coordinates: each tile is `tile_h × tile_w` core + overlap on each interior side.
    # Compute even base sizes (rounded down) and distribute remainder.
    def _split_ranges(total: int, n_tiles: int, ov: int) -> list[tuple[int, int, int, int]]:
        """Returns list of (lat_start, lat_end, core_start, core_end) per tile.

        - lat_start/lat_end: latent slice fed to the VAE (includes overlap).
        - core_start/core_end: latent indices in the *output coord system* this
          tile is responsible for (used to place the tile into the canvas).
        """
        if n_tiles <= 1:
            return [(0, total, 0, total)]
        base = total // n_tiles
        rem = total - base * n_tiles
        ranges: list[tuple[int, int, int, int]] = []
        cursor = 0
        for i in range(n_tiles):
            core_len = base + (1 if i < rem else 0)
            core_start = cursor
            core_end = cursor + core_len
            lat_start = max(0, core_start - ov)
            lat_end = min(total, core_end + ov)
            ranges.append((lat_start, lat_end, core_start, core_end))
            cursor = core_end
        return ranges

    v_ranges = _split_ranges(h_lat, vertical_tiles, overlap_latent)
    h_ranges = _split_ranges(w_lat, horizontal_tiles, overlap_latent)

    # Output canvas on CPU + weight canvas on CPU. We use float32 to avoid
    # blend precision loss; final cast happens at the end.
    out_cpu = torch.zeros((t_px, h_px, w_px, 3), dtype=torch.float32)
    weight_cpu = torch.zeros((1, h_px, w_px, 1), dtype=torch.float32)

    n_tiles_total = vertical_tiles * horizontal_tiles
    tile_idx = 0
    for vi, (v_lat_s, v_lat_e, v_core_s, v_core_e) in enumerate(v_ranges):
        for hi, (h_lat_s, h_lat_e, h_core_s, h_core_e) in enumerate(h_ranges):
            tile_idx += 1
            tile_latent = chunk_gpu[:, :, :, v_lat_s:v_lat_e, h_lat_s:h_lat_e].contiguous()
            logger.info(
                "[grid-decode] tile %d/%d (v=%d,h=%d) latent shape=%s allocated=%.2f GB",
                tile_idx, n_tiles_total, vi, hi, tuple(tile_latent.shape),
                torch.cuda.memory_allocated() / 1024**3,
            )
            # Un-tiled decode of this small tile. CRITICAL: wrap in
            # ``torch.no_grad()`` — otherwise the VAE forward retains the
            # full autograd graph (activations of every conv3d / upsample
            # stage) and the per-tile workspace (~11 GB) leaks across
            # tiles. With no_grad the graph is dropped at function exit
            # and ``empty_cache()`` actually reclaims everything.
            with torch.no_grad():
                decoded_parts = list(decoder.decode_video(tile_latent, None, generator))
            del tile_latent
            # Each yielded chunk is (f, h, w, c) on GPU.
            tile_pix_gpu = torch.cat(decoded_parts, dim=0) if len(decoded_parts) > 1 else decoded_parts[0]
            del decoded_parts
            tile_pix_cpu = tile_pix_gpu.detach().to("cpu", non_blocking=False).float()
            del tile_pix_gpu
            torch.cuda.empty_cache()

            # Pixel coordinates for this tile's latent slice.
            tile_h_px = (v_lat_e - v_lat_s) * _VAE_SPATIAL_SCALE
            tile_w_px = (h_lat_e - h_lat_s) * _VAE_SPATIAL_SCALE
            # Sanity (last latent row/col may be off if total is odd vs scale, but LTX always 32-aligned).
            assert tile_pix_cpu.shape[1] == tile_h_px, (
                f"tile pixel H mismatch: {tile_pix_cpu.shape[1]} vs {tile_h_px}"
            )
            assert tile_pix_cpu.shape[2] == tile_w_px, (
                f"tile pixel W mismatch: {tile_pix_cpu.shape[2]} vs {tile_w_px}"
            )

            # Build a 2D linear-ramp blend mask for this tile in pixel space.
            mask = torch.ones((1, tile_h_px, tile_w_px, 1), dtype=torch.float32)
            # Vertical ramp (top/bottom):
            if vi > 0 and overlap_px > 0:
                ramp = torch.linspace(0.0, 1.0, overlap_px, dtype=torch.float32).view(1, -1, 1, 1)
                mask[:, :overlap_px, :, :] *= ramp
            if vi < vertical_tiles - 1 and overlap_px > 0:
                ramp = torch.linspace(1.0, 0.0, overlap_px, dtype=torch.float32).view(1, -1, 1, 1)
                mask[:, -overlap_px:, :, :] *= ramp
            # Horizontal ramp (left/right):
            if hi > 0 and overlap_px > 0:
                ramp = torch.linspace(0.0, 1.0, overlap_px, dtype=torch.float32).view(1, 1, -1, 1)
                mask[:, :, :overlap_px, :] *= ramp
            if hi < horizontal_tiles - 1 and overlap_px > 0:
                ramp = torch.linspace(1.0, 0.0, overlap_px, dtype=torch.float32).view(1, 1, -1, 1)
                mask[:, :, -overlap_px:, :] *= ramp

            # Paste into canvas.
            y0 = v_lat_s * _VAE_SPATIAL_SCALE
            y1 = y0 + tile_h_px
            x0 = h_lat_s * _VAE_SPATIAL_SCALE
            x1 = x0 + tile_w_px
            out_cpu[:, y0:y1, x0:x1, :] += tile_pix_cpu * mask
            weight_cpu[:, y0:y1, x0:x1, :] += mask

            del tile_pix_cpu, mask

    weight_cpu = weight_cpu.clamp_min(1e-8)
    out_cpu = out_cpu / weight_cpu
    return out_cpu.to(torch.float32)


def _streaming_video_decode(
    video_decoder,  # ltx_pipelines.utils.blocks.VideoDecoder (avoid hard import)
    full_latent_cpu: torch.Tensor,
    *,
    tiling_config: TilingConfig | None,
    chunk_latent_frames: int,
    overlap_latent_frames: int,
    device: torch.device,
    generator: torch.Generator | None,
    grid_horizontal_tiles: int = 1,
    grid_vertical_tiles: int = 1,
    grid_overlap_latent: int = 2,
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

    use_grid = grid_horizontal_tiles > 1 or grid_vertical_tiles > 1
    if use_grid:
        logger.info(
            "[streaming-decode] using ComfyUI-style spatial grid tiling: "
            "%dx%d (overlap_latent=%d)",
            grid_vertical_tiles, grid_horizontal_tiles, grid_overlap_latent,
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
            if use_grid:
                decoded_cpu = _grid_tiled_decode_chunk(
                    decoder, chunk_gpu,
                    horizontal_tiles=grid_horizontal_tiles,
                    vertical_tiles=grid_vertical_tiles,
                    overlap_latent=grid_overlap_latent,
                    generator=generator,
                    device=device,
                )
                del chunk_gpu
                torch.cuda.empty_cache()
                yield decoded_cpu
            else:
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
    if tiling_config is None:
        logger.warning("[streaming-decode] tiling_config is None — VAE will run un-tiled. This is the OOM path.")
    else:
        logger.info(
            "[streaming-decode] tiling_config: temporal=%s spatial=%s",
            tiling_config.temporal_config, tiling_config.spatial_config,
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
            # On the FIRST chunk only, enable CUDA memory snapshot so an OOM
            # produces an attributed allocation dump we can analyse offline.
            if i == 0:
                try:
                    torch.cuda.memory._record_memory_history(max_entries=200_000)
                except Exception as _e:  # noqa: BLE001
                    logger.warning("[streaming-decode] could not enable memory history: %s", _e)
            try:
                if use_grid:
                    decoded_cpu = _grid_tiled_decode_chunk(
                        decoder, chunk_gpu,
                        horizontal_tiles=grid_horizontal_tiles,
                        vertical_tiles=grid_vertical_tiles,
                        overlap_latent=grid_overlap_latent,
                        generator=generator,
                        device=device,
                    )
                    decoded_parts = None  # not used in grid path
                else:
                    decoded_parts = [
                        frames.detach().to("cpu", non_blocking=False)
                        for frames in decoder.decode_video(chunk_gpu, tiling_config, generator)
                    ]
            except torch.cuda.OutOfMemoryError:
                snap_path = "/tmp/streaming_decode_oom_snapshot.pickle"
                try:
                    torch.cuda.memory._dump_snapshot(snap_path)
                    logger.error(
                        "[streaming-decode] OOM at chunk %d/%d; snapshot dumped to %s",
                        i + 1, len(ranges), snap_path,
                    )
                except Exception as _e:  # noqa: BLE001
                    logger.error("[streaming-decode] OOM but snapshot dump failed: %s", _e)
                # also dump memory_stats
                try:
                    stats = torch.cuda.memory_stats()
                    logger.error(
                        "[streaming-decode] memory_stats: active.all.current=%d (%.2f GB), "
                        "active.all.peak=%d (%.2f GB), allocation.all.current=%d, allocation.all.peak=%d",
                        stats.get("active.all.current", -1),
                        stats.get("active_bytes.all.current", 0) / 1024**3,
                        stats.get("active.all.peak", -1),
                        stats.get("active_bytes.all.peak", 0) / 1024**3,
                        stats.get("allocation.all.current", -1),
                        stats.get("allocation.all.peak", -1),
                    )
                except Exception:  # noqa: BLE001
                    pass
                raise
            finally:
                if i == 0:
                    try:
                        torch.cuda.memory._record_memory_history(enabled=None)
                    except Exception:  # noqa: BLE001
                        pass
            del chunk_gpu
            torch.cuda.empty_cache()

            if not use_grid:
                decoded_cpu = torch.cat(decoded_parts, dim=0)  # (f, h, w, c)
                del decoded_parts
            # In grid mode, decoded_cpu is already produced by _grid_tiled_decode_chunk.

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
