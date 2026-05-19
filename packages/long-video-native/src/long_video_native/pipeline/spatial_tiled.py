"""Spatial tiling wrapper around :class:`LoopingPipeline`.

Direct port of ComfyUI ``LTXVLoopingSampler``'s 2-D spatial tiling
(``_create_spatial_weights`` + the (v, h) inner loop in
``_process_temporal_chunks``).

Algorithm
---------

For each segment (temporal chunk) we partition the full ``[height, width]``
target into a grid of overlapping spatial tiles. Every tile runs through
``LoopingPipeline.generate_one_segment`` independently — so it gets its own
prompt-conditioning, its own previous-tail latent (per-tile memory), and its
own RNG seed. Tile outputs are merged in latent space using a 2-D cosine-
style blend mask (linear ramp on each side that borders another tile).

Keyframe reprojection: every user-supplied global keyframe is cropped /
re-coordinated for each tile it overlaps.

Tile memory: a :class:`TileMemory` instance stores the previous-segment
tail latent per ``(v, h)`` cell and, when negative-index is enabled, a
per-tile anchor latent captured after segment 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from ltx_core.model.video_vae import TilingConfig
from ltx_core.types import Audio
from ltx_pipelines.utils.helpers import assert_resolution

from long_video_native.core.adain import adain_match
from long_video_native.core.blending import (
    plan_tile_grid,
    spatial_tile_weights,
    temporal_overlap_blend,
)
from long_video_native.core.conditioning_builder import slice_anchor
from long_video_native.core.state import TileMemory
from long_video_native.pipeline.looping import LoopingConfig, LoopingPipeline

logger = logging.getLogger(__name__)

# VAE spatial scale factor for LTX-2 (matches ltx_core.types.VIDEO_SCALE_FACTORS).
_LATENT_SPATIAL_SCALE = 32


@dataclass(frozen=True)
class SpatialTilingConfig:
    """Configuration for spatial tile generation.

    All sizes are in **pixel** units; the wrapper converts to latent units
    automatically using the LTX-2 VAE scale (32 spatial / 8 temporal).
    """

    tile_height_px: int = 704
    tile_width_px: int = 1216
    spatial_overlap_px: int = 128         # overlap between adjacent tiles
    per_tile_seed_stride: int = 1009      # large prime, prevents tile correlation

    def __post_init__(self) -> None:
        for name, val in (
            ("tile_height_px", self.tile_height_px),
            ("tile_width_px", self.tile_width_px),
            ("spatial_overlap_px", self.spatial_overlap_px),
        ):
            if val <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.tile_height_px % 64 != 0 or self.tile_width_px % 64 != 0:
            raise ValueError(
                "tile_height_px and tile_width_px must be divisible by 64 "
                "(two-stage second stage requirement)"
            )
        if self.spatial_overlap_px % _LATENT_SPATIAL_SCALE != 0:
            raise ValueError(
                f"spatial_overlap_px must be a multiple of "
                f"{_LATENT_SPATIAL_SCALE} (VAE spatial scale)"
            )


class SpatialTiledLoopingPipeline:
    """Wraps a :class:`LoopingPipeline` to add spatial tiling.

    Construction takes an already-built inner pipeline so the heavy model
    components (transformer builder, encoders, decoders) are shared and the
    wrapper introduces no extra weight loading.
    """

    def __init__(self, inner: LoopingPipeline) -> None:
        self.inner = inner

    # ------------------------------------------------------------------ API

    @torch.inference_mode()
    def generate_long(
        self,
        prompts: list[str],
        *,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        keyframes_per_segment: list[list] | None = None,
        looping_config: LoopingConfig | None = None,
        spatial_config: SpatialTilingConfig | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[torch.Tensor, Audio]:
        """Generate a long, high-resolution video using temporal × spatial
        tiling.

        Returns ``(decoded_video_iterator, decoded_audio)`` — same shape as
        :meth:`LoopingPipeline.generate_long`.
        """
        if not prompts:
            raise ValueError("prompts must be non-empty")
        looping_config = looping_config or LoopingConfig()
        spatial_config = spatial_config or SpatialTilingConfig()
        assert_resolution(height=height, width=width, is_two_stage=True)
        keyframes_per_segment = keyframes_per_segment or [[] for _ in prompts]
        if len(keyframes_per_segment) != len(prompts):
            raise ValueError("keyframes_per_segment length must match prompts length")

        # --- plan tile grid ------------------------------------------------
        v_tiles = plan_tile_grid(
            height, spatial_config.tile_height_px, spatial_config.spatial_overlap_px
        )
        h_tiles = plan_tile_grid(
            width, spatial_config.tile_width_px, spatial_config.spatial_overlap_px
        )
        n_v, n_h = len(v_tiles), len(h_tiles)
        logger.info(
            "[spatial-tiling] grid %dx%d (target %dx%d, tile %dx%d, overlap %dpx)",
            n_v, n_h, height, width,
            spatial_config.tile_height_px, spatial_config.tile_width_px,
            spatial_config.spatial_overlap_px,
        )

        if n_v == 1 and n_h == 1:
            # Fall through to the plain temporal pipeline — no tiling needed.
            logger.info("[spatial-tiling] grid is 1x1, delegating to LoopingPipeline.")
            return self.inner.generate_long(
                prompts=prompts,
                seed=seed,
                height=height,
                width=width,
                frame_rate=frame_rate,
                keyframes_per_segment=keyframes_per_segment,
                config=looping_config,
                tiling_config=tiling_config,
                enhance_prompt=enhance_prompt,
            )

        memory = TileMemory(vertical_tiles=n_v, horizontal_tiles=n_h)

        all_video_latents: list[torch.Tensor] = []
        # Audio is generated only by the centre tile to avoid 4× redundant
        # generation; in practice ComfyUI does the same since audio is short
        # and global to the scene rather than spatially localised.
        all_audio_latents: list[torch.Tensor] = []
        centre_v = n_v // 2
        centre_h = n_h // 2

        for seg_idx, prompt in enumerate(prompts):
            logger.info(
                "[spatial-tiling] segment %d/%d prompt=%r",
                seg_idx + 1, len(prompts), prompt[:60],
            )

            # Allocate the segment's full-resolution accumulator in latent
            # space. We only know the latent T/H/W after the first tile
            # returns — lazily allocate on first tile.
            segment_accumulator: torch.Tensor | None = None
            segment_weight_sum: torch.Tensor | None = None

            for v_idx, (v_px_start, v_px_end) in enumerate(v_tiles):
                for h_idx, (h_px_start, h_px_end) in enumerate(h_tiles):
                    tile_h_px = v_px_end - v_px_start
                    tile_w_px = h_px_end - h_px_start

                    # Reproject global keyframes to this tile's local coords.
                    tile_keyframes = _reproject_keyframes(
                        keyframes_per_segment[seg_idx],
                        v_px_start, v_px_end, h_px_start, h_px_end,
                    )

                    # Per-tile seed: differs across (seg, v, h) so each tile
                    # gets independent noise — prevents repeating patterns.
                    # ``looping_config.seeding_mode='comfyui'`` switches to
                    # the ComfyUI ``calc_tile_seed`` formula
                    # (base + start*V*H + v*H + h + per_tile_offset).
                    if looping_config.seeding_mode == "comfyui":
                        from long_video_native.core.seeding import (  # noqa: PLC0415
                            calc_tile_seed,
                            per_tile_offset_for,
                        )
                        overlap_pix = (
                            looping_config.overlap_latent_frames * 8
                        )
                        start_pix = seg_idx * (
                            looping_config.chunk_num_frames - overlap_pix
                        )
                        tile_seed = calc_tile_seed(
                            base_seed=seed,
                            start_index=start_pix,
                            vertical_tiles=n_v,
                            horizontal_tiles=n_h,
                            v=v_idx,
                            h=h_idx,
                            per_tile_offset=per_tile_offset_for(
                                list(looping_config.per_tile_seed_offsets),
                                v_idx * n_h + h_idx,
                            ),
                        )
                    else:
                        tile_seed = (
                            seed
                            + seg_idx * (n_v * n_h)
                            + (v_idx * n_h + h_idx) * spatial_config.per_tile_seed_stride
                        )

                    prev_tail = memory.get_tail(v_idx, h_idx)
                    anchor = (
                        memory.get_anchor(v_idx, h_idx)
                        if looping_config.enable_negative_index
                        else None
                    )

                    logger.info(
                        "[spatial-tiling]  tile (v=%d,h=%d) px=[%d:%d,%d:%d] seed=%d",
                        v_idx, h_idx, v_px_start, v_px_end, h_px_start, h_px_end,
                        tile_seed,
                    )

                    tile_video_latent, tile_audio_latent = (
                        self.inner.generate_one_segment(
                            prompt=prompt,
                            seed=tile_seed,
                            height=tile_h_px,
                            width=tile_w_px,
                            num_frames=looping_config.chunk_num_frames,
                            frame_rate=frame_rate,
                            images=tile_keyframes,
                            prev_tail_latent=prev_tail,
                            anchor_latent=anchor,
                            config=looping_config,
                            enhance_prompt=(
                                enhance_prompt and seg_idx == 0
                                and v_idx == centre_v and h_idx == centre_h
                            ),
                        )
                    )

                    # Allocate the segment accumulator now that latent shape
                    # is known.
                    if segment_accumulator is None:
                        full_latent_h = height // _LATENT_SPATIAL_SCALE
                        full_latent_w = width // _LATENT_SPATIAL_SCALE
                        latent_t = tile_video_latent.shape[2]
                        segment_accumulator = torch.zeros(
                            tile_video_latent.shape[0],
                            tile_video_latent.shape[1],
                            latent_t,
                            full_latent_h,
                            full_latent_w,
                            dtype=tile_video_latent.dtype,
                            device=tile_video_latent.device,
                        )
                        segment_weight_sum = torch.zeros_like(segment_accumulator)

                    # Blend the tile back into the segment accumulator.
                    spatial_overlap_lat = (
                        spatial_config.spatial_overlap_px // _LATENT_SPATIAL_SCALE
                    )
                    weights = spatial_tile_weights(
                        tile_shape=tile_video_latent.shape,
                        v_idx=v_idx,
                        h_idx=h_idx,
                        vertical_tiles=n_v,
                        horizontal_tiles=n_h,
                        spatial_overlap=spatial_overlap_lat,
                        device=tile_video_latent.device,
                        dtype=tile_video_latent.dtype,
                    )
                    v_lat_start = v_px_start // _LATENT_SPATIAL_SCALE
                    h_lat_start = h_px_start // _LATENT_SPATIAL_SCALE
                    v_lat_end = v_lat_start + tile_video_latent.shape[3]
                    h_lat_end = h_lat_start + tile_video_latent.shape[4]
                    segment_accumulator[
                        :, :, :, v_lat_start:v_lat_end, h_lat_start:h_lat_end
                    ] += tile_video_latent * weights
                    segment_weight_sum[
                        :, :, :, v_lat_start:v_lat_end, h_lat_start:h_lat_end
                    ] += weights

                    # Update per-tile memory for next segment's overlap.
                    if looping_config.overlap_latent_frames > 0:
                        memory.set_tail(
                            v_idx, h_idx,
                            tile_video_latent[
                                :, :, -looping_config.overlap_latent_frames:, :, :
                            ].clone(),
                        )

                    # Snapshot per-tile anchor after segment 0.
                    if seg_idx == 0 and looping_config.enable_negative_index:
                        memory.set_anchor(
                            v_idx, h_idx,
                            slice_anchor(
                                tile_video_latent,
                                looping_config.negative_index_anchor_latent_frames,
                                from_start=True,
                            ),
                        )

                    # Keep audio from the centre tile only.
                    if v_idx == centre_v and h_idx == centre_h:
                        all_audio_latents.append(tile_audio_latent)

            assert segment_accumulator is not None and segment_weight_sum is not None
            # Normalize: weights are designed so the sum equals 1 in the
            # interior; we still clamp for numerical safety on the borders.
            full_segment_latent = segment_accumulator / segment_weight_sum.clamp_min(1e-6)

            # AdaIN per-segment against the assembled first segment.
            if seg_idx > 0 and looping_config.adain_factor > 0:
                full_segment_latent = adain_match(
                    full_segment_latent,
                    all_video_latents[0],
                    factor=looping_config.adain_factor,
                )

            all_video_latents.append(full_segment_latent)

        # --- assemble timeline --------------------------------------------
        full_video_latent = all_video_latents[0]
        for nxt in all_video_latents[1:]:
            full_video_latent = temporal_overlap_blend(
                full_video_latent, nxt, overlap=looping_config.overlap_latent_frames
            )
        full_audio_latent = torch.cat(all_audio_latents, dim=-1)

        logger.info(
            "[spatial-tiling] assembled latent shape video=%s audio=%s; decoding...",
            tuple(full_video_latent.shape), tuple(full_audio_latent.shape),
        )
        generator = torch.Generator(device=self.inner.device).manual_seed(seed)
        decoded_video = self.inner.video_decoder(
            full_video_latent, tiling_config, generator
        )
        decoded_audio = self.inner.audio_decoder(full_audio_latent)
        return decoded_video, decoded_audio


# ---------------------------------------------------------------------------
# Keyframe reprojection
# ---------------------------------------------------------------------------


def _reproject_keyframes(
    global_keyframes: list,
    v_px_start: int,
    v_px_end: int,
    h_px_start: int,
    h_px_end: int,
) -> list:
    """Filter global keyframes to those overlapping the given tile.

    ltx-pipelines' :class:`ImageConditioningInput` is ``(path, frame_idx,
    strength, crf)`` — it conditions on **whole frames**, not spatial sub-
    regions, so each keyframe is either fully included in a tile or skipped.
    The "reprojection" here therefore reduces to: include the same keyframe
    in every tile — and rely on ltx-pipelines' built-in image loader to
    resize the image to the tile's spatial dims (which it does in
    ``load_image_and_preprocess``).

    A future enhancement could crop the source image to the tile's bounding
    box; for now we keep parity with ComfyUI which also passes full frames
    per tile.
    """
    # All global keyframes apply to every tile — preprocessing handles resize.
    return list(global_keyframes)
