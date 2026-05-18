"""CLI runner for long-video-native.

Reads a YAML scene description, builds a ``LoopingPipeline`` (or wraps it
in ``SpatialTiledLoopingPipeline`` when ``spatial_tiling`` is set), and
streams the decoded video to disk via ltx-pipelines' ``encode_video``.

Example::

    ltxv-long-video --config scene.yaml

YAML schema (see ``scene.yaml.example``)::

    model:
      distilled_checkpoint_path: /models/distilled.safetensors
      spatial_upsampler_path: /models/upsampler.safetensors
      gemma_root: /models/gemma-3
      loras: []                          # list of [path, strength?]
      quantization: null                 # null | fp8-cast | fp8-scaled-mm
      offload: none                      # none | cpu | disk
      compile: false

    common:
      height: 704
      width: 1216
      frame_rate: 24.0
      seed: 42
      output_path: /out/long.mp4
      enhance_prompt: false

    looping:
      chunk_num_frames: 121
      overlap_latent_frames: 3
      overlap_strength: 0.5
      enable_negative_index: true
      negative_index_anchor_latent_frames: 2
      negative_index_strength: 0.3
      negative_frame_offset: -16
      adain_factor: 0.5

    spatial_tiling:                       # optional; omit for single-tile mode
      tile_height_px: 704
      tile_width_px: 1216
      spatial_overlap_px: 128

    prompts:                              # one per segment
      - "A woman walks into a neon-lit cafe at night"
      - "She sits at the bar, the bartender pours coffee"
      - "She takes a slow sip, steam rising"

    keyframes_per_segment:                # optional; aligned with prompts
      - []
      - [["./refs/cafe_anchor.png", 0, 0.9]]
      - []
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.utils.args import DEFAULT_IMAGE_CRF, ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import OffloadMode

from long_video_native.pipeline.looping import LoopingConfig, LoopingPipeline
from long_video_native.pipeline.spatial_tiled import (
    SpatialTiledLoopingPipeline,
    SpatialTilingConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level mapping")
    return data


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _parse_loras(spec: list[Any] | None) -> tuple[LoraPathStrengthAndSDOps, ...]:
    if not spec:
        return ()
    parsed: list[LoraPathStrengthAndSDOps] = []
    for entry in spec:
        if isinstance(entry, str):
            path, strength = entry, 1.0
        elif isinstance(entry, (list, tuple)) and len(entry) in (1, 2):
            path = entry[0]
            strength = float(entry[1]) if len(entry) == 2 else 1.0
        else:
            raise ValueError(
                f"lora entry must be a string or [path, strength?] list; got {entry!r}"
            )
        parsed.append(
            LoraPathStrengthAndSDOps(
                str(_resolve_path(path)),
                strength,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        )
    return tuple(parsed)


def _parse_quantization(
    value: str | None, checkpoint_path: str
) -> QuantizationPolicy | None:
    if value is None:
        return None
    value = value.lower()
    if value in ("none", "null", ""):
        return None
    if value == "fp8-cast":
        return QuantizationPolicy.fp8_cast()
    if value == "fp8-scaled-mm":
        return QuantizationPolicy.fp8_scaled_mm(checkpoint_path)
    raise ValueError(
        f"unknown quantization {value!r}; expected one of: "
        "none, fp8-cast, fp8-scaled-mm"
    )


def _parse_offload(value: str) -> OffloadMode:
    try:
        return OffloadMode(value.lower())
    except ValueError as e:
        raise ValueError(
            f"unknown offload mode {value!r}; expected: none, cpu, disk"
        ) from e


def _parse_keyframes(
    spec: list[list[Any]] | None,
    num_segments: int,
) -> list[list[ImageConditioningInput]]:
    if not spec:
        return [[] for _ in range(num_segments)]
    if len(spec) != num_segments:
        raise ValueError(
            f"keyframes_per_segment has {len(spec)} entries; "
            f"expected {num_segments} (one per prompt)"
        )
    out: list[list[ImageConditioningInput]] = []
    for seg_entries in spec:
        seg_keys: list[ImageConditioningInput] = []
        for kf in seg_entries or []:
            if not isinstance(kf, (list, tuple)) or len(kf) not in (3, 4):
                raise ValueError(
                    f"keyframe must be [path, frame_idx, strength] or "
                    f"[path, frame_idx, strength, crf]; got {kf!r}"
                )
            path = str(_resolve_path(kf[0]))
            frame_idx = int(kf[1])
            strength = float(kf[2])
            crf = int(kf[3]) if len(kf) == 4 else DEFAULT_IMAGE_CRF
            seg_keys.append(
                ImageConditioningInput(
                    path=path, frame_idx=frame_idx, strength=strength, crf=crf
                )
            )
        out.append(seg_keys)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ltxv-long-video",
        description="Long-video generation with temporal × spatial tiling on "
        "top of ltx-pipelines (ComfyUI LTXVLoopingSampler equivalent, "
        "pure-CLI).",
    )
    parser.add_argument(
        "--config", "-c", required=True, type=Path,
        help="Path to YAML scene config (see runner.py docstring).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    cfg = _load_yaml(args.config)
    model_cfg = cfg.get("model") or {}
    common = cfg.get("common") or {}
    looping_yaml = cfg.get("looping") or {}
    spatial_yaml = cfg.get("spatial_tiling")  # may be absent
    prompts = cfg.get("prompts") or []
    keyframes_cfg = cfg.get("keyframes_per_segment")

    if not prompts:
        raise SystemExit("config must define a non-empty 'prompts' list")

    # --- build inner pipeline ---------------------------------------------
    logger.info("Building distilled pipeline…")
    distilled_ckpt = str(_resolve_path(model_cfg["distilled_checkpoint_path"]))
    inner = LoopingPipeline(
        distilled_checkpoint_path=distilled_ckpt,
        spatial_upsampler_path=str(_resolve_path(model_cfg["spatial_upsampler_path"])),
        gemma_root=str(_resolve_path(model_cfg["gemma_root"])),
        loras=list(_parse_loras(model_cfg.get("loras"))),
        quantization=_parse_quantization(
            model_cfg.get("quantization"), distilled_ckpt
        ),
        torch_compile=bool(model_cfg.get("compile", False)),
        offload_mode=_parse_offload(model_cfg.get("offload", "none")),
    )

    # --- build configs ----------------------------------------------------
    looping_config = LoopingConfig(
        chunk_num_frames=int(looping_yaml.get("chunk_num_frames", 121)),
        overlap_latent_frames=int(looping_yaml.get("overlap_latent_frames", 3)),
        overlap_strength=float(looping_yaml.get("overlap_strength", 0.5)),
        enable_negative_index=bool(
            looping_yaml.get("enable_negative_index", False)
        ),
        negative_index_anchor_latent_frames=int(
            looping_yaml.get("negative_index_anchor_latent_frames", 2)
        ),
        negative_index_strength=float(
            looping_yaml.get("negative_index_strength", 0.3)
        ),
        negative_frame_offset=int(
            looping_yaml.get("negative_frame_offset", -16)
        ),
        adain_factor=float(looping_yaml.get("adain_factor", 0.5)),
    )

    height = int(common.get("height", 704))
    width = int(common.get("width", 1216))
    frame_rate = float(common.get("frame_rate", 24.0))
    seed = int(common.get("seed", 42))
    output_path = _resolve_path(common["output_path"])
    enhance_prompt = bool(common.get("enhance_prompt", False))

    keyframes_per_segment = _parse_keyframes(keyframes_cfg, len(prompts))

    # --- VAE tiling config (optional yaml override) -----------------------
    # The video VAE decoder is the single biggest memory hog at decode time;
    # for long videos at A100-80GB the default 80-frame temporal tile can
    # OOM during the mid-block conv3d workspace allocation. Allow the user
    # to shrink the tile sizes via yaml:
    #   vae_tiling:
    #     temporal_tile_frames: 32   # default 80, must be >=16 and %8==0
    #     temporal_overlap_frames: 8 # default 24, must be %8==0 and < tile
    #     spatial_tile_pixels: 512   # default 768, must be >=64 and %32==0
    #     spatial_overlap_pixels: 32 # default 64, must be %32==0
    vae_tiling_yaml = cfg.get("vae_tiling") or {}
    default_tiling = TilingConfig.default()
    from ltx_core.model.video_vae.tiling import (  # noqa: PLC0415
        SpatialTilingConfig as VaeSpatialTilingConfig,
    )
    from ltx_core.model.video_vae.tiling import (  # noqa: PLC0415
        TemporalTilingConfig as VaeTemporalTilingConfig,
    )

    tiling_config = TilingConfig(
        spatial_config=VaeSpatialTilingConfig(
            tile_size_in_pixels=int(
                vae_tiling_yaml.get(
                    "spatial_tile_pixels",
                    default_tiling.spatial_config.tile_size_in_pixels,
                )
            ),
            tile_overlap_in_pixels=int(
                vae_tiling_yaml.get(
                    "spatial_overlap_pixels",
                    default_tiling.spatial_config.tile_overlap_in_pixels,
                )
            ),
        ),
        temporal_config=VaeTemporalTilingConfig(
            tile_size_in_frames=int(
                vae_tiling_yaml.get(
                    "temporal_tile_frames",
                    default_tiling.temporal_config.tile_size_in_frames,
                )
            ),
            tile_overlap_in_frames=int(
                vae_tiling_yaml.get(
                    "temporal_overlap_frames",
                    default_tiling.temporal_config.tile_overlap_in_frames,
                )
            ),
        ),
    )

    # --- run --------------------------------------------------------------
    if spatial_yaml:
        spatial_config = SpatialTilingConfig(
            tile_height_px=int(spatial_yaml.get("tile_height_px", 704)),
            tile_width_px=int(spatial_yaml.get("tile_width_px", 1216)),
            spatial_overlap_px=int(spatial_yaml.get("spatial_overlap_px", 128)),
            per_tile_seed_stride=int(spatial_yaml.get("per_tile_seed_stride", 1009)),
        )
        pipeline = SpatialTiledLoopingPipeline(inner)
        video, audio = pipeline.generate_long(
            prompts=prompts,
            seed=seed,
            height=height,
            width=width,
            frame_rate=frame_rate,
            keyframes_per_segment=keyframes_per_segment,
            looping_config=looping_config,
            spatial_config=spatial_config,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
        )
    else:
        video, audio = inner.generate_long(
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

    # Total pixel frames after stitching: T0 + Σ (Ti - overlap_pixel)
    overlap_pixel = looping_config.overlap_latent_frames * 8
    total_frames = (
        looping_config.chunk_num_frames
        + (len(prompts) - 1)
        * (looping_config.chunk_num_frames - overlap_pixel)
    )
    video_chunks_number = get_video_chunks_number(total_frames, tiling_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Encoding video → %s (%d total frames)", output_path, total_frames)
    encode_video(
        video=video,
        fps=frame_rate,
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=video_chunks_number,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
