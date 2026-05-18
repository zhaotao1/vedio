"""Smoke test: 1 segment, 41 frames @ 320x512, no LoRA, no tiling.

Verifies that LoopingPipeline.generate_one_segment runs end-to-end and
returns the expected latent shapes. Requires real checkpoints and a GPU
(or MPS / CPU if you have a lot of patience).

Run with::

    LTXV_DISTILLED=/path/to/distilled.safetensors \
    LTXV_UPSAMPLER=/path/to/upsampler.safetensors \
    LTXV_GEMMA=/path/to/gemma-3 \
    pytest tests/test_smoke_segment.py -s
"""

from __future__ import annotations

import os

import pytest
import torch

from long_video_native.pipeline.looping import LoopingConfig, LoopingPipeline

_DISTILLED = os.environ.get("LTXV_DISTILLED")
_UPSAMPLER = os.environ.get("LTXV_UPSAMPLER")
_GEMMA = os.environ.get("LTXV_GEMMA")

pytestmark = pytest.mark.skipif(
    not all([_DISTILLED, _UPSAMPLER, _GEMMA]),
    reason="set LTXV_DISTILLED / LTXV_UPSAMPLER / LTXV_GEMMA to run.",
)


@pytest.fixture(scope="module")
def pipeline() -> LoopingPipeline:
    return LoopingPipeline(
        distilled_checkpoint_path=_DISTILLED,
        spatial_upsampler_path=_UPSAMPLER,
        gemma_root=_GEMMA,
        loras=[],
    )


def test_single_segment_runs(pipeline: LoopingPipeline) -> None:
    cfg = LoopingConfig(
        chunk_num_frames=41,           # 8*5+1 — smallest reasonable
        overlap_latent_frames=0,
        enable_negative_index=False,
        adain_factor=0.0,
    )
    video_lat, audio_lat = pipeline.generate_one_segment(
        prompt="a cat walks across a sunny rug",
        seed=0,
        height=320,
        width=512,
        num_frames=41,
        frame_rate=24.0,
        images=[],
        prev_tail_latent=None,
        anchor_latent=None,
        config=cfg,
    )
    # Stage-2 latent: T = (41-1)//8 + 1 = 6 ; H = 320//32 = 10 ; W = 512//32 = 16
    assert video_lat.shape[2] == 6
    assert video_lat.shape[3] == 10
    assert video_lat.shape[4] == 16
    assert audio_lat.ndim == 3
