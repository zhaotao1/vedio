"""Standalone diagnostic: reproduce the 78GB VAE-decode OOM without
running 12 minutes of segment generation.

Builds JUST the VideoDecoder (same way LoopingPipeline does), feeds it a
zero latent of the exact shape and tiling_config that PID 694 OOM'd on,
and dumps a CUDA memory snapshot at the moment of failure.

Run on the A100:
    python packages/long-video-native/scripts/diag_vae_decode_oom.py
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("diag")


def gb(n: int) -> float:
    return n / 1024**3


def report(tag: str) -> None:
    log.info(
        "[mem:%s] allocated=%.2f GB reserved=%.2f GB peak=%.2f GB",
        tag,
        gb(torch.cuda.memory_allocated()),
        gb(torch.cuda.memory_reserved()),
        gb(torch.cuda.max_memory_allocated()),
    )


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Same paths as scene_jurassic_60s.yaml
    distilled = "/root/sj-tmp/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
    if not Path(distilled).exists():
        log.error("checkpoint missing: %s", distilled)
        return 1

    log.info("Importing VideoDecoder + TilingConfig…")
    from ltx_pipelines.utils.blocks import VideoDecoder  # noqa: PLC0415
    from ltx_core.model.video_vae.tiling import (  # noqa: PLC0415
        SpatialTilingConfig,
        TemporalTilingConfig,
        TilingConfig,
    )

    import os  # noqa: PLC0415
    spt = int(os.environ.get("SPT", "64"))   # spatial tile pixels (min 64, %32==0)
    tpt = int(os.environ.get("TPT", "16"))   # temporal tile frames (min 16, %8==0)
    tpo = int(os.environ.get("TPO", "8"))    # temporal overlap frames (%8==0)
    spo = int(os.environ.get("SPO", "32"))   # spatial overlap pixels (%32==0)
    # Exact config the failing run used (16/8 temporal, 384/64 spatial)
    tiling = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=spt, tile_overlap_in_pixels=spo
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=tpt, tile_overlap_in_frames=tpo
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=spt, tile_overlap_in_pixels=spo
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=tpt, tile_overlap_in_frames=tpo
        ),
    )
    log.info("tiling: %s", tiling)

    log.info("Resetting CUDA and starting memory history recording…")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=200_000)
    report("startup")

    import os  # noqa: PLC0415
    mem_eff = os.environ.get("MEM_EFF", "1") != "0"
    log.info("Building VideoDecoder (memory_efficient=%s)…", mem_eff)
    decoder_block = VideoDecoder(
        checkpoint_path=distilled,
        dtype=dtype,
        device=device,
        memory_efficient=mem_eff,
    )

    # Exact shape of the failing chunk: (1, 128, 16, 16, 24)
    # batch=1, channels=128, latent_frames=16, lat_h=16 (=128 px / 8),
    # lat_w=24 (=192 px / 8).  scene height=512 width=768 → lat_h=32 lat_w=48.
    # Let me recompute from the log: shape=(1, 128, 16, 16, 24).
    # That's height_in_pixels = 16 * VAE_spatial_downscale.
    # We can just use that exact shape; VAE will internally upscale.
    lat_shape = (1, 128, 16, 16, 24)
    log.info("Allocating fake latent on GPU shape=%s dtype=%s", lat_shape, dtype)
    latent = torch.zeros(lat_shape, device=device, dtype=dtype)
    report("after-latent")

    generator = torch.Generator(device=device).manual_seed(0)

    snapshot_path = "/tmp/diag_vae_decode.pickle"
    log.info("Calling decoder.persistent() + decode_video()…")
    try:
        with decoder_block.persistent() as decoder:
            report("after-build")
            n = 0
            for frames in decoder.decode_video(latent, tiling, generator):
                n += 1
                log.info(
                    "[decode] yielded part %d shape=%s; mem allocated=%.2f GB peak=%.2f GB",
                    n, tuple(frames.shape),
                    gb(torch.cuda.memory_allocated()),
                    gb(torch.cuda.max_memory_allocated()),
                )
                del frames
                torch.cuda.empty_cache()
            log.info("DECODE OK — %d parts yielded", n)
    except torch.OutOfMemoryError as e:
        log.error("OOM during decode_video: %s", e)
        log.error("Peak allocated=%.2f GB reserved=%.2f GB",
                  gb(torch.cuda.max_memory_allocated()),
                  gb(torch.cuda.max_memory_reserved()))
        log.error("Dumping snapshot → %s", snapshot_path)
        try:
            torch.cuda.memory._dump_snapshot(snapshot_path)
        except Exception as dump_err:
            log.error("snapshot dump failed: %s", dump_err)
        # Memory stats summary
        s = torch.cuda.memory_stats()
        log.error(
            "stats: allocations.all.peak=%s active_bytes.all.peak=%.2f GB "
            "reserved_bytes.all.peak=%.2f GB num_alloc_retries=%s",
            s.get("allocation.all.peak"),
            gb(s.get("active_bytes.all.peak", 0)),
            gb(s.get("reserved_bytes.all.peak", 0)),
            s.get("num_alloc_retries"),
        )
        traceback.print_exc()
        return 2
    except Exception:
        log.exception("non-OOM failure during decode")
        return 3
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)

    return 0


if __name__ == "__main__":
    sys.exit(main())
