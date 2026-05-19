"""Unit tests for the YAML/CLI runner — no model required."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from long_video_native.runner import (
    _parse_distilled_lora,
    _parse_guider_params,
    _parse_keyframes,
    _parse_loras,
    _parse_offload,
    _parse_quantization,
)
from ltx_core.components.guiders import MultiModalGuiderParams


def test_parse_offload_modes():
    assert _parse_offload("none").value == "none"
    assert _parse_offload("CPU").value == "cpu"
    assert _parse_offload("Disk").value == "disk"
    with pytest.raises(ValueError):
        _parse_offload("bogus")


def test_parse_quantization_none():
    assert _parse_quantization(None, "/x") is None
    assert _parse_quantization("none", "/x") is None


def test_parse_quantization_fp8_cast():
    pol = _parse_quantization("fp8-cast", "/x")
    assert pol is not None


def test_parse_loras_accepts_string_or_pair(tmp_path: Path):
    f = tmp_path / "lora.safetensors"
    f.write_text("")  # only path needs to exist for _resolve_path
    out = _parse_loras([str(f), [str(f), 0.7]])
    assert len(out) == 2
    assert out[0].strength == 1.0
    assert out[1].strength == 0.7


def test_parse_distilled_lora_path(tmp_path: Path):
    f = tmp_path / "distilled-lora.safetensors"
    f.write_text("")
    out = _parse_distilled_lora({"distilled_lora_path": str(f)})
    assert len(out) == 1
    assert out[0].strength == 1.0


def test_parse_distilled_lora_required():
    with pytest.raises(ValueError, match="distilled_lora"):
        _parse_distilled_lora({})


def test_parse_guider_params_overrides_defaults():
    default = MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[28],
    )
    out = _parse_guider_params({"cfg_scale": 4.0, "stg_blocks": []}, default)
    assert out.cfg_scale == 4.0
    assert out.stg_blocks == []
    assert out.modality_scale == 3.0


def test_parse_keyframes_length_mismatch():
    with pytest.raises(ValueError, match="expected 3"):
        _parse_keyframes([[], []], num_segments=3)


def test_parse_keyframes_per_segment(tmp_path: Path):
    img = tmp_path / "anchor.png"
    img.write_text("")
    out = _parse_keyframes([[[str(img), 0, 0.9]], []], num_segments=2)
    assert len(out) == 2
    assert out[0][0].frame_idx == 0
    assert out[0][0].strength == 0.9


def test_yaml_schema_round_trip(tmp_path: Path):
    """Sanity-check the example YAML parses without crashing."""
    cfg = yaml.safe_load(
        (Path(__file__).parent.parent / "scene.yaml.example").read_text()
    )
    assert "prompts" in cfg
    assert "looping" in cfg
    assert "model" in cfg
