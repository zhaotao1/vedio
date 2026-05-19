"""Disk loaders for optional reference latents (P5 of the alignment plan).

Supports two file formats:

* ``.pt`` / ``.pth`` — ``torch.save`` of a 5-D tensor ``[B, C, T, H, W]``
  (or any tensor that is reshape-equivalent).
* ``.safetensors`` — single tensor under the key ``"latent"``.

Used for ``normalizing_latents_path`` (AdaIN reference) and
``negative_index.external_latents_path`` (long-term memory anchor).
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_reference_latent(path: str | Path) -> torch.Tensor:
    """Load a 5-D ``[B, C, T, H, W]`` latent tensor from disk.

    Files are loaded onto CPU; the caller is responsible for moving the
    tensor to the active device.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"reference latent not found: {p}")

    suffix = p.suffix.lower()
    if suffix in (".pt", ".pth"):
        obj = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            if "latent" in obj:
                obj = obj["latent"]
            elif "samples" in obj:
                obj = obj["samples"]
            else:
                raise ValueError(
                    f"{p} is a dict but has neither 'latent' nor 'samples' "
                    f"key; got {list(obj.keys())}"
                )
        if not isinstance(obj, torch.Tensor):
            raise TypeError(f"{p} did not load to a tensor; got {type(obj)}")
    elif suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "safetensors is required to load .safetensors files; "
                "pip install safetensors"
            ) from e
        loaded = load_file(str(p), device="cpu")
        if "latent" not in loaded:
            raise ValueError(
                f"{p} must contain a tensor under key 'latent'; got "
                f"keys {list(loaded.keys())}"
            )
        obj = loaded["latent"]
    else:
        raise ValueError(
            f"unsupported latent file format {suffix!r}; expected .pt, .pth, "
            ".safetensors"
        )

    if obj.dim() != 5:
        raise ValueError(
            f"reference latent must be 5-D [B, C, T, H, W]; got shape "
            f"{tuple(obj.shape)}"
        )
    return obj.detach().contiguous()
