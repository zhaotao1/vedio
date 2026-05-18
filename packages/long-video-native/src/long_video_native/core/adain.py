"""Latent-space AdaIN colour alignment.

Used to suppress slow colour / brightness drift across long-video segments.
We operate on the VAE latent (5-D ``[B, C, T, H, W]``) rather than pixels
because:

1. it lets us realign before VAE decode, saving an encode round-trip;
2. LTX-2's VAE is approximately linear in colour-related channels, so latent
   statistics carry the same first-order chromatic signal as pixel statistics;
3. it mirrors ComfyUI's ``_apply_adain`` which also runs in latent space.

The realignment matches per-channel mean / std of ``target`` to ``reference``,
weighted by ``factor`` in ``[0, 1]``.
"""

from __future__ import annotations

import torch


def adain_match(
    target: torch.Tensor,
    reference: torch.Tensor,
    factor: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match per-channel mean/std of ``target`` to ``reference``.

    Args:
        target: ``[B, C, T, H, W]`` latent to be adjusted.
        reference: ``[B, C, T_ref, H, W]`` latent providing target statistics.
            ``T_ref`` may differ from ``T``.
        factor: 0.0 returns ``target`` unchanged; 1.0 applies full match.
        eps: numerical floor for std.

    Returns:
        New tensor with the same shape as ``target``.
    """
    if factor <= 0.0:
        return target

    # Compute statistics over (T, H, W), keeping (B, C) — exactly the axes
    # that "describe a frame" — so colour shifts are corrected without
    # collapsing spatial variation.
    ref_mean = reference.mean(dim=(2, 3, 4), keepdim=True)
    ref_std = reference.std(dim=(2, 3, 4), keepdim=True).clamp_min(eps)
    tgt_mean = target.mean(dim=(2, 3, 4), keepdim=True)
    tgt_std = target.std(dim=(2, 3, 4), keepdim=True).clamp_min(eps)

    matched = (target - tgt_mean) / tgt_std * ref_std + ref_mean
    if factor >= 1.0:
        return matched
    return target + factor * (matched - target)
