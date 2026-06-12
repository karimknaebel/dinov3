# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Expand a 3-channel pretrained checkpoint to N channels.

The new channels are filled with the mean of the original 3 RGB filters, then
the whole ``patch_embed.proj.weight`` is multiplied by ``3 / N`` so the per-patch
activation magnitude is preserved at initialization. This is the recipe used
to bootstrap multi-channel (e.g. 4-channel HPA) runs from the public DINOv3
LVD-1689M ViT-L teacher checkpoint.

Example:

    python -m dinov3.utils.expand_patch_embed \\
        --input  dinov3_vitl16_pretrain_lvd1689m_teacher.pth \\
        --output dinov3_vitl16_pretrain_lvd1689m_teacher_4ch.pth \\
        --in-chans 4
"""

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger("dinov3")

_PATCH_EMBED_KEY = "backbone.patch_embed.proj.weight"


def expand_patch_embed(weight: torch.Tensor, target_in_chans: int, normalize: bool = True) -> torch.Tensor:
    """Inflate a ``[out, C_old, kh, kw]`` patch-embed weight to ``[out, target_in_chans, kh, kw]``.

    The extra channels are filled with the mean of the existing channels.
    When ``normalize`` is True the result is multiplied by ``C_old / target_in_chans``
    so the expected per-patch activation magnitude matches the original.
    """
    if weight.ndim != 4:
        raise ValueError(f"Expected 4D patch_embed weight, got shape {tuple(weight.shape)}")
    c_old = weight.shape[1]
    if target_in_chans == c_old:
        return weight
    if target_in_chans < c_old:
        raise ValueError(
            f"Refusing to shrink patch_embed: target_in_chans={target_in_chans} < current={c_old}. "
            "Use a channel-subset utility instead."
        )
    mean_filter = weight.mean(dim=1, keepdim=True)
    extra = mean_filter.expand(-1, target_in_chans - c_old, -1, -1).contiguous()
    expanded = torch.cat([weight, extra], dim=1)
    if normalize:
        expanded = expanded * (c_old / target_in_chans)
    return expanded


def _expand_state_dict(state_dict: dict, target_in_chans: int, normalize: bool) -> dict:
    if _PATCH_EMBED_KEY not in state_dict:
        raise KeyError(
            f"{_PATCH_EMBED_KEY!r} not found in state dict. "
            f"Top-level keys: {sorted(state_dict)[:8]}{'...' if len(state_dict) > 8 else ''}"
        )
    original = state_dict[_PATCH_EMBED_KEY]
    expanded = expand_patch_embed(original, target_in_chans, normalize=normalize)
    state_dict[_PATCH_EMBED_KEY] = expanded
    logger.info(
        "Expanded %s: %s -> %s (normalize=%s)",
        _PATCH_EMBED_KEY,
        tuple(original.shape),
        tuple(expanded.shape),
        normalize,
    )
    return state_dict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Path to the source .pth checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the expanded checkpoint to")
    parser.add_argument("--in-chans", type=int, required=True, help="Target number of input channels (must be > current)")
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        default="teacher",
        help="Top-level key holding the model state dict (default: 'teacher'). "
        "Pass empty string if the file is the state dict itself.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip the C_old / C_new rescaling. Off by default — keep on unless you know what you are doing.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Loading %s", args.input)
    checkpoint = torch.load(args.input, map_location="cpu")

    key = args.checkpoint_key or None
    if key is not None:
        if key not in checkpoint:
            raise KeyError(
                f"Checkpoint key {key!r} not found at top level. "
                f"Available: {sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint)}"
            )
        checkpoint[key] = _expand_state_dict(checkpoint[key], args.in_chans, normalize=not args.no_normalize)
    else:
        checkpoint = _expand_state_dict(checkpoint, args.in_chans, normalize=not args.no_normalize)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
