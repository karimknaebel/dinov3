# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from collections import defaultdict

logger = logging.getLogger("dinov3")


def get_vit_lr_decay_rate(
    name,
    lr_decay_rate=1.0,
    num_layers=12,
    force_is_backbone=False,
    chunked_blocks=False,
):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".storage_tokens" in name
        ):
            layer_id = 0
        elif force_is_backbone and (
            "pos_embed" in name
            or "patch_embed" in name
            or "mask_token" in name
            or "cls_token" in name
            or "storage_tokens" in name
        ):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        elif chunked_blocks and "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
        elif "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay(model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0, dino_head_wd_multiplier=1.0):
    chunked_blocks = False
    if hasattr(model, "n_blocks"):
        logger.info("chunked fsdp")
        n_blocks = model.n_blocks
        chunked_blocks = model.chunked_blocks
    elif hasattr(model, "blocks"):
        logger.info("first code branch")
        n_blocks = len(model.blocks)
    elif hasattr(model, "backbone"):
        logger.info("second code branch")
        n_blocks = len(model.backbone.blocks)
    else:
        logger.info("else code branch")
        n_blocks = 0
    all_param_groups = []

    for name, param in model.named_parameters():
        name = remove_fsdp_compile_names(name)
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(
            name,
            lr_decay_rate,
            num_layers=n_blocks,
            force_is_backbone=n_blocks > 0,
            chunked_blocks=chunked_blocks,
        )
        d = {
            "name": name,
            "params": param,
            "is_last_layer": False,
            "lr_multiplier": decay_rate,
            "wd_multiplier": 1.0,
        }

        if "dino_head" in name:
            d["wd_multiplier"] = dino_head_wd_multiplier

        if "last_layer" in name:
            d["is_last_layer"] = True

        # No weight-decay on biases, norm parameters, layer scale gamma, learned tokens and embeddings
        if name.endswith("bias") or "norm" in name or "gamma" in name or "fourier_w" in name:
            d["wd_multiplier"] = 0.0

        if "patch_embed" in name:
            d["lr_multiplier"] *= patch_embed_lr_mult

        all_param_groups.append(d)
        logger.info(f"{name}: lr_multiplier: {d['lr_multiplier']}, wd_multiplier: {d['wd_multiplier']}")

    return all_param_groups


def fuse_params_groups(
    all_params_groups,
    keys=("lr_multiplier", "wd_multiplier", "is_last_layer", "is_frozen_backbone", "always_frozen"),
):
    fused_params_groups = defaultdict(lambda: {"params": []})
    for d in all_params_groups:
        identifier = ""
        for k in keys:
            identifier += k + str(d.get(k, False)) + "_"

        for k in keys:
            fused_params_groups[identifier][k] = d.get(k, False)
        fused_params_groups[identifier]["params"].append(d["params"])

    return fused_params_groups.values()


def get_params_groups_with_decay_fsdp(
    model,
    lr_decay_rate=1.0,
    patch_embed_lr_mult=1.0,
    dino_head_wd_multiplier=1.0,
    freeze_patch_embed=True,
    freeze_cls_token=True,
    freeze_mask_token=True,
    unfreeze_last_n_blocks=0,
):
    """Build per-parameter groups for an FSDP-wrapped submodule.

    The ``freeze_*`` and ``unfreeze_last_n_blocks`` knobs annotate backbone
    params with ``is_frozen_backbone`` and ``always_frozen`` flags, which the
    training loop honors via ``apply_optim_scheduler`` to implement two-stage
    finetuning. Defaults preserve the no-freeze case so callers that don't
    pass these knobs behave identically to before.
    """
    if hasattr(model, "module"):  # SimpleFSDP
        is_backbone = hasattr(model.module, "blocks")
        n_blocks = len(model.module.blocks) if is_backbone else 0
    else:  # FSDP2
        is_backbone = hasattr(model, "blocks")
        n_blocks = len(model.blocks) if is_backbone else 0

    all_param_groups = []

    for name, param in model.named_parameters():
        name = remove_fsdp_compile_names(name)
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(
            name,
            lr_decay_rate,
            num_layers=n_blocks,
            force_is_backbone=n_blocks > 0,
            chunked_blocks=False,
        )
        d = {
            "name": name,
            "params": param,
            "is_last_layer": False,
            "is_frozen_backbone": False,
            "always_frozen": False,
            "lr_multiplier": decay_rate,
            "wd_multiplier": 1.0,
        }

        if "dino_head" in name:
            d["wd_multiplier"] = dino_head_wd_multiplier

        if "last_layer" in name:
            d["is_last_layer"] = True

        # No weight-decay on biases, norm parameters, layer scale gamma, learned tokens and embeddings
        if name.endswith("bias") or "norm" in name or "gamma" in name or "fourier_w" in name:
            d["wd_multiplier"] = 0.0

        if "patch_embed" in name:
            d["lr_multiplier"] *= patch_embed_lr_mult

        # Two-stage finetune flags only apply to backbone submodules; non-backbone
        # submodules (dino_head, guide heads) leave both flags False.
        if is_backbone:
            is_patch_embed = "patch_embed" in name
            is_cls_token = name.endswith("cls_token")
            is_mask_token = name.endswith("mask_token")
            gated_off = (
                (is_patch_embed and not freeze_patch_embed)
                or (is_cls_token and not freeze_cls_token)
                or (is_mask_token and not freeze_mask_token)
            )
            if not gated_off:
                d["is_frozen_backbone"] = True

            if unfreeze_last_n_blocks > 0:
                if "blocks." in name:
                    block_idx = int(name.split("blocks.")[1].split(".")[0])
                    if block_idx < n_blocks - unfreeze_last_n_blocks:
                        d["always_frozen"] = True
                else:
                    d["always_frozen"] = True

        all_param_groups.append(d)
        logger.info(
            f"{name}: lr_multiplier: {d['lr_multiplier']}, wd_multiplier: {d['wd_multiplier']}, "
            f"is_frozen_backbone: {d['is_frozen_backbone']}, always_frozen: {d['always_frozen']}"
        )

    return all_param_groups


def remove_fsdp_compile_names(name: str):
    name = name.replace("_fsdp_wrapped_module.", "")  # Added by FSDP
    name = name.replace("_checkpoint_wrapped_module.", "")  # Added by activation checkpointing for xFSDP
    name = name.replace("parametrizations.", "")  # Added by xFSDP
    name = name.removesuffix(".original")  # Added by xFSDP
    name = name.replace("module.", "")  # Added by xFSDP
    name = name.replace("_orig_mod.", "")  # Added by torch.compile
    return name
