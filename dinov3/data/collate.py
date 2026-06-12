# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import random
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import torch


def _collate_metadata(metadata_list: list) -> Any:
    """Collate a list of per-sample metadata dataclasses into one batched dataclass.

    Each field is stacked into a tensor when numeric; non-numeric fields (str,
    mixed tuples, etc.) are kept as a Python list. Returns ``None`` if the
    samples don't carry a dataclass-typed metadata payload.
    """
    first = metadata_list[0]

    if not is_dataclass(first):
        return None

    data_type = type(first)
    collated = {}

    for f in fields(first):
        vals = [getattr(m, f.name) for m in metadata_list]

        if vals[0] is None:
            collated[f.name] = None
        elif isinstance(vals[0], float):
            arr = np.array(vals, dtype=np.float32)
            collated[f.name] = torch.from_numpy(arr)
        elif isinstance(vals[0], (int, np.integer)):
            arr = np.array(vals)
            collated[f.name] = torch.from_numpy(arr).to(torch.long)
        elif isinstance(vals[0], (np.floating, np.ndarray)):
            arr = np.array(vals)
            if arr.dtype in (np.float32, np.float64):
                collated[f.name] = torch.from_numpy(arr.astype(np.float32))
            else:
                collated[f.name] = torch.from_numpy(arr).to(torch.long)
        elif isinstance(vals[0], tuple) and all(isinstance(v, (int, float)) for v in vals[0]):
            arr = np.array(vals, dtype=np.float32)
            collated[f.name] = torch.from_numpy(arr)
        else:
            collated[f.name] = vals

    return data_type(**collated)


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
):
    # Extract metadata when samples are shaped (transform_output, (target, metadata)).
    # Standard 2-tuple samples (transform_output, target) leave metadata as None.
    metadata = None
    first_sample = samples_list[0]
    if len(first_sample) > 1 and isinstance(first_sample[1], tuple) and len(first_sample[1]) > 1:
        metadata_list = [s[1][1] for s in samples_list]
        if metadata_list[0] is not None:
            metadata = _collate_metadata(metadata_list)

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack(
        [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )  # [n_global_crops, B, ...]
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    if "gram_teacher_crops" in samples_list[0][0]:
        collated_gram_teacher_crops = torch.stack(
            [s[0]["gram_teacher_crops"][i] for i in range(n_global_crops) for s in samples_list]
        )  # [n_global_crops, B, ...]
    else:
        collated_gram_teacher_crops = None

    if local_batch_size is not None:
        # multi-distillation case, number of masks is different because the number of samples masked
        # is different of the number of samples passed into the teacher initially
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        if random_circular_shift:  # apply le random circular shift to
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
        upperbound += int(N * prob_max)
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        "metadata": metadata,
    }
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)
    return out


# def get_batch_subset(collated_data_batch, target_bs):
def get_batch_subset(collated_data_batch, divide_by):
    old_bs = collated_data_batch["collated_global_crops"].shape[0] // 2
    target_bs = (old_bs + divide_by - 1) // divide_by
    collated_global_crops = (
        collated_data_batch["collated_global_crops"].unflatten(0, (2, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )
    collated_local_crops = (
        collated_data_batch["collated_local_crops"].unflatten(0, (-1, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )

    masks_old_bs = collated_data_batch["collated_masks"].shape[0] // 2
    masks_target_bs = masks_old_bs // divide_by
    collated_masks = (
        collated_data_batch["collated_masks"]
        .unflatten(0, (2, masks_old_bs))
        .narrow(1, 0, masks_target_bs)
        .flatten(0, 1)
    )
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    while mask_indices_list.shape[0] == 0:
        _unbind = list(collated_data_batch["collated_masks"].unbind(0))
        random.shuffle(_unbind)
        _bind = torch.stack(_unbind, dim=0)
        collated_masks = _bind.unflatten(0, (2, masks_old_bs)).narrow(1, 0, masks_target_bs).flatten(0, 1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    upperbound = collated_data_batch["upperbound"]

    new_batch = {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }

    if "global_batch_size" in collated_data_batch.keys():
        new_batch["global_batch_size"] = collated_data_batch["global_batch_size"] // divide_by

    metadata = collated_data_batch.get("metadata")
    if metadata is not None and is_dataclass(metadata):
        subset_metadata = {}
        for f in fields(metadata):
            val = getattr(metadata, f.name)
            if isinstance(val, torch.Tensor) and val.shape[0] == old_bs:
                subset_metadata[f.name] = val[:target_bs]
            else:
                subset_metadata[f.name] = val
        new_batch["metadata"] = type(metadata)(**subset_metadata)
    elif metadata is not None:
        new_batch["metadata"] = metadata

    return new_batch
