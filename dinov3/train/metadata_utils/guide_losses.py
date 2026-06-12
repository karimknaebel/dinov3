# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Loss computation utilities for metadata-guided learning heads."""

import torch
from torch import Tensor, nn


def compute_classification_loss(
    head: nn.Module,
    loss_fn: nn.Module,
    cls_input: Tensor,
    labels: Tensor,
    lambda_value: float,
    use_bce: bool = False,
) -> tuple[Tensor, float]:
    """Compute classification loss and accuracy over all crops.

    Args:
        head: Classification head module (expects forward(x, lambda_)).
        loss_fn: Loss function (CrossEntropyLoss or BCEWithLogitsLoss).
        cls_input: CLS token embeddings [n_crops, B, D].
        labels: Ground truth labels [B].
        lambda_value: Gradient scaling lambda.
        use_bce: If True, cast labels to float for BCE loss.

    Returns:
        Tuple of (loss, accuracy).
    """
    outputs = []
    for crop_cls in cls_input:
        outputs.append(head(crop_cls, lambda_value))

    if use_bce:
        labels_float = labels.float()
        loss = sum(loss_fn(out.float(), labels_float) for out in outputs) / len(outputs)
        with torch.no_grad():
            pred = (torch.sigmoid(outputs[0].float()) > 0.5).float()
            accuracy = ((pred == labels_float).sum(dim=1) == labels.shape[1]).float().mean().item()
    else:
        loss = sum(loss_fn(out.float(), labels) for out in outputs) / len(outputs)
        with torch.no_grad():
            predictions = outputs[0].argmax(dim=1)
            accuracy = (predictions == labels.squeeze()).float().mean().item()

    return loss, accuracy


def compute_regression_loss(
    head: nn.Module,
    loss_fn: nn.Module,
    cls_input: Tensor,
    labels: Tensor,
    lambda_value: float,
) -> tuple[Tensor, float]:
    """Compute regression loss and MSE metric over all crops.

    Args:
        head: Regression head module (expects forward(x, lambda_)).
        loss_fn: Loss function (e.g. MSELoss).
        cls_input: CLS token embeddings [n_crops, B, D].
        labels: Ground truth labels [B] or [B, n_outputs].
        lambda_value: Gradient scaling lambda.

    Returns:
        Tuple of (loss, mse).
    """
    labels_float = labels.float() if labels.dtype != torch.float32 else labels
    if labels_float.dim() == 1:
        labels_float = labels_float.unsqueeze(-1)

    reg_module = head.module if hasattr(head, "module") else head
    if hasattr(reg_module, "normalize_targets"):
        labels_float = reg_module.normalize_targets(labels_float)

    outputs = []
    for crop_cls in cls_input:
        outputs.append(head(crop_cls, lambda_value))

    loss = sum(loss_fn(out.float(), labels_float) for out in outputs) / len(outputs)

    with torch.no_grad():
        mse = loss_fn(outputs[0].float(), labels_float).item()

    return loss, mse


def compute_prototypical_loss(
    head: nn.Module,
    cls_input: Tensor,
    labels: Tensor,
    lambda_value: float,
    teacher_cls_input: Tensor | None = None,
) -> tuple[Tensor, float]:
    """Compute prototypical contrastive loss over all crops, then update centroids.

    Args:
        head: PrototypicalContrastiveHead instance.
        cls_input: Student CLS embeddings [n_crops, B, D].
        labels: Class labels [B].
        lambda_value: Gradient scaling lambda.
        teacher_cls_input: Teacher CLS embeddings [n_crops, B, D] for centroid updates.

    Returns:
        Tuple of (loss, accuracy).
    """
    losses, accuracies = [], []
    for crop_cls in cls_input:
        loss, acc = head(crop_cls, labels, lambda_value)
        losses.append(loss)
        accuracies.append(acc)

    if teacher_cls_input is not None:
        for crop_teacher in teacher_cls_input:
            head.update_centroids_and_concentration(crop_teacher, labels)

    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)
