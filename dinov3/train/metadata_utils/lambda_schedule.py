# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Lambda schedule for gradient/loss scaling with optional warmup."""

import math


def compute_lambda(
    iteration: int,
    max_iterations: int,
    schedule_type: str = "sigmoid",
    warmup_steps: int = 0,
) -> float:
    """Compute lambda value for gradient/loss scaling.

    Supports sigmoid, linear, and constant schedules with optional warmup.
    During warmup (iteration < warmup_steps), returns 0.0.

    Args:
        iteration: Current training iteration.
        max_iterations: Total number of training iterations.
        schedule_type: Schedule type ("sigmoid", "linear", or "constant").
        warmup_steps: Number of warmup steps before lambda starts increasing.

    Returns:
        Lambda value in [0, 1].
    """
    if iteration < warmup_steps:
        return 0.0
    effective_max = max(1, max_iterations - warmup_steps)
    progress = (iteration - warmup_steps) / effective_max
    if schedule_type == "sigmoid":
        return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    elif schedule_type == "linear":
        return progress
    return 1.0
