# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Metadata utilities for guided DINO training."""

from .classification_head import Classifier
from .gradient_scaling import GradientScalingFunction, GradientScalingLayer
from .guide_losses import (
    compute_classification_loss,
    compute_prototypical_loss,
    compute_regression_loss,
)
from .lambda_schedule import compute_lambda
from .prototypical_head import PrototypicalContrastiveHead
from .regression_head import Regressor

__all__ = [
    "Classifier",
    "GradientScalingFunction",
    "GradientScalingLayer",
    "PrototypicalContrastiveHead",
    "Regressor",
    "compute_classification_loss",
    "compute_lambda",
    "compute_prototypical_loss",
    "compute_regression_loss",
]
