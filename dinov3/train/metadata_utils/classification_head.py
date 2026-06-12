# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Classification head for metadata-guided learning."""

import torch.nn as nn

from .gradient_scaling import GradientScalingLayer


class Classifier(nn.Module):
    """MLP classifier with gradient scaling support for GRL.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: List of hidden layer dimensions.
        num_classes: Number of output classes.
        dropout: Dropout probability (0 = no dropout).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list[int],
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gradient_scaling = GradientScalingLayer()

        use_dropout = dropout > 0.0
        layers = [
            nn.Linear(input_dim, hidden_dim[0]),
            nn.GELU(),
            nn.Dropout(dropout) if use_dropout else nn.Identity(),
        ]

        for i in range(len(hidden_dim) - 1):
            layers.extend([
                nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout) if use_dropout else nn.Identity(),
            ])

        layers.append(nn.Linear(hidden_dim[-1], num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x, lambda_: float = 0.0):
        x = self.gradient_scaling(x, lambda_)
        return self.classifier(x)
