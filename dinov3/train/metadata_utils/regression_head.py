# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Regression head for metadata-guided learning."""

import torch
import torch.nn as nn

from .gradient_scaling import GradientScalingLayer


class Regressor(nn.Module):
    """MLP regressor with gradient scaling support for GRL.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: List of hidden layer dimensions.
        n_outputs: Number of regression outputs.
        dropout: Dropout probability (0 = no dropout).
        output_activation: Output activation ("none", "sigmoid", "tanh").
        output_min: Tuple of min values for target normalization.
        output_max: Tuple of max values for target normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list[int],
        n_outputs: int = 1,
        dropout: float = 0.0,
        output_activation: str = "none",
        output_min: tuple[float, ...] | None = None,
        output_max: tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.gradient_scaling = GradientScalingLayer()
        self.n_outputs = n_outputs
        self.output_activation = output_activation

        self._output_min_values = output_min
        self._output_max_values = output_max

        if output_min is not None:
            self.register_buffer("output_min", torch.tensor(output_min, dtype=torch.float32))
            self.register_buffer("output_max", torch.tensor(output_max, dtype=torch.float32))
        else:
            self.output_min = None
            self.output_max = None

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

        layers.append(nn.Linear(hidden_dim[-1], n_outputs))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x, lambda_: float = 0.0):
        x = self.gradient_scaling(x, lambda_)
        out = self.regressor(x)

        if self.output_activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.output_activation == "tanh":
            out = torch.tanh(out)

        return out

    def reset_buffers(self) -> None:
        """Restore output_min/output_max buffers from config values (after NaN init)."""
        if self._output_min_values is not None:
            self.output_min.copy_(torch.tensor(self._output_min_values, dtype=torch.float32))
            self.output_max.copy_(torch.tensor(self._output_max_values, dtype=torch.float32))

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize targets to [0, 1] range using configured min/max."""
        if self.output_min is None:
            return targets
        return (targets - self.output_min) / (self.output_max - self.output_min + 1e-8)

    def denormalize_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Denormalize outputs from [0, 1] to original range."""
        if self.output_min is None:
            return outputs
        return outputs * (self.output_max - self.output_min) + self.output_min
