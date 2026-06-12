# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Gradient scaling layer for Gradient Reversal Layer (GRL) and lambda-based gradient control."""

import torch.nn as nn
from torch.autograd import Function


class GradientScalingFunction(Function):
    """Custom autograd function that scales gradients during backprop."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.lambda_, None


class GradientScalingLayer(nn.Module):
    """Layer that scales gradients by a configurable lambda value.

    For GRL: use negative lambda to reverse gradients.
    For warmup: use lambda in [0, 1] to gradually enable gradients.
    """

    def __init__(self):
        super().__init__()
        self.lambda_ = 0.0

    def forward(self, x, lambda_: float | None = None):
        if lambda_ is not None:
            self.lambda_ = lambda_
        return GradientScalingFunction.apply(x, self.lambda_)
