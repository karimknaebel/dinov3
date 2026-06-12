# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Spectral Independence Gradient Regularization (SIGReg) loss.

Regularizes representations toward a standard normal distribution using
empirical characteristic functions with distributed synchronization.
"""

import torch
import torch.distributed as dist
import torch.nn as nn


class DistributedSIGReg(nn.Module):
    """SIGReg loss with distributed all-reduce for multi-GPU training.

    Computes the L2 distance between the empirical characteristic function (ECF)
    of random projections of the input and the target (standard normal)
    characteristic function, integrated via the trapezoidal rule.

    Args:
        num_slices: Number of random projections.
        range_max: Grid range for characteristic function evaluation.
        n_knots: Number of grid points for integration.
    """

    def __init__(self, num_slices: int = 1024, range_max: float = 5.0, n_knots: int = 17):
        super().__init__()
        self.num_slices = num_slices
        self.range_max = range_max
        self.n_knots = n_knots

        t = torch.linspace(-range_max, range_max, n_knots)
        weights = torch.exp(-0.5 * t.square())

        self.register_buffer("t", t)
        self.register_buffer("weights", weights)
        self.register_buffer("target_phi", weights.clone())

    def reset_buffers(self) -> None:
        t = torch.linspace(-self.range_max, self.range_max, self.n_knots, device=self.t.device)
        weights = torch.exp(-0.5 * t.square())
        self.t.copy_(t)
        self.weights.copy_(weights)
        self.target_phi.copy_(weights)

    def forward(self, z: torch.Tensor, seed_step: int | None = None) -> torch.Tensor:
        N, D = z.shape
        device = z.device

        generator = torch.Generator(device=device)
        if seed_step is not None:
            generator.manual_seed(int(seed_step))

        A = torch.randn(D, self.num_slices, device=device, generator=generator, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)

        projections = z @ A

        arg = projections.unsqueeze(-1) * self.t.view(1, 1, -1)
        local_cos = torch.cos(arg).mean(dim=0)
        local_sin = torch.sin(arg).mean(dim=0)

        stats = torch.stack([local_cos, local_sin])

        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            stats /= dist.get_world_size()

        ecf_real, ecf_imag = stats[0], stats[1]

        diff_real = ecf_real - self.target_phi.view(1, -1)
        loss_unreduced = (diff_real.square() + ecf_imag.square()) * self.weights.view(1, -1)

        return torch.trapezoid(loss_unreduced, self.t, dim=-1).mean()
