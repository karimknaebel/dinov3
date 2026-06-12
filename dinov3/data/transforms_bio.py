# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Augmentation primitives for biological multi-channel imagery (HPA).

Used by the SSL crop pipeline when ``train.cell_augmentation_type == "hpa"``
and by the eval transforms when ``transform.use_bio_transform == True``. All
modules operate on (C, H, W) tensors with C >= 1; the channel-aware variants
assume HPA's 4-channel layout where channel index 1 is the protein stain.
"""

import numpy as np
import torch
from torchvision import transforms


class Div255(torch.nn.Module):
    """uint8 tensor / PIL → float in [0, 1]."""

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            return transforms.functional.to_tensor(x)
        return x / 255


class SelfNormalizeNoDiv(torch.nn.Module):
    """Per-image, per-channel z-score. Assumes the input is already in [0, 1]."""

    def forward(self, x):
        m = x.mean((-2, -1), keepdim=True)
        s = x.std((-2, -1), unbiased=False, keepdim=True)
        x = x - m
        x = x / (s + 1e-7)
        return x


class RandomContrast(torch.nn.Module):
    """Per-channel gaussian contrast jitter (factor ~ N(1, p), clipped at 0.5)."""

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def forward(self, img):
        if img.max() == 0:
            return img
        for c in range(img.shape[0]):
            factor = max(np.random.normal(1, self.p), 0.5)
            img[c] = transforms.functional.adjust_contrast(img[c][None, ...], factor)
        return img


class RandomBrightness(torch.nn.Module):
    """Per-channel gaussian brightness jitter (factor ~ N(1, p), clipped at 0.5)."""

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def forward(self, img):
        if img.max() == 0:
            return img
        for c in range(img.shape[0]):
            factor = max(np.random.normal(1, self.p), 0.5)
            img[c] = transforms.functional.adjust_brightness(img[c], factor)
        return img


class RandomRemoveChannelExceptProtein(torch.nn.Module):
    """With probability ``p``, zero out one of the non-protein channels (0, 2, or 3).

    Only applies when the input has at least 4 channels — pass-through otherwise.
    """

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def forward(self, img):
        if img.shape[0] < 4:
            return img
        if np.random.rand() <= self.p:
            channel = int(np.random.choice(np.array([0, 2, 3])))
            img[channel] = torch.zeros(1, *img.shape[1:])
        return img


class RandomContrastProteinChannel(torch.nn.Module):
    """With probability ``p``, rescale the protein channel (idx 1) by a random factor."""

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def forward(self, img):
        if img.max() == 0:
            return img
        if np.random.rand() <= self.p:
            random_factor = (np.random.rand() * 2) / img.max()
            img[1] = img[1] * random_factor
        return img
