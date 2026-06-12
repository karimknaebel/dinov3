# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image


class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        f = BytesIO(self._image_data)
        return Image.open(f).convert(mode="RGB")


class PackedXChannelImageDecoder(Decoder):
    """Decoder for multi-channel images packed Fortran-style along the width of a square JPG.

    The on-disk file is a square (H, H) image whose C channels were concatenated
    along the width axis with Fortran ordering. Reshape recovers (H, H, C); a
    permute then yields (C, H, H). Used by the HPA datasets where the four
    fluorescence channels are stored as one tall JPG.
    """

    def __init__(self, image_data: bytes, num_channels: int = 4) -> None:
        self._image_data = image_data
        self._num_channels = num_channels

    def decode(self) -> torch.Tensor:
        im = np.asarray(Image.open(BytesIO(self._image_data)))
        im2 = np.reshape(im, (im.shape[0], im.shape[0], -1), order="F")
        tensor = torch.from_numpy(im2).permute(2, 0, 1).contiguous().float()
        return tensor[: self._num_channels, :, :]


class TargetDecoder(Decoder):
    def __init__(self, target: Any):
        self._target = target

    def decode(self) -> Any:
        return self._target


class DenseTargetDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        f = BytesIO(self._image_data)
        return Image.open(f)
