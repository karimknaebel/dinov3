# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Pack per-channel HPA images into the 4-channel layout :class:`HPAWholeHR` consumes.

The upstream per-channel files are four single-channel images per cell —
``<id>_red.<ext>``, ``<id>_green.<ext>``, ``<id>_blue.<ext>``, ``<id>_yellow.<ext>``.
For external images, ``ext = jpg`` (the proteinatlas.org JPGs fetched by
``hpa_whole_hr_download``). For kaggle full-size images, ``ext = tif`` (the
TIFFs unpacked from kaggle's ``{train,test}_full_size.7z``). The downsampled
kaggle PNGs (``train.zip`` / ``test.zip`` in the same competition) use
``ext = png``.

This script:

1. Resizes each channel to ``--target-size`` (default 768) with anti-aliased
   bilinear interpolation.
2. Reduces each channel to single-channel grayscale using a channel-specific
   rule (red/green/blue pull their own RGB plane; yellow averages R+G).
3. Concatenates the four grayscales horizontally as ``[red | green | blue |
   yellow]`` and saves a single grayscale JPG ``<id>.jpg``.

The resulting JPG is shape ``(H, W*4)``; :class:`PackedXChannelImageDecoder`
reshapes it back to ``(4, H, H)`` Fortran-style at load time. The HPA channels
correspond to: red = microtubules, green = protein of interest, blue = nucleus,
yellow = endoplasmic reticulum.

GPU mode runs sequentially on a single CUDA device; CPU mode uses a process
pool. Both produce bit-identical output up to interpolation rounding.

Usage::

    # external (proteinatlas.org JPGs from hpa_whole_hr_download)
    python -m dinov3.data.datasets.hpa_whole_hr_pack_channels \\
        --input-dir <PATH/TO/per_channel_jpgs/> \\
        --output-dir <root>/HPAexternal/jpg_768x768_4channels/ \\
        [--use-gpu] [--num-workers 16] [--target-size 768]

    # kaggle full-size TIFFs (after extracting train_full_size.7z / test_full_size.7z)
    python -m dinov3.data.datasets.hpa_whole_hr_pack_channels \\
        --input-dir <PATH/TO/train_full_size/> \\
        --output-dir <root>/HPAImageKaggle/jpg_768_train/ \\
        --extension tif
    python -m dinov3.data.datasets.hpa_whole_hr_pack_channels \\
        --input-dir <PATH/TO/test_full_size/> \\
        --output-dir <root>/HPAImageKaggle/jpg_768_test/ \\
        --extension tif
"""

import argparse
import os
import warnings
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

CHANNELS = ["red", "green", "blue", "yellow"]
JPEG_QUALITY = 100


def get_image_ids(input_dir: str, extension: str):
    """Extract unique image IDs from filenames like '<id>_red.<ext>'."""
    ext = extension.lstrip(".")
    ids = set()
    for f in os.listdir(input_dir):
        if not f.endswith(f".{ext}"):
            continue
        for channel in CHANNELS:
            suffix = f"_{channel}.{ext}"
            if f.endswith(suffix):
                ids.add(f[: -len(suffix)])
                break
    return sorted(ids)


def _grayscale(resized, channel_name):
    if resized.shape[-3] < 3:
        return resized[..., 0:1, :, :]
    if channel_name == "red":
        return resized[..., 0:1, :, :]
    if channel_name == "green":
        return resized[..., 1:2, :, :]
    if channel_name == "blue":
        return resized[..., 2:3, :, :]
    if channel_name == "yellow":
        return (resized[..., 0:1, :, :] + resized[..., 1:2, :, :]) / 2.0
    return 0.299 * resized[..., 0:1, :, :] + 0.587 * resized[..., 1:2, :, :] + 0.114 * resized[..., 2:3, :, :]


def process_single_image_gpu(args):
    image_id, input_dir, output_dir, target_size, extension, use_gpu = args
    try:
        channel_images = []
        for channel_name in CHANNELS:
            filepath = os.path.join(input_dir, f"{image_id}_{channel_name}.{extension}")
            if not os.path.exists(filepath):
                return (image_id, False, f"Missing {channel_name} channel")

            img = np.array(Image.open(filepath))
            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            scale = 65535.0 if img.dtype == np.uint16 else 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / scale
            if use_gpu and torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            resized = F.interpolate(
                img_tensor, size=(target_size, target_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
            gray = _grayscale(resized, channel_name)
            gray_np = (gray.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            channel_images.append(gray_np)

        img_concatenated = np.concatenate(channel_images, axis=1)
        output_path = os.path.join(output_dir, f"{image_id}.jpg")
        Image.fromarray(img_concatenated, mode="L").save(output_path, format="JPEG", quality=JPEG_QUALITY)
        return (image_id, True, None)
    except Exception as e:
        return (image_id, False, str(e))


def process_single_image_cpu(args):
    image_id, input_dir, output_dir, target_size, extension, _ = args
    try:
        import skimage.transform

        channel_images = []
        for channel_name in CHANNELS:
            filepath = os.path.join(input_dir, f"{image_id}_{channel_name}.{extension}")
            if not os.path.exists(filepath):
                return (image_id, False, f"Missing {channel_name} channel")

            raw = np.array(Image.open(filepath))
            scale = 65535.0 if raw.dtype == np.uint16 else 255.0
            img = raw.astype(np.float32) / scale
            img_resized = skimage.transform.resize(img, (target_size, target_size), anti_aliasing=True)

            if img_resized.ndim == 2:
                gray = img_resized
            elif img_resized.shape[2] >= 3:
                if channel_name == "red":
                    gray = img_resized[:, :, 0]
                elif channel_name == "green":
                    gray = img_resized[:, :, 1]
                elif channel_name == "blue":
                    gray = img_resized[:, :, 2]
                elif channel_name == "yellow":
                    gray = (img_resized[:, :, 0] + img_resized[:, :, 1]) / 2.0
                else:
                    gray = 0.299 * img_resized[:, :, 0] + 0.587 * img_resized[:, :, 1] + 0.114 * img_resized[:, :, 2]
            else:
                gray = img_resized[:, :, 0]

            gray_np = (gray * 255).clip(0, 255).astype(np.uint8)
            channel_images.append(gray_np)

        img_concatenated = np.concatenate(channel_images, axis=1)
        output_path = os.path.join(output_dir, f"{image_id}.jpg")
        Image.fromarray(img_concatenated, mode="L").save(output_path, format="JPEG", quality=JPEG_QUALITY)
        return (image_id, True, None)
    except Exception as e:
        return (image_id, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Pack HPA per-channel images into 4-channel JPGs for HPAWholeHR.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing <id>_{red,green,blue,yellow}.<extension> files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write packed <id>.jpg files into.")
    parser.add_argument("--extension", type=str, default="jpg", help="Per-channel file extension (jpg for external, tif for kaggle full-size, png for kaggle 512).")
    parser.add_argument("--target-size", type=int, default=768, help="Per-channel square size after resize.")
    parser.add_argument("--use-gpu", action="store_true", help="Use a single CUDA device sequentially (otherwise multi-process CPU).")
    parser.add_argument("--num-workers", type=int, default=16, help="CPU worker count (ignored in --use-gpu mode).")
    parser.add_argument("--chunk-size", type=int, default=100, help="Pool.imap chunksize (CPU mode).")
    args = parser.parse_args()
    extension = args.extension.lstrip(".")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Input dir : {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Extension : {extension}, target size: {args.target_size}, JPEG quality: {JPEG_QUALITY}")

    image_ids = get_image_ids(args.input_dir, extension)
    print(f"Found {len(image_ids)} unique images")

    n_success = n_failed = 0
    if args.use_gpu and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        for image_id in tqdm(image_ids, desc="Processing (GPU)"):
            result = process_single_image_gpu((image_id, args.input_dir, args.output_dir, args.target_size, extension, True))
            if result[1]:
                n_success += 1
            else:
                n_failed += 1
                if n_failed <= 5:
                    print(f"Failed: {result[0]} - {result[2]}")
    else:
        print(f"Using CPU with {args.num_workers} workers")
        work_args = [(image_id, args.input_dir, args.output_dir, args.target_size, extension, False) for image_id in image_ids]
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_image_cpu, work_args, chunksize=args.chunk_size),
                total=len(image_ids),
                desc="Processing (CPU)",
            ))
        for result in results:
            if result[1]:
                n_success += 1
            else:
                n_failed += 1

    print(f"Successfully converted: {n_success}")
    print(f"Failed: {n_failed}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
