# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Download HPA-WholeHR external images from a manifest CSV.

Reads the ``file`` column of the user-supplied manifest (e.g.
``hpa_whole_external_hr.csv``), builds the four-colour-channel URLs on
``images.proteinatlas.org``, and downloads each channel as a separate JPG.

URL construction example:

    file      : 49232_769_H3_1.png
    stem      : 49232/769_H3_1                (first '_' -> '/', drop '.png')
    channel   : https://images.proteinatlas.org/49232/769_H3_1_blue.jpg

The output JPGs are named ``<save_stem>_<colour>.jpg`` where ``save_stem`` is
the manifest basename with the leading antibody prefix removed.

Rows whose ``file`` basename contains a ``-`` are skipped — those entries do
not follow the proteinatlas URL scheme.

Pair the downloaded per-channel JPGs with :mod:`hpa_whole_hr_pack_channels`
to produce the packed 4-channel JPGs that :class:`HPAWholeHR` consumes.

Usage::

    python -m dinov3.data.datasets.hpa_whole_hr_download \\
        --csv <PATH/TO/hpa_whole_external_hr.csv> \\
        --output-dir <PATH/TO/per_channel_jpgs/> \\
        --start 0 --stop 1000
"""

import argparse
import csv
import os
import re
import urllib.request

BASE_URL = "https://images.proteinatlas.org"
INVALID_PATTERN = re.compile(r"-")
COLORS = ["blue", "red", "green", "yellow"]


def build_download_list(csv_path: str):
    """Return a list of (url_stem, save_stem) tuples for valid entries."""
    entries = []
    skipped = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            basename = os.path.basename(row["file"])
            if INVALID_PATTERN.search(basename):
                skipped += 1
                continue
            name = basename.replace(".png", "")
            url_stem = f"{BASE_URL}/{name.replace('_', '/', 1)}"
            save_stem = name.split("_", 1)[1]
            entries.append((url_stem, save_stem))
    print(f"Built download list: {len(entries)} valid images, {skipped} skipped")

    seen = set()
    unique = []
    for entry in entries:
        if entry[1] not in seen:
            seen.add(entry[1])
            unique.append(entry)
    print(f"Total unique images to download: {len(unique)}")
    return unique


def main():
    parser = argparse.ArgumentParser(description="Download HPA-WholeHR per-channel JPGs.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the HPA manifest CSV (e.g. hpa_whole_external_hr.csv).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write per-channel JPGs into.")
    parser.add_argument("--start", type=int, default=0, help="First entry index to download (0-based, inclusive).")
    parser.add_argument("--stop", type=int, default=None, help="Last entry index (exclusive). Defaults to len(entries).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    entries = build_download_list(args.csv)
    stop = args.stop if args.stop is not None else len(entries)
    subset = entries[args.start:stop]
    print(f"Downloading entries {args.start}..{stop} ({len(subset)} images) -> {args.output_dir}")

    fail = 0
    for idx, (url_stem, save_stem) in enumerate(subset, start=args.start):
        try:
            for color in COLORS:
                img_url = f"{url_stem}_{color}.jpg"
                save_path = os.path.join(args.output_dir, f"{save_stem}_{color}.jpg")
                urllib.request.urlretrieve(img_url, filename=save_path)
            if idx % 100 == 0:
                print(idx)
        except Exception:
            fail += 1

    print(f"Number of failed downloads: {fail}")


if __name__ == "__main__":
    main()
