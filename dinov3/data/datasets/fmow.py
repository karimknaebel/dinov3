# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Functional Map of the World (FMoW) dataset with per-sample metadata.

Used as the reference dataset for metadata-guided SSL training. Returns
``(image, (label, _Metadata))`` so the collate fn can route metadata fields
(country, sub_region, year, ...) to guide heads.

Images are flat-indexed PNGs under ``<root>/images/rgb_img_{row_index}.png``.
Metadata comes from ``<root>/rgb_metadata_v2.csv`` and country mappings from
``<root>/country_code_mapping.csv``.

Splits follow the WILDS temporal convention:

- TRAIN:    original ``train``,  year < 2013
- VAL_ID:   original ``val``,    year < 2013
- TEST_ID:  original ``test``,   year < 2013
- VAL_OOD:  original ``val``,    2013 <= year <= 2015
- TEST_OOD: original ``test``,   year >= 2016
"""

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset

FMOW_REGION_NAMES = ("Africa", "Americas", "Asia", "Europe", "Oceania", "Unknown")

_LAT_MIN, _LAT_MAX = -90.0, 90.0
_LON_MIN, _LON_MAX = -180.0, 180.0
_GSD_MIN, _GSD_MAX = 0.2, 1.8
_OFF_NADIR_MIN, _OFF_NADIR_MAX = 0.0, 60.0
_SUN_ELEV_MIN, _SUN_ELEV_MAX = 0.0, 80.0

_MONTH_TO_SEASON = {
    12: 0, 1: 0, 2: 0,   # winter
    3: 1, 4: 1, 5: 1,    # spring
    6: 2, 7: 2, 8: 2,    # summer
    9: 3, 10: 3, 11: 3,  # fall
}


@dataclass
class _Metadata:
    """Per-sample FMoW metadata. Field names must match guide ``name`` entries."""

    country: int            # Contiguous country ID (207 classes)
    region: int             # Continent (6 classes)
    sub_region: int         # Sub-region (18 classes)
    cloud_cover: int        # Cloud cover percentage (0-100)
    month: int              # Month, 0-indexed (12 classes)
    year: int               # Year index (16 classes: 0-15 for 2002-2017)
    hour_utc: int           # Hour UTC (24 classes)
    season: int             # Season (4 classes)
    visible: int            # Object visible (2 classes)
    coordinates: Tuple[float, float]  # (lat_norm, lon_norm) in [0, 1]
    gsd: float              # Ground sample distance (regression, 0.2-1.8)
    off_nadir_angle: float  # Off-nadir angle in degrees (regression, 0-60)
    sun_elevation: float    # Sun elevation in degrees (regression, 0-80)
    hour_sincos: Tuple[float, float]  # Cyclical hour encoding


class _Split(Enum):
    TRAIN = "train"
    VAL_ID = "val_id"
    TEST_ID = "test_id"
    VAL_OOD = "val_ood"
    TEST_OOD = "test_ood"


def _extract_year(ts_str: str) -> Optional[int]:
    if not ts_str or ts_str == "nan":
        return None
    try:
        return int(ts_str[:4])
    except (ValueError, IndexError):
        return None


def _split_filter(orig_split: str, year: Optional[int], target_split: _Split) -> bool:
    if year is None:
        return False
    if target_split == _Split.TRAIN:
        return orig_split == "train" and year < 2013
    if target_split == _Split.VAL_ID:
        return orig_split == "val" and year < 2013
    if target_split == _Split.TEST_ID:
        return orig_split == "test" and year < 2013
    if target_split == _Split.VAL_OOD:
        return orig_split == "val" and 2013 <= year <= 2015
    if target_split == _Split.TEST_OOD:
        return orig_split == "test" and year >= 2016
    return False


def _parse_timestamp(ts_str: str) -> Tuple[int, int, int, int]:
    """Parse ISO timestamp into (month_0idx, year_idx, hour, season)."""
    ts_str = ts_str.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts_str)
    year_idx = max(0, min(15, dt.year - 2002))
    return dt.month - 1, year_idx, dt.hour, _MONTH_TO_SEASON[dt.month]


@lru_cache(maxsize=2)
def _load_country_mappings(root: str) -> Tuple[dict, dict]:
    country_to_region: dict[str, str] = {}
    country_to_subregion: dict[str, str] = {}
    with open(os.path.join(root, "country_code_mapping.csv"), "r") as f:
        for row in csv.DictReader(f):
            code = row["alpha-3"]
            country_to_region[code] = row.get("region", "")
            country_to_subregion[code] = row.get("sub-region", "")
    return country_to_region, country_to_subregion


@lru_cache(maxsize=8)
def _load_file_names_and_metadata(
    root: str,
    split: _Split,
) -> Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[_Metadata, ...]]:
    csv_path = os.path.join(root, "rgb_metadata_v2.csv")
    country_to_region, country_to_subregion = _load_country_mappings(root)

    # First pass: collect unique values across ALL splits for stable ID maps.
    all_countries: set[str] = set()
    all_categories: set[str] = set()
    all_regions: set[str] = set()
    all_subregions: set[str] = set()
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            all_countries.add(row["country_code"])
            all_categories.add(row["category"])
            reg = country_to_region.get(row["country_code"], "") or "Unknown"
            all_regions.add(reg)
            subreg = country_to_subregion.get(row["country_code"], "") or "Unknown"
            all_subregions.add(subreg)

    country_to_id = {c: i for i, c in enumerate(sorted(all_countries))}
    category_to_id = {c: i for i, c in enumerate(sorted(all_categories))}
    region_to_id = {r: i for i, r in enumerate(sorted(all_regions))}
    subregion_to_id = {s: i for i, s in enumerate(sorted(all_subregions))}

    image_paths: list[str] = []
    labels: list[int] = []
    metadata_list: list[_Metadata] = []

    with open(csv_path, "r") as f:
        for row_idx, row in enumerate(csv.DictReader(f)):
            year = _extract_year(row.get("timestamp", ""))
            if not _split_filter(row["split"], year, split):
                continue

            img_path = os.path.join("images", f"rgb_img_{row_idx}.png")

            month, year_idx, hour, season = _parse_timestamp(row.get("timestamp", ""))
            cc = row["country_code"]
            reg = country_to_region.get(cc, "") or "Unknown"
            subreg = country_to_subregion.get(cc, "") or "Unknown"
            lat, lon = float(row["lat"]), float(row["lon"])

            image_paths.append(img_path)
            labels.append(category_to_id[row["category"]])
            metadata_list.append(_Metadata(
                country=country_to_id[cc],
                region=region_to_id[reg],
                sub_region=subregion_to_id[subreg],
                cloud_cover=int(row["cloud_cover"]),
                month=month,
                year=year_idx,
                hour_utc=hour,
                season=season,
                visible=1 if row["visible"] == "True" else 0,
                coordinates=(
                    (lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN),
                    (lon - _LON_MIN) / (_LON_MAX - _LON_MIN),
                ),
                gsd=max(_GSD_MIN, min(_GSD_MAX, float(row["gsd"]))),
                off_nadir_angle=max(_OFF_NADIR_MIN, min(_OFF_NADIR_MAX, float(row["off_nadir_angle_dbl"]))),
                sun_elevation=max(_SUN_ELEV_MIN, min(_SUN_ELEV_MAX, float(row["sun_elevation_dbl"]))),
                hour_sincos=(
                    math.sin(2.0 * math.pi * hour / 24.0),
                    math.cos(2.0 * math.pi * hour / 24.0),
                ),
            ))

    return tuple(image_paths), tuple(labels), tuple(metadata_list)


class FMoW(ExtendedVisionDataset):
    """FMoW satellite imagery dataset returning ``(image, (label, _Metadata))``.

    Args:
        split: WILDS temporal split (TRAIN / VAL_ID / TEST_ID / VAL_OOD / TEST_OOD).
        root: Directory containing ``rgb_metadata_v2.csv``, ``country_code_mapping.csv``,
            and ``images/rgb_img_<idx>.png``.
        with_metadata: When TRUE (default), ``__getitem__`` returns
            ``(image, (label, _Metadata))``. When FALSE, returns ``(image, label)``
            — required for evaluators whose default collate can't batch the
            dataclass.
    """

    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "FMoW.Split" = _Split.TRAIN,
        root: str,
        with_metadata: bool = True,
        **kwargs: Any,
    ) -> None:
        self._split = split
        self._with_metadata = with_metadata
        super().__init__(root=root, **kwargs)

        self._image_paths, self._labels, self._metadata = _load_file_names_and_metadata(
            root=root, split=split,
        )

    @property
    def split(self) -> "FMoW.Split":
        return self._split

    def get_image_data(self, index: int) -> bytes:
        with open(os.path.join(self.root, self._image_paths[index]), "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        if self._with_metadata:
            return self._labels[index], self._metadata[index]
        return self._labels[index]

    def get_targets(self) -> np.ndarray:
        return np.array(self._labels)

    def __len__(self) -> int:
        return len(self._image_paths)
