# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""HPA-WholeHR multi-channel cellular imagery dataset with optional metadata.

Returns 4-channel tensors decoded by :class:`PackedXChannelImageDecoder` from
JPG files whose channels are packed Fortran-style along the width. When
``with_metadata=True``, ``__getitem__`` yields ``(image, (label, _HPAMetadata))``
so guide heads can attach to per-sample plate/antibody/cell-type fields.

Two CSVs (``hpa_whole_external_hr.csv``, ``hpa_whole_kaggle.csv``) must be
supplied at ``root``. Images live under ``HPAexternal/jpg_768x768_4channels``
and ``HPAImageKaggle/jpg_768_{train,test}`` relative to ``root``.

Splits:

- TRAIN_SSL: self-supervised pretraining; iterates over the on-disk JPGs.
- BAL_TRAIN: class-balanced protein-location split (~4634 samples per class).
- VAL: labeled validation split.
- TEST: kaggle held-out images (no labels — dummy one-hot is emitted).
"""

import csv
import os
import random
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .decoders import PackedXChannelImageDecoder
from .extended import ExtendedVisionDataset


@dataclass
class _HPAMetadata:
    """Per-sample HPA metadata. Field names must match guide ``name`` entries."""

    environment_labels: Any  # plate_id (contiguous int, -1 if missing)
    guiding_labels: Any      # antibody index (contiguous int, -1 if missing)
    secondary_technical_labels: Any  # cell-type one-hot (np.ndarray)


CELL_TYPE = [
    "BJ", "LHCN-M2", "RH-30", "SH-SY5Y", "U-2 OS",
    "ASC TERT1", "HaCaT", "A-431", "U-251 MG", "HEK 293",
    "A549", "RT4", "HeLa", "MCF7", "PC-3",
    "hTERT-RPE1", "SK-MEL-30", "EFO-21", "AF22", "HEL",
    "Hep G2", "HUVEC TERT2", "THP-1", "CACO-2", "JURKAT",
    "RPTEC TERT1", "SuSa", "REH", "HDLM-2", "K-562",
    "hTCEpi", "NB-4", "HAP1", "OE19", "SiHa",
]

# Matches https://www.kaggle.com/c/human-protein-atlas-image-classification/data
PROTEIN_LOCATION = [
    "nucleoplasm", "nuclear membrane", "nucleoli", "nucleoli fibrillar center",
    "nuclear speckles", "nuclear bodies", "endoplasmic reticulum",
    "golgi apparatus", "peroxisomes", "endosomes", "lysosomes",
    "intermediate filaments", "actin filaments", "focal adhesion sites",
    "microtubules", "microtubule ends", "cytokinetic bridge",
    "mitotic spindle", "microtubule organizing center", "centrosome",
    "lipid droplets", "plasma membrane", "cell junctions", "mitochondria",
    "aggresome", "cytosol", "cytoplasmic bodies", "rods & rings",
]
NUM_LABELS = len(PROTEIN_LOCATION)


_SampleT = Tuple[str, np.ndarray]
_SamplesT = List[_SampleT]


class _Split(Enum):
    TRAIN_SSL = "train_ssl"
    BAL_TRAIN = "bal_train"
    VAL = "val"
    TEST = "test"


_cell_type_to_idx = {c: i for i, c in enumerate(CELL_TYPE)}


def _parse_metadata_from_row(row: Dict[str, str]) -> Tuple[np.ndarray, int, int]:
    """Return (cell_type_onehot, plate_id, antibody_id) from a CSV row.

    ``plate_id`` and ``antibody_id`` are ``-1`` when missing (e.g. kaggle rows).
    """
    cell_type_onehot = np.zeros(len(CELL_TYPE), dtype=np.int_)
    cell_line_csv = row.get("cell_type", "")
    if cell_line_csv in _cell_type_to_idx:
        cell_type_onehot[_cell_type_to_idx[cell_line_csv]] = 1

    plate_str = row.get("Plate", "")
    plate_id = int(plate_str) if plate_str != "" else -1

    antibody_str = row.get("Antibody", "")
    antibody_id = int(antibody_str) if antibody_str != "" else -1

    return cell_type_onehot, plate_id, antibody_id


def _parse_csv(
    root: str,
    csv_and_dirs: List[Tuple[str, str]],
    split_filter: Optional[str] = None,
):
    """Parse the HPA CSVs into protein-location multilabel samples + metadata.

    Returns:
        samples: list of ``(image_relpath, protein_location_onehot)``.
        index_images_by_class / index_labels_by_class: per-class indexed views
            used by the BAL_TRAIN balanced reshuffle.
        cell_type_labels / plate_ids / antibody_ids: aligned per-sample arrays.
            plate / antibody ids are remapped to contiguous 0-indexed labels;
            the ``-1`` sentinel for missing rows is preserved.
    """
    samples: _SamplesT = []
    index_images_by_class: Dict[int, List[str]] = {k: [] for k in range(NUM_LABELS)}
    index_labels_by_class: Dict[int, List[np.ndarray]] = {k: [] for k in range(NUM_LABELS)}
    cell_type_labels: List[np.ndarray] = []
    plate_ids: List[int] = []
    antibody_ids: List[int] = []

    for csv_labels_relpath, image_reldir in csv_and_dirs:
        with open(os.path.join(root, csv_labels_relpath)) as f:
            for row in csv.DictReader(f):
                if split_filter is not None and row.get("split") != split_filter:
                    continue
                protein_location = np.zeros(NUM_LABELS, dtype=np.int_)
                for k in range(NUM_LABELS):
                    if row[PROTEIN_LOCATION[k]] == "True":
                        protein_location[k] = 1
                for k in range(NUM_LABELS):
                    if row[PROTEIN_LOCATION[k]] == "True":
                        image_relpath = os.path.join(image_reldir, row["hr_image"])
                        index_images_by_class[k].append(image_relpath)
                        index_labels_by_class[k].append(protein_location)
                if protein_location.max() > 0.5 and row["hr_image"] != "":
                    image_relpath = os.path.join(image_reldir, row["hr_image"])
                    samples.append((image_relpath, protein_location))
                    ct_onehot, pid, aid = _parse_metadata_from_row(row)
                    cell_type_labels.append(ct_onehot)
                    plate_ids.append(pid)
                    antibody_ids.append(aid)

    plate_arr = np.array(plate_ids)
    antibody_arr = np.array(antibody_ids)
    if len(plate_arr) > 0:
        valid_plate = plate_arr != -1
        if valid_plate.any():
            _, plate_arr[valid_plate] = np.unique(plate_arr[valid_plate], return_inverse=True)
        plate_ids = plate_arr.tolist()
        valid_ab = antibody_arr != -1
        if valid_ab.any():
            _, antibody_arr[valid_ab] = np.unique(antibody_arr[valid_ab], return_inverse=True)
        antibody_ids = antibody_arr.tolist()

    return samples, index_images_by_class, index_labels_by_class, cell_type_labels, plate_ids, antibody_ids


EXTERNAL_HR_CSV_RELPATH = "hpa_whole_external_hr.csv"
KAGGLE_CSV_RELPATH = "hpa_whole_kaggle.csv"

# Relative to dataset root.
EXTERNAL_HR_IMAGE_RELDIR = os.path.join("HPAexternal", "jpg_768x768_4channels")
KAGGLE_IMAGE_RELDIR = os.path.join("HPAImageKaggle", "jpg_768_train")
KAGGLE_TEST_IMAGE_RELDIR = os.path.join("HPAImageKaggle", "jpg_768_test")


def _load_file_names_and_labels_val(root: str):
    csv_and_dirs = [
        (EXTERNAL_HR_CSV_RELPATH, EXTERNAL_HR_IMAGE_RELDIR),
        (KAGGLE_CSV_RELPATH, KAGGLE_IMAGE_RELDIR),
    ]
    samples, _, _, cell_type_labels, plate_ids, antibody_ids = _parse_csv(
        root, csv_and_dirs, split_filter="test",
    )
    image_paths, labels = zip(*samples)
    return list(image_paths), list(labels), cell_type_labels, plate_ids, antibody_ids


def _load_file_names_and_labels_balanced(root: str):
    """Per-class balanced reshuffle (4634 samples per class)."""
    csv_and_dirs = [
        (EXTERNAL_HR_CSV_RELPATH, EXTERNAL_HR_IMAGE_RELDIR),
        (KAGGLE_CSV_RELPATH, KAGGLE_IMAGE_RELDIR),
    ]
    _, index_images_by_class, index_labels_by_class, _, _, _ = _parse_csv(
        root, csv_and_dirs, split_filter="train",
    )

    nb_classes: List[int] = [len(index_images_by_class[c]) for c in range(NUM_LABELS)]
    dataset_size = sum(nb_classes)

    return index_images_by_class, index_labels_by_class, nb_classes, dataset_size


def _load_file_names_and_labels_ssl(root: str):
    """Iterate the on-disk JPGs; join metadata from the two CSVs by filename."""
    metadata_by_filename: Dict[str, Tuple[np.ndarray, int, int]] = {}
    for csv_relpath in [EXTERNAL_HR_CSV_RELPATH, KAGGLE_CSV_RELPATH]:
        with open(os.path.join(root, csv_relpath)) as f:
            for row in csv.DictReader(f):
                hr_image = row.get("hr_image", "")
                if hr_image:
                    metadata_by_filename[hr_image] = _parse_metadata_from_row(row)

    image_paths: List[str] = []
    cell_type_labels: List[np.ndarray] = []
    plate_ids: List[int] = []
    antibody_ids: List[int] = []

    for image_reldir in [EXTERNAL_HR_IMAGE_RELDIR, KAGGLE_IMAGE_RELDIR]:
        dir_path = os.path.join(root, image_reldir)
        jpg_files = sorted(f for f in os.listdir(dir_path) if f.endswith(".jpg"))
        for fname in jpg_files:
            image_paths.append(os.path.join(image_reldir, fname))
            if fname in metadata_by_filename:
                ct, pid, aid = metadata_by_filename[fname]
                cell_type_labels.append(ct)
                plate_ids.append(pid)
                antibody_ids.append(aid)
            else:
                cell_type_labels.append(np.zeros(len(CELL_TYPE), dtype=np.int_))
                plate_ids.append(-1)
                antibody_ids.append(-1)

    labels = list(range(len(image_paths)))

    plate_arr = np.array(plate_ids)
    antibody_arr = np.array(antibody_ids)
    if len(plate_arr) > 0:
        valid_plate = plate_arr != -1
        if valid_plate.any():
            _, plate_arr[valid_plate] = np.unique(plate_arr[valid_plate], return_inverse=True)
        plate_ids = plate_arr.tolist()
        valid_ab = antibody_arr != -1
        if valid_ab.any():
            _, antibody_arr[valid_ab] = np.unique(antibody_arr[valid_ab], return_inverse=True)
        antibody_ids = antibody_arr.tolist()

    return image_paths, labels, cell_type_labels, plate_ids, antibody_ids


def _load_file_names_and_labels_test(root: str):
    """Kaggle held-out split: image filenames only, with a dummy one-hot label."""
    jpg_dir_path = os.path.join(root, KAGGLE_TEST_IMAGE_RELDIR)
    image_files = sorted(f for f in os.listdir(jpg_dir_path) if f.endswith(".jpg"))
    image_paths = [os.path.join(KAGGLE_TEST_IMAGE_RELDIR, f) for f in image_files]

    fake_label = np.zeros((NUM_LABELS,), dtype=np.int_)
    fake_label[0] = 1
    labels = [fake_label for _ in image_files]
    return image_paths, labels


class HPAWholeHR(ExtendedVisionDataset):
    """HPA whole-image dataset returning 4-channel tensors and optional metadata.

    Args:
        split: One of TRAIN_SSL / BAL_TRAIN / VAL / TEST.
        with_metadata: When True (default), ``__getitem__`` returns
            ``(image, (label, _HPAMetadata))``. Use ``with_metadata=false`` for
            eval splits that go through label-only collate fns (e.g.
            ``pad_multilabel_and_collate``).
        in_chans: Number of channels to keep after the packed-JPG decode. The
            on-disk files store 4 channels; ``in_chans < 4`` truncates.
        root: Directory containing the two CSVs.
    """

    Split = _Split
    Metadata = _HPAMetadata

    def __init__(
        self,
        *,
        split: _Split = _Split.TRAIN_SSL,
        with_metadata: bool = True,
        in_chans: int = 4,
        root: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            image_decoder=partial(PackedXChannelImageDecoder, num_channels=in_chans),
            root=root,
            **kwargs,
        )

        self._split = _Split(split) if isinstance(split, str) else split
        self._with_metadata = with_metadata
        self._in_chans = in_chans
        self._cell_labels: Optional[List[np.ndarray]] = None
        self._plate_ids: Optional[List[int]] = None
        self._antibody_ids: Optional[List[int]] = None

        if self._split == _Split.VAL:
            (
                self._image_relpaths, self._labels,
                self._cell_labels, self._plate_ids, self._antibody_ids,
            ) = _load_file_names_and_labels_val(root)
        elif self._split == _Split.BAL_TRAIN:
            (
                self._image_relpaths, self._labels,
                self._nb_classes, self._dataset_size,
            ) = _load_file_names_and_labels_balanced(root)
        elif self._split == _Split.TRAIN_SSL:
            (
                self._image_relpaths, self._labels,
                self._cell_labels, self._plate_ids, self._antibody_ids,
            ) = _load_file_names_and_labels_ssl(root)
        else:  # _Split.TEST
            self._image_relpaths, self._labels = _load_file_names_and_labels_test(root)

    @property
    def split(self) -> _Split:
        return self._split

    def _balanced_pick(self, index: int) -> Tuple[int, int]:
        class_id = index % NUM_LABELS
        random.seed(int(index))
        n = random.randint(0, self._nb_classes[class_id] - 1)
        return class_id, n

    def get_image_relpath(self, index: int) -> str:
        if self._split == _Split.BAL_TRAIN:
            class_id, n = self._balanced_pick(index)
            return self._image_relpaths[class_id][n]
        return self._image_relpaths[index]

    def get_image_data(self, index: int) -> bytes:
        with open(os.path.join(self.root, self.get_image_relpath(index)), "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        if self._split == _Split.BAL_TRAIN:
            class_id, n = self._balanced_pick(index)
            target = self._labels[class_id][n]
        else:
            target = self._labels[index]

        if not self._with_metadata:
            return target

        # BAL_TRAIN / TEST do not carry per-sample metadata — emit sentinels.
        metadata = _HPAMetadata(
            environment_labels=self._plate_ids[index] if self._plate_ids is not None else -1,
            guiding_labels=self._antibody_ids[index] if self._antibody_ids is not None else -1,
            secondary_technical_labels=(
                self._cell_labels[index] if self._cell_labels is not None
                else np.zeros(len(CELL_TYPE), dtype=np.int_)
            ),
        )
        return (target, metadata)

    def get_targets(self) -> np.ndarray:
        if self._split == _Split.BAL_TRAIN:
            return np.vstack([self._labels[c] for c in range(NUM_LABELS)])
        return np.array(self._labels)

    def __len__(self) -> int:
        if self._split == _Split.BAL_TRAIN:
            return self._dataset_size
        return len(self._image_relpaths)
