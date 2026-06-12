# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Build HPAWholeHR's two manifest CSVs from the upstream EBI biostudies dump.

The :class:`HPAWholeHR` dataset reads two manifests under ``root``:

- ``hpa_whole_external_hr.csv`` — one row per ``HPAexternal/`` JPG
- ``hpa_whole_kaggle.csv`` — one row per ``HPAImageKaggle/`` JPG

Both are derived from two publicly-available upstream CSVs (download with
``wget`` from ``ftp.ebi.ac.uk``):

- ``Master_fovHPA_512.csv`` — every image with its protein-location labels,
  per-cell-line one-hots, and ``source`` column distinguishing ``Website``
  (HPA external) from ``Kaggle``.
- ``whole_images_512_train.csv`` — train/test partition. A file is in the
  ``train`` split iff its png basename appears here; otherwise ``test``.

Per-row column derivation:

- ``file`` — basename of ``Path`` (``.tiff``→``.png``).
- ``protein_location, cell_type, ID, <28 protein cols>`` — copied verbatim.
- ``split`` — ``train`` if file in ``whole_images_512_train``, else ``test``.
- ``hr_image, Antibody, Plate, Well, Extra``:
   - external: split the filename stem on ``_`` into 4 parts —
     ``Antibody / Plate / Well / Extra`` — and join parts[1:] for ``hr_image``.
   - kaggle: ``hr_image = file.replace(".png", ".jpg")``; antibody/plate/well/
     extra are empty.

The upstream Master CSV may carry one row per image; the on-disk packed JPGs
produced by ``hpa_whole_hr_pack_channels`` are also 1:1, so this script emits
exactly one row per png (the shared internal CSVs carried duplicate rows; that
multiplicity is an upstream pipeline artifact the dataset class does not need).

Usage::

    # 1. Fetch upstream CSVs (one-time)
    wget https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/443/S-BIAD2443/Files/HPA/Master_fovHPA_512.csv
    wget https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/443/S-BIAD2443/Files/HPA/whole_images_512_train.csv

    # 2. Build the two HPAWholeHR manifests
    python -m dinov3.data.datasets.hpa_whole_hr_build_csv \\
        --master-csv <PATH/TO/Master_fovHPA_512.csv> \\
        --train-csv  <PATH/TO/whole_images_512_train.csv> \\
        --output-dir <PATH/TO/HPAwholeHR_root/>
"""

import argparse
import csv
import os

EXTERNAL_OUT = "hpa_whole_external_hr.csv"
KAGGLE_OUT = "hpa_whole_kaggle.csv"

PROTEIN_COLS = [
    "nucleoplasm", "nuclear membrane", "nucleoli", "nucleoli fibrillar center",
    "nuclear speckles", "nuclear bodies", "endoplasmic reticulum",
    "golgi apparatus", "peroxisomes", "endosomes", "lysosomes",
    "intermediate filaments", "actin filaments", "focal adhesion sites",
    "microtubules", "microtubule ends", "cytokinetic bridge",
    "mitotic spindle", "microtubule organizing center", "centrosome",
    "lipid droplets", "plasma membrane", "cell junctions", "mitochondria",
    "aggresome", "cytosol", "cytoplasmic bodies", "rods & rings",
]

OUTPUT_COLS = (
    ["file", "protein_location", "cell_type", "ID"]
    + PROTEIN_COLS
    + ["split", "hr_image", "Antibody", "Plate", "Well", "Extra"]
)


def _load_train_basenames(train_csv: str) -> set:
    """Return the set of png basenames that belong to the ``train`` split."""
    s = set()
    with open(train_csv, newline="") as f:
        for row in csv.DictReader(f):
            s.add(os.path.basename(row["file"]))
    return s


def _derive_filename_parts(file_png: str, source: str):
    """Return (hr_image, Antibody, Plate, Well, Extra) for a row."""
    stem = file_png[:-4]  # strip .png
    if source == "Kaggle":
        return (stem + ".jpg", "", "", "", "")
    parts = stem.split("_")
    hr_image = "_".join(parts[1:]) + ".jpg"
    antibody = parts[0] if len(parts) > 0 else ""
    plate = parts[1] if len(parts) > 1 else ""
    well = parts[2] if len(parts) > 2 else ""
    extra = parts[3] if len(parts) > 3 else ""
    return (hr_image, antibody, plate, well, extra)


def build(master_csv: str, train_csv: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    train_basenames = _load_train_basenames(train_csv)
    print(f"Loaded {len(train_basenames)} train-split filenames from {train_csv}")

    ext_path = os.path.join(output_dir, EXTERNAL_OUT)
    kag_path = os.path.join(output_dir, KAGGLE_OUT)
    n_ext = n_kag = n_skip = 0

    with open(master_csv, newline="") as mf, \
         open(ext_path, "w", newline="") as ef, \
         open(kag_path, "w", newline="") as kf:
        reader = csv.DictReader(mf)
        ext_w = csv.DictWriter(ef, fieldnames=OUTPUT_COLS)
        kag_w = csv.DictWriter(kf, fieldnames=OUTPUT_COLS)
        ext_w.writeheader()
        kag_w.writeheader()

        seen = set()
        for row in reader:
            file_png = os.path.basename(row["Path"]).replace(".tiff", ".png")
            if file_png in seen:
                continue
            seen.add(file_png)

            source = row.get("source", "")
            if source not in ("Website", "Kaggle"):
                n_skip += 1
                continue

            split = "train" if file_png in train_basenames else "test"
            hr_image, antibody, plate, well, extra = _derive_filename_parts(file_png, source)

            out = {
                "file": file_png,
                "protein_location": row["protein_location"],
                "cell_type": row["cell_type"],
                "ID": row["ID"],
                "split": split,
                "hr_image": hr_image,
                "Antibody": antibody,
                "Plate": plate,
                "Well": well,
                "Extra": extra,
            }
            for col in PROTEIN_COLS:
                out[col] = row[col]

            if source == "Website":
                ext_w.writerow(out)
                n_ext += 1
            else:
                kag_w.writerow(out)
                n_kag += 1

    print(f"Wrote {n_ext} external rows -> {ext_path}")
    print(f"Wrote {n_kag} kaggle rows   -> {kag_path}")
    if n_skip:
        print(f"Skipped {n_skip} rows with unrecognized source")


def main():
    parser = argparse.ArgumentParser(description="Build HPAWholeHR manifest CSVs from EBI biostudies CSVs.")
    parser.add_argument("--master-csv", type=str, required=True, help="Path to Master_fovHPA_512.csv (upstream).")
    parser.add_argument("--train-csv", type=str, required=True, help="Path to whole_images_512_train.csv (upstream).")
    parser.add_argument("--output-dir", type=str, required=True, help="HPAWholeHR root: writes hpa_whole_external_hr.csv + hpa_whole_kaggle.csv here.")
    args = parser.parse_args()
    build(args.master_csv, args.train_csv, args.output_dir)


if __name__ == "__main__":
    main()
