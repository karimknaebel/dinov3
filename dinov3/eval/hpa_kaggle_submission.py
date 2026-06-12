# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Convert attention-pooling TEST predictions into a Kaggle HPA submission CSV.

The HPA-WholeHR ``TEST`` split is the Kaggle held-out set with no public labels;
:func:`dinov3.eval.attention_pooling.eval_attention_pooling` saves per-image
logits as ``preds_<dataset_str>.npy`` under the eval ``output_dir``. This
script reads that file, applies sigmoid + 0.3 threshold (falling back to the
top-1 prediction when no class exceeds the threshold), and writes a Kaggle
submission CSV.

The model's 28 logits are assumed to be aligned with the Kaggle protein-location
class order (the order shipped by :data:`dinov3.data.datasets.hpa_whole_hr.PROTEIN_LOCATION`).

Usage::

    python -m dinov3.eval.hpa_kaggle_submission \\
        --preds <EVAL/OUTPUT>/preds_HPAWholeHR_split=TEST_..._with_metadata=false.npy \\
        --sample-submission <PATH/TO/kaggle_hpa_sample_submission.csv> \\
        --output submission.csv
"""

import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn


def write_results(preds_path: str, sample_submission_path: str, output_path: str) -> None:
    """Threshold per-image sigmoid logits and write a Kaggle submission CSV."""
    print(f"Loading predictions from {preds_path}")
    preds = torch.from_numpy(np.load(preds_path))

    print(f"Loading sample submission template from {sample_submission_path}")
    df = pd.read_csv(sample_submission_path)

    if len(df) != preds.shape[0]:
        raise ValueError(
            f"Row mismatch: sample submission has {len(df)} rows but predictions "
            f"have {preds.shape[0]}. The TEST split order is sorted-by-filename — "
            f"the sample submission row order must match."
        )

    sigmoid = nn.Sigmoid()
    for p in range(preds.shape[0]):
        sig_output = sigmoid(preds[p])
        pred_tensor = sig_output > 0.3
        list_of_ints = pred_tensor.nonzero(as_tuple=True)[0].tolist()
        if not list_of_ints:
            list_of_ints = [sig_output.argmax().item()]
        df.loc[p, "Predicted"] = " ".join(str(i) for i in list_of_ints)

    df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a Kaggle HPA submission from attention-pool predictions.")
    parser.add_argument(
        "--preds",
        type=str,
        required=True,
        help="Path to the preds_<dataset_str>.npy file written by attention_pooling.py on the TEST split.",
    )
    parser.add_argument(
        "--sample-submission",
        type=str,
        required=True,
        help="Path to the Kaggle sample submission CSV (used as the row order / Id template).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the submission CSV.",
    )
    args = parser.parse_args()
    write_results(args.preds, args.sample_submission, args.output)


if __name__ == "__main__":
    main()
