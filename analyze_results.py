#!/usr/bin/env python3
"""
Convert the original Jupyter notebook to a standalone Python script.

What it does:
- Computes AUC / EER / balanced accuracy from score files
- Prints dataset-wise results for several experiment patterns

Usage examples:
  python eval_results.py
  python eval_results.py --root /nas/home/dsalvi/mixture_of_experts/results/lcnn_results --pattern "LCNN_ADD.txt"
  python eval_results.py --root /nas/home/vnegroni/mixture_of_experts/results/lcnn_results --pattern "MOE_CLASSIC.txt"
  python eval_results.py --root /nas/home/vnegroni/mixture_of_experts/results/lcnn_results
  python eval_results.py --no-tex
"""
import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from typing import Iterable, List, Optional, Tuple
from sklearn.metrics import auc, balanced_accuracy_score, roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from matplotlib import rcParams


def set_matplotlib_style(use_tex: bool = True, font_size: int = 34) -> None:
    """Match the notebook's matplotlib rcParams, with a flag to disable LaTeX."""
    rcParams["font.family"] = "serif"
    rcParams["text.usetex"] = "true" if use_tex else "false"
    rcParams["font.size"] = font_size


def compute_metrics(labels: Iterable[int], pred: Iterable[float]) -> Tuple[float, float, float]:
    """
    Compute ROC AUC, EER (equal error rate), and balanced accuracy with fake-probability threshold of 0.5.

    Args:
        labels: Ground-truth labels, one per recording:
            0 = real (bona fide), 1 = fake (spoof).
            Both classes must be present.

        pred: Predicted probabilities of REAL audio, one per recording, in the same order as labels.
            Values must be between 0 and 1.
            Internally, (1.0 - pred) converts them to probabilities of FAKE audio.

    Returns:
        rocauc: Area under the ROC curve, treating fake as the positive class. Higher is better.

        eer: Interpolated equal error rate, where the FPR and FNR are equal. Lower is better.

        acc: Balanced accuracy, the mean recall across the two classes.
            A recording is classified as fake when its predicted probability of fake is greater than 0.5.
            probability of exactly 0.5 is classified as real. Higher is better.
    """
    labels = np.asarray(labels)
    pred = 1.0 - np.asarray(pred)  # <- Probability of fake

    fpr, tpr, _ = roc_curve(labels, pred)
    rocauc = auc(fpr, tpr)

    fnr = 1.0 - tpr
    eer_func = interp1d(fpr, fnr)
    eer = brentq(lambda x: eer_func(x) - x, 0.0, 1.0)

    binary_pred = np.where(pred > 0.5, 1, 0)
    acc = balanced_accuracy_score(labels, binary_pred)

    return rocauc, eer, acc

def read_score_file(path: str) -> pd.DataFrame:
    """
    Read a score txt with format:
      <utt_or_path> <score> <label>
    """
    df = pd.read_csv(path, header=None, sep=";")
    df = df.rename(columns={0: "path", 1: "score", 2: "label"})
    return df


def eval_dataset_results(root: str, pattern: str) -> None:
    """
    Replicates the notebook blocks that iterate over datasets and print results.
    Also computes a simple mean-score ensemble for all results.
    """
    dataset_list = sorted(glob(os.path.join(root, "*")))
    for dataset in dataset_list:
        score_ensemble: List[np.ndarray] = []

        print(os.path.basename(dataset).upper() + " DATASET")
        print(dataset)

        result_list = sorted(glob(os.path.join(dataset, pattern)))
        if not result_list:
            print(f"(no matches for pattern {pattern})\n")
            continue

        last_df: Optional[pd.DataFrame] = None

        for result in result_list:
            df = read_score_file(result)
            last_df = df

            rocauc, eer, bal_acc = compute_metrics(df["label"], df["score"])

            exp_name = os.path.basename(result)[:-4]  # strip .txt

            score_ensemble.append(df["score"].to_numpy())

            print(f"{exp_name}\t AUC: {rocauc*100:.2f}\t EER: {eer*100:.2f}\t bal_acc: {bal_acc*100:.2f}")

        # Ensemble (only if we accumulated multiple scores and we have labels)
        if last_df is not None and len(score_ensemble) > 0:
            ens = np.mean(np.stack(score_ensemble, axis=0), axis=0)
            rocauc, eer, bal_acc = compute_metrics(last_df["label"], ens)
            # Notebook had this print commented out; keeping it commented by default.
            # print(f"ENSEMBLE\t AUC: {rocauc*100:.2f}\t EER: {eer*100:.2f}\t bal_acc: {bal_acc*100:.2f}")

        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LCNN/MoE result files (converted from notebook).")
    parser.add_argument(
        "--root", action="append",
        default=None,
        help="Root directory containing per-dataset subfolders. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pattern", default='*.txt',
        help="Glob pattern inside each dataset folder (e.g., 'LCNN*.txt', or 'MoE-LCNN*.txt'). "
    )
    parser.add_argument("--no-tex", action="store_true", help="Disable LaTeX rendering in matplotlib.")
    parser.add_argument("--font-size", type=int, default=34, help="Matplotlib base font size.")
    args = parser.parse_args()

    set_matplotlib_style(use_tex=not args.no_tex, font_size=args.font_size)

    # If user specified a pattern, apply it to all roots provided.
    for r in args.root or ["results"]:
        eval_dataset_results(root=r, pattern=args.pattern)


if __name__ == "__main__":
    main()
