
"""
Split normal/pneumonia images into train/val/test with 7:2:1 ratio,
while keeping the number of normal images equal to the number
of pneumonia images in each split.

This script expects:
    src_root/
        normal/        # image folder
        pneumonia/     # image folder
        normal.csv     # filename list
        pneumonia.csv  # filename list
"""

import os
import argparse
import shutil
import numpy as np
import pandas as pd


def split_indices(n, train_ratio=0.7, val_ratio=0.2):
    """
    Split `n` samples into train/val/test using given ratios.

    Returns:
        n_train, n_val, n_test
    """
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    assert n_train > 0 and n_val > 0 and n_test > 0
    return n_train, n_val, n_test


def copy_split(df, indices, src_root, subdir, dst_dir):
    """
    Copy selected rows from df into dst_dir.

    Args:
        df: DataFrame with a 'filename' column.
        indices: Iterable of row indices.
        src_root: Root directory containing 'subdir'.
        subdir: Subdirectory name ('normal' or 'pneumonia').
        dst_dir: Output directory.
    """
    os.makedirs(dst_dir, exist_ok=True)
    missing = 0

    for idx in indices:
        row = df.iloc[idx]
        fname = row["filename"]
        src = os.path.join(src_root, subdir, fname)
        dst = os.path.join(dst_dir, fname)

        if not os.path.exists(src):
            missing += 1
            continue

        shutil.copy2(src, dst)

    if missing > 0:
        print(f"[WARN] {dst_dir}: {missing} files were not found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        required=True,
        help="Root directory with 'normal'/'pneumonia' folders and CSVs.",
    )
    parser.add_argument(
        "--dst-root",
        required=True,
        help="Root directory where train/val/test splits will be created.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pneu-subdir",
        default="pneumonia",
        help="Subfolder name for pneumonia images (if different).",
    )
    parser.add_argument(
        "--normal-subdir",
        default="normal",
        help="Subfolder name for normal images (if different).",
    )
    args = parser.parse_args()

    src_root = args.src_root
    dst_root = args.dst_root
    os.makedirs(dst_root, exist_ok=True)

    # Load CSV
    normal_csv = os.path.join(src_root, "normal.csv")
    pneu_csv = os.path.join(src_root, "pneumonia.csv")

    normal_df = pd.read_csv(normal_csv)
    pneu_df = pd.read_csv(pneu_csv)

    N_pneu = len(pneu_df)
    N_normal = len(normal_df)

    print(f"[INFO] Normal total    : {N_normal}")
    print(f"[INFO] Pneumonia total : {N_pneu}")

    # Split pneumonia first (7:2:1)
    n_train_p, n_val_p, n_test_p = split_indices(N_pneu, 0.7, 0.2)
    print(f"[INFO] Pneumonia split → train={n_train_p}, val={n_val_p}, test={n_test_p}")

    rng_p = np.random.RandomState(args.seed)
    pneu_indices = np.arange(N_pneu)
    rng_p.shuffle(pneu_indices)

    train_p_idx = pneu_indices[:n_train_p]
    val_p_idx = pneu_indices[n_train_p: n_train_p + n_val_p]
    test_p_idx = pneu_indices[n_train_p + n_val_p:]

    # Normal: use the same number of samples as pneumonia
    total_needed = n_train_p + n_val_p + n_test_p
    assert N_normal >= total_needed, "Not enough normal images."

    rng_n = np.random.RandomState(args.seed + 1)
    normal_indices_all = np.arange(N_normal)
    rng_n.shuffle(normal_indices_all)

    train_n_idx = normal_indices_all[:n_train_p]
    val_n_idx = normal_indices_all[n_train_p: n_train_p + n_val_p]
    test_n_idx = normal_indices_all[n_train_p + n_val_p: n_train_p + n_val_p + n_test_p]

    print(f"[INFO] Normal used     → train={len(train_n_idx)}, "
          f"val={len(val_n_idx)}, test={len(test_n_idx)}")

    # Create split folders
    for sp in ["train", "val", "test"]:
        for cls in ["normal", "pneumonia"]:
            os.makedirs(os.path.join(dst_root, sp, cls), exist_ok=True)

    print("[INFO] Copying TRAIN...")
    copy_split(normal_df, train_n_idx, src_root, args.normal_subdir,
               os.path.join(dst_root, "train", "normal"))
    copy_split(pneu_df, train_p_idx, src_root, args.pneu_subdir,
               os.path.join(dst_root, "train", "pneumonia"))

    print("[INFO] Copying VAL...")
    copy_split(normal_df, val_n_idx, src_root, args.normal_subdir,
               os.path.join(dst_root, "val", "normal"))
    copy_split(pneu_df, val_p_idx, src_root, args.pneu_subdir,
               os.path.join(dst_root, "val", "pneumonia"))

    print("[INFO] Copying TEST...")
    copy_split(normal_df, test_n_idx, src_root, args.normal_subdir,
               os.path.join(dst_root, "test", "normal"))
    copy_split(pneu_df, test_p_idx, src_root, args.pneu_subdir,
               os.path.join(dst_root, "test", "pneumonia"))

    print("\n[DONE] Created 7:2:1 split for RWR-Gap experiments.")
    print(f" - Train: normal={len(train_n_idx)}, pneu={len(train_p_idx)}")
    print(f" - Val  : normal={len(val_n_idx)}, pneu={len(val_p_idx)}")
    print(f" - Test : normal={len(test_n_idx)}, pneu={len(test_p_idx)}")
    print("Now you can further split 'test' into Easy/Robust/Conflict splits.")


if __name__ == "__main__":
    main()
