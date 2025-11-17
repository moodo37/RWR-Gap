
"""
Prepare ISIC 2019 dataset for binary classification (melanoma vs. nevus).

- Scan all image files under ISIC_2019_Training_Input.
- Read ISIC_2019_Training_GroundTruth.csv.
- Copy:
    MEL == 1  → pneumonia/ (positive class)
    NV  == 1  → normal/    (negative class)
- Save normal.csv and pneumonia.csv listing filenames.

This script does NOT upload any data; it only reorganizes files locally.
"""

import os
import shutil
import argparse
import pandas as pd


def scan_images(img_root):
    """
    Recursively scan img_root and build a mapping: stem -> full image path.

    Only the first occurrence for each stem is kept, so that duplicated
    filenames in different subfolders are avoided.
    """
    stem2path = {}

    for root, dirs, files in os.walk(img_root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                stem, _ = os.path.splitext(f)
                if stem not in stem2path:
                    stem2path[stem] = os.path.join(root, f)

    print(f"[scan] found {len(stem2path)} images (unique stems)")
    return stem2path


def copy_by_ids(stem2path, ids, outdir, tag):
    """
    Copy images whose stem is in `ids` into `outdir`.

    Args:
        stem2path (dict): mapping from image stem to full path.
        ids (list[str]): list of stems to copy.
        outdir (str): output directory path.
        tag (str): label for logging.

    Returns:
        (copied, missing): numbers of successfully copied and missing images.
    """
    os.makedirs(outdir, exist_ok=True)
    copied, missing = 0, 0
    missing_samples = []

    for stem in ids:
        p = stem2path.get(stem)
        if p and os.path.exists(p):
            shutil.copy2(p, outdir)
            copied += 1
        else:
            missing += 1
            if len(missing_samples) < 5:
                missing_samples.append(stem)

    print(f"[{tag}] copied={copied}, missing={missing}")
    if missing_samples:
        print(f"[{tag}] example missing stems: {missing_samples}")
    return copied, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--isic-root",
        required=True,
        help="Root directory of the ISIC 2019 dataset "
             "(the directory containing ISIC_2019_Training_Input/ "
             "and ISIC_2019_Training_GroundTruth.csv).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory where 'normal' and 'pneumonia' folders "
             "and CSV files will be created.",
    )
    args = parser.parse_args()

    isic_root = args.isic_root
    out_root = args.output_root

    img_root = os.path.join(isic_root, "ISIC_2019_Training_Input")
    assert os.path.isdir(img_root), f"Not found: {img_root}"

    stem2path = scan_images(img_root)

    # Read ground truth CSV
    csv_path = os.path.join(isic_root, "ISIC_2019_Training_GroundTruth.csv")
    df = pd.read_csv(csv_path)

    mel_ids = df.loc[df["MEL"] == 1, "image"].astype(str).tolist()  # melanoma
    nev_ids = df.loc[df["NV"] == 1, "image"].astype(str).tolist()   # nevus

    # Output folders (mapped to CXR naming for RWR-Gap)
    out_norm = os.path.join(out_root, "normal")
    out_pneu = os.path.join(out_root, "pneumonia")

    c1 = copy_by_ids(stem2path, nev_ids, out_norm, "normal(nevus)")
    c2 = copy_by_ids(stem2path, mel_ids, out_pneu, "pneumonia(melanoma)")

    # Save filename CSVs
    norm_files = sorted(
        f for f in os.listdir(out_norm) if not f.startswith(".")
    )
    pneu_files = sorted(
        f for f in os.listdir(out_pneu) if not f.startswith(".")
    )

    print(f"[OUT] normal files: {len(norm_files)}, pneumonia files: {len(pneu_files)}")

    os.makedirs(out_root, exist_ok=True)
    pd.DataFrame({"filename": norm_files}).to_csv(
        os.path.join(out_root, "normal.csv"), index=False
    )
    pd.DataFrame({"filename": pneu_files}).to_csv(
        os.path.join(out_root, "pneumonia.csv"), index=False
    )


if __name__ == "__main__":
    main()
