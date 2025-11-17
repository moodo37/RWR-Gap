
"""
Analyze CAM energy statistics (marker vs lesion) for scratch vs pretrained.

This script:
  - Loads a CSV produced by generate_cam_triplets_*.py
  - Performs paired t-tests & Wilcoxon tests
  - Estimates 95% bootstrap CI of energy differences
  - Optionally saves a scatter plot
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt


def bootstrap_ci(diff, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(B)]
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="Path to CAM energy CSV (e.g., cam_energy_stats_isic.csv).")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to save plots.")
    ap.add_argument("--prefix", type=str, default="cam_energy",
                    help="Prefix for saved plot filenames.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print("Data:", df.shape)
    print(df.head())

    # 1) Marker energy: scratch vs pretrained
    t_m, p_m = ttest_rel(df["k_m_s"], df["k_m_p"])
    w_m, p_wm = wilcoxon(df["k_m_s"], df["k_m_p"])
    print(f"\n[Marker energy] scratch > pretrained ?")
    print(f"  paired t-test p = {p_m:.3e}")
    print(f"  Wilcoxon       p = {p_wm:.3e}")

    # 2) Lesion (lung/skin) energy: pretrained vs scratch
    t_l, p_l = ttest_rel(df["k_l_p"], df["k_l_s"])
    w_l, p_wl = wilcoxon(df["k_l_p"], df["k_l_s"])
    print(f"\n[Lesion energy] pretrained > scratch ?")
    print(f"  paired t-test p = {p_l:.3e}")
    print(f"  Wilcoxon       p = {p_wl:.3e}")

    # 3) Mean diff + 95% bootstrap CI
    diff_marker = df["k_m_s"] - df["k_m_p"]
    diff_lesion = df["k_l_p"] - df["k_l_s"]
    ci_m = bootstrap_ci(diff_marker.values)
    ci_l = bootstrap_ci(diff_lesion.values)

    print(f"\nMean diff (marker) = {diff_marker.mean():.4f}, "
          f"95% CI [{ci_m[0]:.4f}, {ci_m[1]:.4f}]")
    print(f"Mean diff (lesion) = {diff_lesion.mean():.4f}, "
          f"95% CI [{ci_l[0]:.4f}, {ci_l[1]:.4f}]")

    # 4) Scatter plot (marker energy)
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    plt.figure(figsize=(5, 4))
    lim = max(df["k_m_s"].max(), df["k_m_p"].max()) * 1.05
    plt.scatter(df["k_m_s"], df["k_m_p"], s=8, alpha=0.4)
    plt.plot([0, lim], [0, lim], "r--", linewidth=1)
    plt.xlabel("Scratch marker energy")
    plt.ylabel("Pretrained marker energy")
    plt.title("Marker energy: scratch vs pretrained")
    plt.tight_layout()
    out_path = f"{args.out_dir}/{args.prefix}_marker_scatter.png"
    plt.savefig(out_path, dpi=200)
    print(f"[INFO] Saved scatter → {out_path}")


if __name__ == "__main__":
    main()
