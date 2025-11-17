
"""
Statistical test for RWR-Gap across different random seeds.

Given a CSV with columns:
    tag, mode, rwr_gap, ...

We compare:
    scratch vs pretrained
using:
    - one-sample t-test (mean difference)
    - permutation test (non-parametric)
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats


def permutation_pvalue(a, b_mean, iters=50000, seed=42):
    rng = np.random.default_rng(seed)
    obs = np.mean(a) - b_mean
    cnt = 0
    for _ in range(iters):
        signs = rng.choice([-1, 1], size=len(a))
        diff = np.mean(b_mean + signs * (a - b_mean))
        if diff >= np.mean(a):
            cnt += 1
    return (cnt + 1) / (iters + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="Path to rwr_results.csv.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    scratch = df.loc[df["mode"] == "scratch", "rwr_gap"].values
    pretrained = df.loc[df["mode"] == "pretrained", "rwr_gap"].values

    print("scratch:", scratch)
    print("pretrained:", pretrained)

    pre_gap = pretrained[0] if len(pretrained) == 1 else np.mean(pretrained)

    # 1) One-sample t-test (scratch mean vs pre_gap)
    t_stat, p_val_two = stats.ttest_1samp(scratch, popmean=pre_gap)
    if np.mean(scratch) > pre_gap:
        p_val_one = p_val_two / 2
    else:
        p_val_one = 1 - (p_val_two / 2)

    print(f"\n[One-sample t-test]")
    print(f"pre_gap          = {pre_gap:.4f}")
    print(f"mean(scratch)    = {np.mean(scratch):.4f} ± {np.std(scratch, ddof=1):.4f}")
    print(f"t = {t_stat:.4f}, one-sided p = {p_val_one:.4e}")

    # 2) Permutation test
    p_perm = permutation_pvalue(scratch, pre_gap, iters=50000)
    print(f"[Permutation test] one-sided p = {p_perm:.4e}")


if __name__ == "__main__":
    main()
