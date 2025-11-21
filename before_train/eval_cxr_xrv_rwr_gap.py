
"""
Evaluate XRV DenseNet models (scratch vs pretrained) on the
CXR RWR-Gap benchmark.

This script reproduces the evaluation reported in the RWR-Gap paper.

Given:
  - Scratch model checkpoint (trained using train_cxr_xrv_scratch.py)
  - Pretrained XRV DenseNet (e.g., mimic_nb weights)

It computes:
  - Test-Easy accuracy
  - Test-Robust accuracy
  - Test-Conflict accuracy
  - The RWR-Gap metric

Expected directory structure:
    data/cxr_rwr_gap/
        test_easy/
        test_robust/
        test_conflict/

"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

try:
    import torchxrayvision as xrv
except ImportError:
    raise ImportError("Please pip install torchxrayvision")


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------
class RWRDataset(datasets.ImageFolder):
    """Map 'normal' -> 0, any pneumonia-like label -> 1"""

    def __getitem__(self, i):
        path, _ = self.samples[i]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)

        label_name = Path(path).parent.name.lower()
        label = 0 if "normal" in label_name else 1
        return img, label


# -------------------------------------------------------
# XRV model builders
# -------------------------------------------------------
def build_xrv_scratch(num_classes=1, ckpt=None):
    """Build XRV DenseNet scratch model"""
    model = xrv.models.DenseNet(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.n_classes = num_classes
    model.pathologies = ["Pneumonia"]

    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd["model_state"], strict=False)

    return model


def build_xrv_pretrained(xrv_weights, num_classes=1):
    """
    Load XRV pretrained DenseNet.
    Example weights:
        densenet121-res224-mimic_nb
    """
    model = xrv.models.DenseNet(weights=xrv_weights)

    # Fix classifier dimension if needed
    if getattr(model, "n_classes", None) != num_classes:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        model.n_classes = num_classes

    try:
        model.pathologies = ["Pneumonia"]
    except Exception:
        pass

    return model


# -------------------------------------------------------
# Common evaluation
# -------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(1)

        all_labels.append(y.cpu())
        all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    preds = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return float(acc), float(auc)


def compute_rwr_gap(acc_all, acc_robust, acc_conflict):
    gap_robust = acc_all - acc_robust
    gap_conflict = 1 - acc_conflict
    return gap_robust, gap_conflict, gap_robust + gap_conflict


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # -----------------------------
    # Load models
    # -----------------------------
    scratch = build_xrv_scratch(
        num_classes=1,
        ckpt=args.scratch_ckpt,
    ).to(device)

    pretrained = build_xrv_pretrained(
        xrv_weights=args.xrv_weights,
        num_classes=1,
    ).to(device)

    # -----------------------------
    # Dataloaders
    # -----------------------------
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    root = Path(args.data_root)
    test_easy     = DataLoader(RWRDataset(root/"test_easy",     transform=tf), batch_size=args.batch_size, shuffle=False)
    test_robust   = DataLoader(RWRDataset(root/"test_robust",   transform=tf), batch_size=args.batch_size, shuffle=False)
    test_conflict = DataLoader(RWRDataset(root/"test_conflict", transform=tf), batch_size=args.batch_size, shuffle=False)

    # -----------------------------
    # Evaluate both models
    # -----------------------------
    print("\n=== XRV Scratch Evaluation ===")
    acc_s_easy,     auc_s_easy     = evaluate(scratch, test_easy, device)
    acc_s_robust,   auc_s_robust   = evaluate(scratch, test_robust, device)
    acc_s_conflict, auc_s_conflict = evaluate(scratch, test_conflict, device)

    gap_s_r, gap_s_c, gap_s = compute_rwr_gap(acc_s_easy, acc_s_robust, acc_s_conflict)

    print(f"Scratch — Acc_all={acc_s_easy:.4f}, Acc_robust={acc_s_robust:.4f}, Acc_conflict={acc_s_conflict:.4f}")
    print(f"Scratch — RWR-Gap = {gap_s:.4f}")

    print("\n=== XRV Pretrained Evaluation ===")
    acc_p_easy,     auc_p_easy     = evaluate(pretrained, test_easy, device)
    acc_p_robust,   auc_p_robust   = evaluate(pretrained, test_robust, device)
    acc_p_conflict, auc_p_conflict = evaluate(pretrained, test_conflict, device)

    gap_p_r, gap_p_c, gap_p = compute_rwr_gap(acc_p_easy, acc_p_robust, acc_p_conflict)

    print(f"Pretrained — Acc_all={acc_p_easy:.4f}, Acc_robust={acc_p_robust:.4f}, Acc_conflict={acc_p_conflict:.4f}")
    print(f"Pretrained — RWR-Gap = {gap_p:.4f}")

    # -----------------------------
    # Save JSON
    # -----------------------------
    results = {
        "scratch": {
            "acc_all": acc_s_easy,
            "acc_robust": acc_s_robust,
            "acc_conflict": acc_s_conflict,
            "rwr_gap": gap_s,
        },
        "pretrained": {
            "acc_all": acc_p_easy,
            "acc_robust": acc_p_robust,
            "acc_conflict": acc_p_conflict,
            "rwr_gap": gap_p,
        }
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Saved results → {args.out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate XRV DenseNet scratch vs pretrained on RWR-Gap CXR"
    )

    parser.add_argument("--data_root", type=str, default="data/cxr_rwr_gap")
    parser.add_argument("--scratch_ckpt", type=str, required=True,
                        help="Path to scratch-trained checkpoint (*.pth)")
    parser.add_argument("--xrv_weights", type=str, default="densenet121-res224-mimic_nb",
                        help="XRV pretrained model weights")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_json", type=str, default="outputs/xrv_eval/results.json")

    args = parser.parse_args()
    main(args)
