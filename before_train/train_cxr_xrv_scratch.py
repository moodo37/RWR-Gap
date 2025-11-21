#!/usr/bin/env python
"""
Train an XRV (torchxrayvision) DenseNet model *from scratch*
on the CXR RWR-Gap benchmark.

This script reproduces the baseline "XRV DenseNet scratch" model
used for comparison in the RWR-Gap paper.

Features:
  - Binary classification: normal (0) vs pneumonia (1)
  - BCEWithLogitsLoss
  - ISIC-like RWR-Gap evaluation splitting:
        test_easy, test_robust, test_conflict
  - Saves best model based on validation accuracy
  - Produces a JSON file summarizing all metrics

Expected directory structure:

    data/cxr_rwr_gap/
        train/
            normal/
            pneumonia/
        val/
            normal/
            pneumonia/
        test_easy/
            normal/
            pneumonia/
        test_robust/
            pneumonia/
        test_conflict/
            normal/

"""

import argparse
import os
import random
from pathlib import Path
import json

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
    raise ImportError("Please install torchxrayvision: pip install torchxrayvision")


# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------
# RWR dataset: normal=0, pneumonia=1
# -------------------------------------------------------------
class RWRDataset(datasets.ImageFolder):
    """
    ImageFolder-based dataset for CXR RWR-Gap benchmark.

    Label mapping:
        - folder contains "normal"     → 0
        - otherwise (pneumonia, marker) → 1
    """

    def __getitem__(self, index):
        path, _ = self.samples[index]
        img = self.loader(path)

        if self.transform:
            img = self.transform(img)

        dirname = Path(path).parent.name.lower()
        if "normal" in dirname:
            target = 0
        else:
            target = 1

        return img, target


# -------------------------------------------------------------
# Model builder (XRV DenseNet scratch)
# -------------------------------------------------------------
def build_xrv_densenet(num_classes=1):
    """
    XRV DenseNet initialized from scratch.

    - Output dim = 1 (binary)
    - Uses BCEWithLogitsLoss
    """
    model = xrv.models.DenseNet(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    model.n_classes = num_classes

    # Set pathology names for consistency
    try:
        model.pathologies = ["Pneumonia"]
    except Exception:
        pass

    return model


# -------------------------------------------------------------
# DataLoader creation
# -------------------------------------------------------------
def create_dataloaders(root, batch_size, num_workers):
    """
    Creates dataloaders for train/val/test splits.

    Input images are grayscale CXR, converted to 1 channel.
    """

    root = Path(root)

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # train/val loaders
    train_ds = RWRDataset(root / "train", transform=train_tf)
    val_ds   = RWRDataset(root / "val",   transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # RWR test splits
    test_easy     = RWRDataset(root / "test_easy",     transform=eval_tf)
    test_robust   = RWRDataset(root / "test_robust",   transform=eval_tf)
    test_conflict = RWRDataset(root / "test_conflict", transform=eval_tf)

    test_easy_loader = DataLoader(
        test_easy, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_robust_loader = DataLoader(
        test_robust, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_conflict_loader = DataLoader(
        test_conflict, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test_easy": test_easy_loader,
        "test_robust": test_robust_loader,
        "test_conflict": test_conflict_loader,
    }


# -------------------------------------------------------------
# Training / Evaluation
# -------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y = []
    all_probs = []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(1)

        all_y.append(y.cpu())
        all_probs.append(probs.cpu())

    all_y = torch.cat(all_y).numpy()
    all_probs = torch.cat(all_probs).numpy()

    preds = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_y, preds)
    try:
        auc = roc_auc_score(all_y, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "acc": float(acc),
        "auc": float(auc),
        "y_true": all_y.tolist(),
        "y_prob": all_probs.tolist(),
        "y_pred": preds.tolist(),
    }


def compute_rwr_gap(acc_all, acc_robust, acc_conflict):
    gap_robust = acc_all - acc_robust
    gap_conflict = 1.0 - acc_conflict
    gap = gap_robust + gap_conflict
    return gap_robust, gap_conflict, gap


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[INFO] Using device: {device}")

    if args.exp_name:
        args.out_dir = os.path.join(args.out_dir, args.exp_name)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    loaders = create_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_xrv_densenet(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_acc = 0.0
    best_ckpt = Path(args.out_dir) / "best_model.pth"

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_metrics = evaluate(model, loaders["val"], device)
        scheduler.step()

        print(f"[Train] loss = {train_loss:.4f}")
        print(f"[Val]   acc  = {val_metrics['acc']:.4f}, "
              f"auc = {val_metrics['auc']:.4f}")

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                    "args": vars(args),
                },
                best_ckpt,
            )
            print(f"[INFO] Saved new best model → {best_ckpt}")

    print(f"\n[INFO] Training finished. Best val acc = {best_val_acc:.4f}")

    # -------------------------
    # Test (load best)
    # -------------------------
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    print("\n=== Test Evaluation (XRV Scratch, RWR-Gap) ===")
    easy = evaluate(model, loaders["test_easy"], device)
    robu = evaluate(model, loaders["test_robust"], device)
    conf = evaluate(model, loaders["test_conflict"], device)

    acc_all = easy["acc"]
    acc_robust = robu["acc"]
    acc_conflict = conf["acc"]

    gap_r, gap_c, gap = compute_rwr_gap(acc_all, acc_robust, acc_conflict)

    print(f"[Easy]     acc={acc_all:.4f}, auc={easy['auc']:.4f}")
    print(f"[Robust]   acc={acc_robust:.4f}, auc={robu['auc']:.4f}")
    print(f"[Conflict] acc={acc_conflict:.4f}, auc={conf['auc']:.4f}")
    print()
    print(f"[Gap_robust]   = {gap_r:.4f}")
    print(f"[Gap_conflict] = {gap_c:.4f}")
    print(f"[RWR-Gap]      = {gap:.4f}")

    # Save JSON
    out_json = Path(args.out_dir) / "xrv_scratch_results.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "val_best_acc": best_val_acc,
                "test_easy": easy,
                "test_robust": robu,
                "test_conflict": conf,
                "acc_all": acc_all,
                "acc_robust": acc_robust,
                "acc_conflict": acc_conflict,
                "gap_robust": gap_r,
                "gap_conflict": gap_c,
                "rwr_gap": gap,
            },
            f,
            indent=2,
        )
    print(f"[INFO] Saved results → {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XRV DenseNet (scratch) on CXR RWR-Gap benchmark"
    )

    parser.add_argument("--data_root", type=str, default="data/cxr_rwr_gap")
    parser.add_argument("--out_dir", type=str, default="outputs/xrv_scratch")
    parser.add_argument("--exp_name", type=str, default="xrv_scratch")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
