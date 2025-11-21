#!/usr/bin/env python
"""
train_cxr_rwr_gap_injection.py

Chest X-ray (CXR) RWR-Gap training script with shortcut injection ablations.

This script supports:
  - train_mode:
      * baseline : train on normal + pneumonia
      * shortcut : train on normal + pneumonia_marker
      * injection: train on CSV-mixed dataset with different injection ratios
  - architectures:
      * ResNet-18 / 50 / 152
      * DenseNet-121 / 169 / 201
      * ViT-S/16, ViT-B/16, ViT-L/16  (via timm)
  - pretraining:
      * imagenet
      * scratch

For injection mode, we expect CSV files:
    train_p{p}.csv / val_p{p}.csv
under `split_csv_dir`, each with columns:
    path, label

Expected directory structure (image folders):
    data/cxr_rwr_gap/
        train/
        val/
        test_easy/
        test_robust/
        test_conflict/

Example:
    python train_cxr_rwr_gap_injection.py \
        --data_root data/cxr_rwr_gap \
        --train_mode injection \
        --inject_p 50 \
        --split_csv_dir splits/cxr_injection \
        --arch resnet18 \
        --pretrain imagenet \
        --out_dir outputs/cxr_injection \
        --results_csv outputs/cxr_injection/results_cxr.csv
"""

import argparse
import os
import random
from pathlib import Path
import json
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

import timm  # for ViT models


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Dataset: ImageFolder-based RWR-Gap
#   - normal → 0
#   - others (pneumonia, pneumonia_marker, etc.) → 1
#   - include_classes controls which class folders to use
# -----------------------------
class RWRDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, include_classes=None):
        super().__init__(root, transform=transform)
        if include_classes is not None:
            include_set = {c.lower() for c in include_classes}
            filtered = []
            for path, target in self.samples:
                cls_name = Path(path).parent.name.lower()
                if cls_name in include_set:
                    filtered.append((path, target))
            self.samples = filtered

        print(f"[Dataset] {root} -> {len(self.samples)} images (include={include_classes})")

    def __getitem__(self, index):
        path, _ = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        class_name = Path(path).parent.name.lower()
        if "normal" in class_name:
            target = 0
        else:
            target = 1

        return img, target


# -----------------------------
# Dataset (CSV-based): RWR-Gap with injection strength splits
#   - CSV columns: path, label
# -----------------------------
class RWRCSVDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        # Use the same image loader as ImageFolder
        self.loader = datasets.folder.default_loader

        print(f"[CSVDataset] {self.csv_path} -> {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["label"])

        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label


# -----------------------------
# Model builder
# -----------------------------
def build_model(arch: str, pretrain: str, num_classes: int = 2):
    """
    Build classifier model.

    arch:
      - resnet18, resnet50, resnet152
      - densenet121, densenet169, densenet201
      - vit_s_16, vit_b_16, vit_l_16

    pretrain:
      - imagenet
      - scratch
    """
    arch = arch.lower()
    pretrain = pretrain.lower()

    # 1) ResNet family (torchvision)
    if arch in ["resnet18", "resnet50", "resnet152"]:
        if pretrain == "imagenet":
            weights_map = {
                "resnet18":  models.ResNet18_Weights.IMAGENET1K_V1,
                "resnet50":  models.ResNet50_Weights.IMAGENET1K_V2,
                "resnet152": models.ResNet152_Weights.IMAGENET1K_V2,
            }
            model = getattr(models, arch)(weights=weights_map[arch])
        elif pretrain == "scratch":
            model = getattr(models, arch)(weights=None)
        else:
            raise ValueError(f"Unknown pretrain type: {pretrain}")

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # 2) DenseNet family (torchvision)
    if arch in ["densenet121", "densenet169", "densenet201"]:
        if pretrain == "imagenet":
            weights_map = {
                "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
                "densenet169": models.DenseNet169_Weights.IMAGENET1K_V1,
                "densenet201": models.DenseNet201_Weights.IMAGENET1K_V1,
            }
            model = getattr(models, arch)(weights=weights_map[arch])
        elif pretrain == "scratch":
            model = getattr(models, arch)(weights=None)
        else:
            raise ValueError(f"Unknown pretrain type: {pretrain}")

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    # 3) ViT family (timm)
    vit_map = {
        "vit_s_16": "vit_small_patch16_224",
        "vit_b_16": "vit_base_patch16_224",
        "vit_l_16": "vit_large_patch16_224",
    }
    if arch in vit_map:
        model_name = vit_map[arch]
        if pretrain == "imagenet":
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        elif pretrain == "scratch":
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        else:
            raise ValueError("ViT only supports imagenet/scratch in this script.")
        return model

    raise ValueError(f"Unknown arch: {arch}")


# -----------------------------
# Dataloader builder
# -----------------------------
def create_dataloaders(
    data_root,
    batch_size,
    num_workers,
    train_mode,
    inject_p=None,
    split_csv_dir=None,
):
    """
    Create DataLoaders for train/val/test splits.

    train_mode:
      - baseline : train on [normal, pneumonia]
      - shortcut : train on [normal, pneumonia_marker]
      - injection: use CSV (train_pXX.csv / val_pXX.csv) for mixture
    """
    data_root = Path(data_root)

    # Common transforms
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # -------------------------------
    # Train / Val
    # -------------------------------
    if train_mode in ["baseline", "shortcut"]:
        if train_mode == "baseline":
            include_train = ["normal", "pneumonia"]
        else:  # "shortcut"
            include_train = ["normal", "pneumonia_marker"]

        train_ds = RWRDataset(data_root / "train", transform=train_tf,
                              include_classes=include_train)
        val_ds   = RWRDataset(data_root / "val",   transform=eval_tf,
                              include_classes=include_train)

    elif train_mode == "injection":
        if inject_p is None:
            raise ValueError("train_mode='injection' requires --inject_p.")
        if split_csv_dir is None:
            raise ValueError("train_mode='injection' requires --split_csv_dir.")

        split_csv_dir = Path(split_csv_dir)
        train_csv = split_csv_dir / f"train_p{inject_p}.csv"
        val_csv   = split_csv_dir / f"val_p{inject_p}.csv"

        if not train_csv.exists() or not val_csv.exists():
            raise FileNotFoundError(f"CSV not found: {train_csv}, {val_csv}")

        train_ds = RWRCSVDataset(train_csv, transform=train_tf)
        val_ds   = RWRCSVDataset(val_csv,   transform=eval_tf)

    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # -------------------------------
    # Test splits (always use full splits)
    # -------------------------------
    test_easy_ds      = RWRDataset(data_root / "test_easy",      transform=eval_tf)
    test_robust_ds    = RWRDataset(data_root / "test_robust",    transform=eval_tf)
    test_conflict_ds  = RWRDataset(data_root / "test_conflict",  transform=eval_tf)

    test_easy_loader = DataLoader(
        test_easy_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_robust_loader = DataLoader(
        test_robust_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_conflict_loader = DataLoader(
        test_conflict_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test_easy": test_easy_loader,
        "test_robust": test_robust_loader,
        "test_conflict": test_conflict_loader,
    }
    return loaders


# -----------------------------
# Train / Eval loops
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]  # class 1 prob

        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "acc": float(acc),
        "auc": float(auc),
        "y_true": all_labels.tolist(),
        "y_prob": all_probs.tolist(),
        "y_pred": preds.tolist(),
    }


# -----------------------------
# CSV logger
# -----------------------------
def append_result_row(args, acc_all, acc_robust, acc_conflict, rwr_gap, ckpt_path):
    """Append one experiment summary row into results CSV."""
    if args.results_csv is None:
        return

    row = {
        "arch": args.arch,
        "pretrain": args.pretrain,
        "train_mode": args.train_mode,
        "seed": args.seed,
        "inject_p": args.inject_p,
        "acc_all": acc_all,
        "acc_robust": acc_robust,
        "acc_conflict": acc_conflict,
        "rwr_gap": rwr_gap,
        "ckpt_path": ckpt_path,
        "notes": args.notes,
    }

    csv_path = args.results_csv
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[Info] Appended results to {csv_path}")


# -----------------------------
# Main
# -----------------------------
def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[Info] Using device: {device}")

    # Append exp_name as subfolder of out_dir
    if args.exp_name:
        args.out_dir = os.path.join(args.out_dir, args.exp_name)

    os.makedirs(args.out_dir, exist_ok=True)

    # Dataloaders
    loaders = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_mode=args.train_mode,
        inject_p=args.inject_p,
        split_csv_dir=args.split_csv_dir,
    )

    # Model / Loss / Optimizer
    model = build_model(arch=args.arch, pretrain=args.pretrain, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.out_dir, "best_model.pth")

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
        print(f"[Val]   acc  = {val_metrics['acc']:.4f}, auc = {val_metrics['auc']:.4f}")

        # Save best model based on validation accuracy
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"[Info] New best model saved at {best_ckpt_path}")

    print(f"\n[Info] Training finished. Best val acc = {best_val_acc:.4f}")

    # -------------------------
    # Load best checkpoint and evaluate on test splits
    # -------------------------
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    print("\n=== Test Evaluation (RWR-Gap) ===")
    easy_metrics = evaluate(model, loaders["test_easy"], device)
    robust_metrics = evaluate(model, loaders["test_robust"], device)
    conflict_metrics = evaluate(model, loaders["test_conflict"], device)

    acc_all = easy_metrics["acc"]
    acc_robust = robust_metrics["acc"]
    acc_conflict = conflict_metrics["acc"]

    gap_robust = acc_all - acc_robust
    gap_conflict = 1.0 - acc_conflict
    rwr_gap = gap_robust + gap_conflict

    print(f"[Test-Easy]     Acc_all     = {acc_all:.4f}, AUC = {easy_metrics['auc']:.4f}")
    print(f"[Test-Robust]   Acc_robust  = {acc_robust:.4f}, AUC = {robust_metrics['auc']:.4f}")
    print(f"[Test-Conflict] Acc_conflict= {acc_conflict:.4f}, AUC = {conflict_metrics['auc']:.4f}")
    print()
    print(f"[Gap_robust]   = Acc_all - Acc_robust       = {gap_robust:.4f}")
    print(f"[Gap_conflict] = 1 - Acc_conflict           = {gap_conflict:.4f}")
    print(f"[RWR-Gap]      = Gap_robust + Gap_conflict = {rwr_gap:.4f}")

    # Save JSON
    results = {
        "val_best_acc": best_val_acc,
        "test_easy": easy_metrics,
        "test_robust": robust_metrics,
        "test_conflict": conflict_metrics,
        "acc_all": acc_all,
        "acc_robust": acc_robust,
        "acc_conflict": acc_conflict,
        "rwr_gap": rwr_gap,
    }

    out_json = os.path.join(args.out_dir, "rwr_gap_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Info] Saved RWR-Gap results to {out_json}")

    # Append one row to results CSV
    append_result_row(args, acc_all, acc_robust, acc_conflict, rwr_gap, best_ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CXR RWR-Gap training with shortcut injection ablations."
    )

    # Paths
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/cxr_rwr_gap",
        help="Root directory of CXR RWR-Gap dataset.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/cxr_injection",
        help="Base output directory.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Optional experiment name (subfolder under out_dir).",
    )

    # Model
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=[
            "resnet18", "resnet50", "resnet152",
            "densenet121", "densenet169", "densenet201",
            "vit_s_16", "vit_b_16", "vit_l_16",
        ],
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default="imagenet",
        choices=["imagenet", "scratch"],
        help="Pretraining type (CXR-pretrained can be added separately).",
    )

    # Training mode / injection
    parser.add_argument(
        "--train_mode",
        type=str,
        default="shortcut",
        choices=["baseline", "shortcut", "injection"],
        help=(
            "baseline : train on normal + pneumonia\n"
            "shortcut : train on normal + pneumonia_marker\n"
            "injection: use CSV mixtures (train_pXX / val_pXX)"
        ),
    )
    parser.add_argument(
        "--inject_p",
        type=int,
        default=0,
        help="Injection strength (0/25/50/75/100). Used when train_mode='injection'.",
    )
    parser.add_argument(
        "--split_csv_dir",
        type=str,
        default="splits/cxr_injection",
        help="Directory containing train_pXX.csv / val_pXX.csv for injection mode.",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument(
        "--results_csv",
        type=str,
        default="outputs/cxr_injection/results_cxr.csv",
        help="CSV path to accumulate experiment summaries.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes (e.g., dataset version, comments).",
    )

    args = parser.parse_args()
    main(args)
