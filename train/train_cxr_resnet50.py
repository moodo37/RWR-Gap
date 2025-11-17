
"""
Train and evaluate a ResNet-50 classifier on the CXR RWR-Gap benchmark.

This script:
  - Trains a binary classifier (normal vs. pneumonia/marker)
  - Uses grayscale CXR images converted to 3 channels
  - Supports both ImageNet-pretrained and scratch initialization
  - Evaluates on three test splits: Test-Easy / Test-Robust / Test-Conflict
  - Computes the RWR-Gap metric used in the paper

Expected directory structure:

    data/cxr_rwr_gap/
        train/
            normal/
            pneumonia_marker/    # or pneumonia, pneumonia_marker, etc.
        val/
            normal/
            pneumonia_marker/
        test_easy/
            normal/
            pneumonia_marker/
        test_robust/
            pneumonia/
        test_conflict/
            normal/

You can change `--data_root` to match your local path.
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
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


# Reuse the same helpers / dataset pattern as ISIC
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RWRDataset(datasets.ImageFolder):
    """
    ImageFolder-based dataset for RWR-Gap experiments.

    Label mapping:
        * any folder name containing 'normal' → 0
        * otherwise → 1
    """

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


def build_resnet50(num_classes: int = 2, weights: str = "imagenet"):
    """
    Build a ResNet-50 classifier for CXR.

    weights:
        - "imagenet": ImageNet-1K pretrained
        - "scratch" : random init
    """
    weights = weights.lower()
    if weights == "imagenet":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
    elif weights == "scratch":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Invalid weights: {weights}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def create_dataloaders(data_root: str, batch_size: int, num_workers: int):
    """
    Create dataloaders for CXR RWR-Gap splits.

    Note:
        - Input is grayscale CXR, converted to 3 channels.
        - Normalized with ImageNet statistics.
    """
    data_root = Path(data_root)

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

    train_ds = RWRDataset(data_root / "train", transform=train_tf)
    val_ds   = RWRDataset(data_root / "val",   transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_easy_ds      = RWRDataset(data_root / "test_easy",     transform=eval_tf)
    test_robust_ds    = RWRDataset(data_root / "test_robust",   transform=eval_tf)
    test_conflict_ds  = RWRDataset(data_root / "test_conflict", transform=eval_tf)

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

    return {
        "train": train_loader,
        "val": val_loader,
        "test_easy": test_easy_loader,
        "test_robust": test_robust_loader,
        "test_conflict": test_conflict_loader,
    }


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

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]

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


def compute_rwr_gap(acc_all: float, acc_robust: float, acc_conflict: float):
    gap_robust = acc_all - acc_robust
    gap_conflict = 1.0 - acc_conflict
    rwr_gap = gap_robust + gap_conflict
    return gap_robust, gap_conflict, rwr_gap


def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[Info] Using device: {device}")
    print(f"[Info] ResNet-50 weights: {args.weights}")

    if args.exp_name:
        args.out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.out_dir, exist_ok=True)

    loaders = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_resnet50(num_classes=2, weights=args.weights).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.out_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_metrics = evaluate_classifier(model, loaders["val"], device)
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
                best_ckpt_path,
            )
            print(f"[Info] New best model saved at {best_ckpt_path}")

    print(f"\n[Info] Training finished. Best val acc = {best_val_acc:.4f}")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    print("\n=== Test Evaluation (RWR-Gap, CXR) ===")
    easy_metrics = evaluate_classifier(model, loaders["test_easy"], device)
    robust_metrics = evaluate_classifier(model, loaders["test_robust"], device)
    conflict_metrics = evaluate_classifier(model, loaders["test_conflict"], device)

    acc_all = easy_metrics["acc"]
    acc_robust = robust_metrics["acc"]
    acc_conflict = conflict_metrics["acc"]

    gap_robust, gap_conflict, rwr_gap = compute_rwr_gap(
        acc_all, acc_robust, acc_conflict
    )

    print(f"[Test-Easy]     Acc_all      = {acc_all:.4f}, "
          f"AUC = {easy_metrics['auc']:.4f}")
    print(f"[Test-Robust]   Acc_robust   = {acc_robust:.4f}, "
          f"AUC = {robust_metrics['auc']:.4f}")
    print(f"[Test-Conflict] Acc_conflict = {acc_conflict:.4f}, "
          f"AUC = {conflict_metrics['auc']:.4f}")
    print()
    print(f"[Gap_robust]   = Acc_all - Acc_robust      = {gap_robust:.4f}")
    print(f"[Gap_conflict] = 1 - Acc_conflict          = {gap_conflict:.4f}")
    print(f"[RWR-Gap]      = Gap_robust + Gap_conflict = {rwr_gap:.4f}")

    results = {
        "weights": args.weights,
        "val_best_acc": best_val_acc,
        "test_easy": easy_metrics,
        "test_robust": robust_metrics,
        "test_conflict": conflict_metrics,
        "acc_all": acc_all,
        "acc_robust": acc_robust,
        "acc_conflict": acc_conflict,
        "gap_robust": gap_robust,
        "gap_conflict": gap_conflict,
        "rwr_gap": rwr_gap,
    }

    out_json = os.path.join(args.out_dir, "rwr_gap_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Info] Saved RWR-Gap results to {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RWR-Gap Training & Evaluation (CXR, ResNet-50)"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data/cxr_rwr_gap",
        help="Root directory of the CXR RWR-Gap dataset.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/cxr_resnet50",
        help="Base directory to save checkpoints and results.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="resnet50",
        help="Sub-folder name under out_dir."
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="imagenet",
        choices=["scratch", "imagenet"],
        help="Initialization for ResNet-50.",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
