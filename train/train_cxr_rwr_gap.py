#!/usr/bin/env python
"""
train_cxr_rwr_gap.py

Main training script for Chest X-ray (CXR) RWR-Gap experiments.

This script supports:
    - Baseline training (no marker)
    - Shortcut-injected training (marker injected into pneumonia class)
    - Architecture sweep:
          ResNet-18 / 50 / 152
          DenseNet-121 / 169 / 201
          ViT-S/16, ViT-B/16, ViT-L/16
    - Pretraining:
          - imagenet  (torchvision)
          - scratch   (random initialization)
          - cxr       (torchxrayvision pretrained)
    - Evaluation on:
          test_easy / test_robust / test_conflict

The script saves:
    - best checkpoint (based on validation accuracy)
    - CSV logging results
    - JSON summary

Expected directory structure:
    data/cxr_rwr_gap/
        train/
        val/
        test_easy/
        test_robust/
        test_conflict/

Example:
    python train_cxr_rwr_gap.py \
        --train_mode baseline \
        --arch resnet50 \
        --pretrain imagenet \
        --data_root data/cxr_rwr_gap \
        --results_csv outputs/cxr/results_main.csv
"""

import argparse
import os
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

try:
    import torchxrayvision as xrv
except ImportError:
    pass


# ===============================================================
# Utilities
# ===============================================================
def set_seed(seed: int = 42):
    """Set global random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Return appropriate CUDA/CPU device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================================================
# Dataset:  normal → 0, everything else → 1
# ===============================================================
class RWRDataset(datasets.ImageFolder):
    """ImageFolder wrapper for RWR-Gap (CXR)."""

    def __getitem__(self, index):
        path, _ = self.samples[index]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)

        parent = Path(path).parent.name.lower()
        label = 0 if "normal" in parent else 1
        return img, label


# ===============================================================
# Model builder
# ===============================================================
def build_model(arch: str, pretrain: str, num_classes=2):
    """
    Build various architectures:
        resnet18, resnet50, resnet152
        densenet121, densenet169, densenet201
        vit_s16, vit_b16, vit_l16
        cxr (torchxrayvision pretrained ResNet50)
    """
    arch = arch.lower()

    # -----------------------------------------------------------
    # CXR pretrained ResNet50 (torchxrayvision)
    # -----------------------------------------------------------
    if pretrain == "cxr":
        if arch != "resnet50":
            raise ValueError("CXR pretrained model only supports ResNet50.")
        model = xrv.models.ResNet50(weights="mimic_nb")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # -----------------------------------------------------------
    # ResNet family
    # -----------------------------------------------------------
    if arch == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrain == "imagenet" else None
        )
    elif arch == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrain == "imagenet" else None
        )
    elif arch == "resnet152":
        model = models.resnet152(
            weights=models.ResNet152_Weights.IMAGENET1K_V2 if pretrain == "imagenet" else None
        )
    # -----------------------------------------------------------
    # DenseNet family
    # -----------------------------------------------------------
    elif arch == "densenet121":
        model = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrain == "imagenet" else None
        )
    elif arch == "densenet169":
        model = models.densenet169(
            weights=models.DenseNet169_Weights.IMAGENET1K_V1 if pretrain == "imagenet" else None
        )
    elif arch == "densenet201":
        model = models.densenet201(
            weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrain == "imagenet" else None
        )
    # -----------------------------------------------------------
    # ViT family
    # -----------------------------------------------------------
    elif arch in ["vit_s16", "vit_b16", "vit_l16"]:
        import timm
        name = {
            "vit_s16": "vit_small_patch16_224",
            "vit_b16": "vit_base_patch16_224",
            "vit_l16": "vit_large_patch16_224",
        }[arch]
        model = timm.create_model(
            name,
            pretrained=(pretrain == "imagenet"),
            num_classes=num_classes,
        )
        return model
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    # Replace FC for ResNet/DenseNet
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    if hasattr(model, "classifier"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


# ===============================================================
# Data loaders
# ===============================================================
def create_dataloaders(data_root, batch_size, num_workers, pretrain):
    """
    Create train/val/test dataloaders with preprocessing depending on pretrain type.
    """
    data_root = Path(data_root)

    if pretrain == "cxr":
        train_tf = xrv.datasets.XRayCenterCrop()
        eval_tf  = xrv.datasets.XRayCenterCrop()

    else:
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

        eval_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    # dataset objects
    train_ds = RWRDataset(data_root/"train", transform=train_tf)
    val_ds   = RWRDataset(data_root/"val",   transform=eval_tf)
    test_easy     = RWRDataset(data_root/"test_easy",     transform=eval_tf)
    test_robust   = RWRDataset(data_root/"test_robust",   transform=eval_tf)
    test_conflict = RWRDataset(data_root/"test_conflict", transform=eval_tf)

    # loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    test_easy_loader = DataLoader(test_easy, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
    test_robust_loader = DataLoader(test_robust, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
    test_conflict_loader = DataLoader(test_conflict, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)

    return {
        "train": train_loader,
        "val": val_loader,
        "test_easy": test_easy_loader,
        "test_robust": test_robust_loader,
        "test_conflict": test_conflict_loader,
    }


# ===============================================================
# Training loop
# ===============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    """Return {acc, auc, y_true, y_prob, y_pred}."""
    model.eval()
    labels = []
    probs  = []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:,1]

        labels.append(y.cpu())
        probs.append(p.cpu())

    labels = torch.cat(labels).numpy()
    probs = torch.cat(probs).numpy()

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = float("nan")

    return {
        "acc": float(acc),
        "auc": float(auc),
        "y_true": labels.tolist(),
        "y_prob": probs.tolist(),
        "y_pred": preds.tolist(),
    }


def compute_rwr_gap(acc_all, acc_robust, acc_conflict):
    """Compute (gap_robust, gap_conflict, rwr_gap)."""
    gap_robust = acc_all - acc_robust
    gap_conflict = 1 - acc_conflict
    return gap_robust, gap_conflict, gap_robust + gap_conflict


# ===============================================================
# Main
# ===============================================================
def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[Device] {device}")

    # output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # log file
    results_csv = Path(args.results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    if not results_csv.exists():
        with open(results_csv, "w") as f:
            f.write("arch,train_mode,pretrain,seed,acc_all,acc_robust,acc_conflict,rwr_gap,out_dir\n")

    # dataloaders
    loaders = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrain=args.pretrain,
    )

    # model
    model = build_model(
        arch=args.arch,
        pretrain=args.pretrain,
        num_classes=2
    ).to(device)

    # loss/optim
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # checkpoint
    best_acc = 0
    ckpt = out_dir/"best_model.pth"

    # ---------------------------------------------
    # Train
    # ---------------------------------------------
    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate_classifier(model, loaders["val"], device)

        print(f"[Train] loss={train_loss:.4f}")
        print(f"[Val] acc={val_metrics['acc']:.4f}, auc={val_metrics['auc']:.4f}")

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
            print(f"[Checkpoint] Saved best model → {ckpt}")

        scheduler.step()

    # ---------------------------------------------
    # Evaluate best model on test splits
    # ---------------------------------------------
    print("\n[Testing best checkpoint]")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])

    easy = evaluate_classifier(model, loaders["test_easy"], device)
    robu = evaluate_classifier(model, loaders["test_robust"], device)
    conf = evaluate_classifier(model, loaders["test_conflict"], device)

    acc_all = easy["acc"]
    acc_robust = robu["acc"]
    acc_conflict = conf["acc"]
    gap_r, gap_c, rwr_gap = compute_rwr_gap(acc_all, acc_robust, acc_conflict)

    print(f"[Test] Acc_all={acc_all:.4f}, Acc_robust={acc_robust:.4f}, Acc_conflict={acc_conflict:.4f}")
    print(f"[RWR-Gap] {rwr_gap:.4f}")

    # CSV append
    with open(results_csv, "a") as f:
        f.write(f"{args.arch},{args.train_mode},{args.pretrain},{args.seed},"
                f"{acc_all:.4f},{acc_robust:.4f},{acc_conflict:.4f},{rwr_gap:.4f},{out_dir}\n")

    # JSON save
    info = {
        "arch": args.arch,
        "pretrain": args.pretrain,
        "train_mode": args.train_mode,
        "acc_all": acc_all,
        "acc_robust": acc_robust,
        "acc_conflict": acc_conflict,
        "rwr_gap": rwr_gap,
    }

    with open(out_dir/"summary.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"[Saved JSON] {out_dir/'summary.json'}")


# ===============================================================
# Args
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CXR RWR-Gap main training script.")

    parser.add_argument("--data_root", type=str, default="data/cxr_rwr_gap")
    parser.add_argument("--out_dir", type=str, default="outputs/cxr_main")
    parser.add_argument("--results_csv", type=str, default="outputs/cxr_main/results.csv")

    parser.add_argument("--train_mode", type=str, default="baseline",
                        choices=["baseline", "shortcut"])

    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=[
                            "resnet18","resnet50","resnet152",
                            "densenet121","densenet169","densenet201",
                            "vit_s16","vit_b16","vit_l16"
                        ])

    parser.add_argument("--pretrain", type=str, default="imagenet",
                        choices=["imagenet","scratch","cxr"])

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
