
"""
Generate Grad-CAM triplets (original, scratch CAM, pretrained CAM)
for ISIC 2019 test images, and compute CAM energy statistics.

This script:
  1) Loads two ResNet152 models (scratch vs ImageNet-pretrained).
  2) For each image in a given split (e.g., test_conflict),
     computes Grad-CAM for both models.
  3) Measures CAM energy in marker ROI and lesion ROI.
  4) Saves:
      - cam_energy_stats_isic.csv
      - Top-N triplet images (original + 2 heatmaps).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms

from rwr_gap.analysis.gradcam_core import GradCAM


# --------------------------
# ROI & energy utils
# --------------------------
def marker_and_lesion_rois(H, W):
    """
    ISIC markers: often large artifacts on top-right.

    marker_roi: top 25% × right 25%
    lesion_roi: central rectangle
    """
    y1_m, y2_m = 0, int(0.25 * H)
    x1_m, x2_m = int(0.75 * W), W
    marker_roi = (y1_m, y2_m, x1_m, x2_m)

    y1_l, y2_l = int(0.20 * H), int(0.85 * H)
    x1_l, x2_l = int(0.15 * W), int(0.85 * W)
    lesion_roi = (y1_l, y2_l, x1_l, x2_l)
    return marker_roi, lesion_roi


def cam_energy_ratio(cam, roi):
    (y1, y2, x1, x2) = roi
    patch = cam[y1:y2, x1:x2]
    return float(patch.mean() / (cam.mean() + 1e-8))


def show_triplet(img_np, cam1, cam2, titles, alpha=0.55):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(img_np)
    axes[0].set_title(titles[0])
    axes[1].imshow(img_np)
    axes[1].imshow(cam1, cmap="jet", alpha=alpha)
    axes[1].set_title(titles[1])
    axes[2].imshow(img_np)
    axes[2].imshow(cam2, cmap="jet", alpha=alpha)
    axes[2].set_title(titles[2])
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def build_resnet152(num_classes, ckpt_path, device):
    model = models.resnet152(weights=None)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]
    model.load_state_dict(sd, strict=False)

    model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Path to split directory, e.g. data/ISIC/test_conflict")
    ap.add_argument("--ckpt_scratch", required=True,
                    help="Checkpoint path for scratch-trained ResNet152.")
    ap.add_argument("--ckpt_pretrained", required=True,
                    help="Checkpoint path for ImageNet-pretrained ResNet152.")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=32,
                    help="Maximum number of samples to process.")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.55)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(args.data_root, transform=tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print("class_to_idx:", ds.class_to_idx)

    # Models
    model_s = build_resnet152(args.num_classes, args.ckpt_scratch, device)
    model_p = build_resnet152(args.num_classes, args.ckpt_pretrained, device)

    cam_s = GradCAM(model_s, target_layer="layer4")
    cam_p = GradCAM(model_p, target_layer="layer4")

    all_stats = []
    candidates = []

    for i, (x, y) in enumerate(loader):
        if i >= args.num_samples:
            break

        x = x.to(device)

        # CAMs
        cam1 = cam_s.generate(x.clone())  # scratch
        cam2 = cam_p.generate(x.clone())  # pretrained

        # original image (unnormalize for visualization)
        x_vis = x[0].detach().cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        x_vis = x_vis * std + mean
        x_vis = np.clip(x_vis, 0, 1)
        img_np = np.transpose(x_vis, (1, 2, 0))
        H, W, _ = img_np.shape

        marker_roi, lesion_roi = marker_and_lesion_rois(H, W)
        k_m_s = cam_energy_ratio(cam1, marker_roi)
        k_m_p = cam_energy_ratio(cam2, marker_roi)
        k_l_s = cam_energy_ratio(cam1, lesion_roi)
        k_l_p = cam_energy_ratio(cam2, lesion_roi)

        score = k_m_s + k_l_p - k_m_p
        img_path, label = ds.samples[i]

        all_stats.append({
            "idx": i,
            "path": img_path,
            "label": int(label),
            "k_m_s": float(k_m_s),
            "k_m_p": float(k_m_p),
            "k_l_s": float(k_l_s),
            "k_l_p": float(k_l_p),
            "score": float(score),
        })
        candidates.append((float(score), img_np, cam1, cam2))

    # Save CSV
    df = pd.DataFrame(all_stats)
    csv_path = out_dir / "cam_energy_stats_isic.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved CAM energy stats → {csv_path}")

    # Save Top-N triplets
    candidates.sort(key=lambda t: t[0], reverse=True)
    keep = candidates[:min(len(candidates), args.num_samples)]
    for idx, (sc, img_np, cam1, cam2) in enumerate(keep):
        fig = show_triplet(
            img_np, cam1, cam2,
            titles=(
                f"Original (S={sc:.2f})",
                "Scratch CAM",
                "Pretrained CAM",
            ),
            alpha=args.alpha,
        )
        fig.savefig(out_dir / f"isic_topN_{idx:03d}.png", dpi=170)
        plt.close(fig)
    print(f"[INFO] Saved {len(keep)} triplet images → {out_dir}")


if __name__ == "__main__":
    main()
