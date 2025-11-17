
"""
Generate Grad-CAM triplets (scratch vs pretrained) on CXR images.

- Input  : ImageFolder-style directory (e.g., RWR-Gap test_conflict / test_robust)
- Output : Triplet figures + per-image CAM energy stats (CSV)
- Models : scratch checkpoint vs pretrained (torchvision or XRV DenseNet)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision.models as tvm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 공통 Grad-CAM 모듈
from rwr_gap.analysis.gradcam_core import GradCAM

# (optional) XRV weights
try:
    import torchxrayvision as xrv
    HAS_XRV = True
except Exception:
    HAS_XRV = False


# ------------------------------------------------
# DenseNet helper (arbitrary input channels)
# ------------------------------------------------
def build_densenet121_inch(in_ch: int,
                           num_classes: int,
                           ckpt_path: str = None,
                           return_logits: bool = True) -> nn.Module:
    """
    Build DenseNet-121 with arbitrary input channels and num_classes.
    Used mainly for scratch CXR models where input is 1ch.
    """
    m = tvm.densenet121(weights=None)

    # 1) change input conv
    if in_ch != 3:
        w = m.features.conv0.weight
        m.features.conv0 = nn.Conv2d(
            in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if in_ch == 1:
            # copy mean of RGB filters to single channel
            with torch.no_grad():
                m.features.conv0.weight.copy_(w.mean(dim=1, keepdim=True))

    # 2) classification head
    m.classifier = nn.Linear(1024, num_classes)

    # 3) load checkpoint (if provided)
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(sd, strict=False)

    # 4) optionally wrap with Sigmoid (we usually want logits)
    if return_logits:
        return m

    class Wrap(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, x):
            return torch.sigmoid(self.core(x))

    return Wrap(m)


def to_xrv_input(x3: torch.Tensor) -> torch.Tensor:
    """
    Convert (B,3,H,W) tensor in [-1,1] (or [0,1]) to XRV-style 1ch input.

    XRV expects approximately [-1024, 1024] range:
        x_norm = (2*(x/maxval) - 1) * 1024
    Here we assume maxval=1.
    """
    # undo [-1,1] → [0,1] if necessary
    if x3.min() < 0:
        x3 = (x3 + 1.0) / 2.0

    # 3ch → 1ch mean
    x1 = x3.mean(dim=1, keepdim=True)   # (B,1,H,W), [0,1]

    # XRV normalization
    x1 = (2.0 * x1 - 1.0) * 1024.0
    return x1.float()


# ------------------------------------------------
# CAM ROI utilities
# ------------------------------------------------
def corner_rois(H: int, W: int, frac: float = 0.22):
    """Return two corner ROIs (UL / UR) with given size fraction."""
    h = int(H * frac)
    w = int(W * frac)
    # (y1, y2, x1, x2)
    return [
        (0, h, 0, w),        # upper-left
        (0, h, W - w, W),    # upper-right
    ]


def cam_energy_ratio(cam: np.ndarray, roi):
    """Mean activation inside ROI / global mean."""
    (y1, y2, x1, x2) = roi
    patch = cam[y1:y2, x1:x2]
    return float(patch.mean() / (cam.mean() + 1e-8))


def make_rois(h: int, w: int):
    """
    Heuristic ROIs:
      - marker_roi : upper-right corner
      - lung_roi   : central lung region
    """
    # marker (top-right)
    y1_m, y2_m = 0, int(0.25 * h)
    x1_m, x2_m = int(0.75 * w), w
    marker_roi = (y1_m, y2_m, x1_m, x2_m)

    # lung (center-ish)
    y1_l, y2_l = int(0.25 * h), int(0.85 * h)
    x1_l, x2_l = int(0.15 * w), int(0.85 * w)
    lung_roi = (y1_l, y2_l, x1_l, x2_l)

    return marker_roi, lung_roi


def best_marker_roi(cam: np.ndarray):
    """
    Among 4 corners (UL, UR, LL, LR), find the corner with maximum CAM energy.
    """
    H, W = cam.shape
    rois = [
        (0, int(0.25 * H), int(0.75 * W), W),        # UR
        (0, int(0.25 * H), 0, int(0.25 * W)),        # UL
        (int(0.75 * H), H, int(0.75 * W), W),        # LR
        (int(0.75 * H), H, 0, int(0.25 * W)),        # LL
    ]
    vals = []
    m = cam.mean() + 1e-8
    for (y1, y2, x1, x2) in rois:
        vals.append(cam[y1:y2, x1:x2].mean() / m)
    return rois[int(np.argmax(vals))]


# ------------------------------------------------
# Infer ckpt spec (for transform/model)
# ------------------------------------------------
def infer_out_dim_from_ckpt(ckpt_path: str):
    """Infer output dimension from classifier weight shape."""
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]
    for k, v in sd.items():
        if k.endswith(("classifier.weight", "fc.weight")):
            return v.shape[0]
    return None


def infer_in_ch_from_ckpt(ckpt_path: str):
    """
    Infer input channels from first conv weight.
    Works for torchvision DenseNet / ResNet style checkpoints.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]
    for k, v in sd.items():
        if k.endswith(("features.conv0.weight", "conv1.weight", "features.0.weight")):
            # v.shape = (out_ch, in_ch, k, k)
            return v.shape[1]
    return 3  # default assume 3ch


# ------------------------------------------------
# Model builder (scratch / pretrained / XRV)
# ------------------------------------------------
def build_model(arch: str,
                num_classes: int,
                ckpt_path: str = None,
                xrv_weights: str = None) -> nn.Module:
    """
    Generic model builder.

    - If xrv_weights is given → XRV DenseNet121
    - Otherwise torchvision resnet18 / resnet50 / densenet121
    """
    arch = arch.lower()

    # XRV branch
    if xrv_weights:
        if not HAS_XRV:
            raise RuntimeError(
                "torchxrayvision is required. Please `pip install torchxrayvision`."
            )
        model = xrv.models.DenseNet(weights=xrv_weights)

        # replace classification head if needed
        if getattr(model, "n_classes", None) != num_classes:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes, bias=True)
            model.n_classes = num_classes

        # op_threshs buffer length = num_classes
        new_thresh = torch.full((num_classes,), float("nan"))
        model.register_buffer("op_threshs", new_thresh)

        try:
            model.pathologies = [f"class_{i}" for i in range(num_classes)]
        except Exception:
            pass

        if ckpt_path and Path(ckpt_path).exists():
            sd = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(sd, strict=False)

        # for CAM target layer
        model._rwr_target_layer = "features"
        return model

    # torchvision branch
    if arch == "resnet18":
        model = tvm.resnet18(weights=None, num_classes=num_classes)
        target_default = "layer4"
    elif arch == "resnet50":
        model = tvm.resnet50(weights=None, num_classes=num_classes)
        target_default = "layer4"
    elif arch == "densenet121":
        model = tvm.densenet121(weights=None, num_classes=num_classes)
        target_default = "features"
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "model_state" in sd:
            sd = sd["model_state"]
        model.load_state_dict(sd, strict=False)

    model._rwr_target_layer = target_default
    return model


# ------------------------------------------------
# Visualization utilities
# ------------------------------------------------
def to_uint01(img: np.ndarray) -> np.ndarray:
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def show_triplet(img_np: np.ndarray,
                 cam1: np.ndarray,
                 cam2: np.ndarray,
                 titles=("Original", "Scratch CAM", "Pretrained CAM"),
                 alpha: float = 0.55):
    """
    Draw original + 2 CAM overlays.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title(titles[0])

    axes[1].imshow(img_np, cmap="gray")
    axes[1].imshow(cam1, cmap="jet", alpha=alpha)
    axes[1].set_title(titles[1])

    axes[2].imshow(img_np, cmap="gray")
    axes[2].imshow(cam2, cmap="jet", alpha=alpha)
    axes[2].set_title(titles[2])

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    return fig


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="e.g., /path/to/RWR-Gap/data/test_conflict",
    )
    ap.add_argument(
        "--arch",
        type=str,
        default="densenet121",
        choices=["resnet18", "resnet50", "densenet121"],
    )
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--ckpt_scratch", type=str, required=True,
                    help="scratch-trained checkpoint path")
    ap.add_argument(
        "--ckpt_pretrained",
        type=str,
        default="",
        help="(optional) pretrained fine-tuned checkpoint path",
    )
    ap.add_argument(
        "--xrv_weights",
        type=str,
        default="",
        help="e.g., densenet121-res224-mimic_nb (if you use XRV as pretrained)",
    )
    ap.add_argument(
        "--target_layer",
        type=str,
        default="",
        help="explicit target layer for Grad-CAM (if empty, use model._rwr_target_layer)",
    )
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument(
        "--target_cls",
        type=int,
        default=-1,
        help="fixed CAM class index; -1 → use Pneumonia index from dataset",
    )
    ap.add_argument("--alpha", type=float, default=0.55,
                    help="heatmap transparency")
    ap.add_argument(
        "--max_seen",
        type=int,
        default=400,
        help="upper bound on scanned pneumonia samples",
    )

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) infer scratch spec BEFORE building transform
    in_ch_s = infer_in_ch_from_ckpt(args.ckpt_scratch) or 3
    out_dim_s = infer_out_dim_from_ckpt(args.ckpt_scratch) or args.num_classes
    print(f"[scratch] inferred in_ch={in_ch_s}, out_dim={out_dim_s}")

    # 2) transform (depends on input channels)
    if in_ch_s == 1:
        tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
    else:
        tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5]),
            ]
        )

    # 3) dataset / dataloader
    ds = datasets.ImageFolder(args.data_root, transform=tf)
    print("class_to_idx:", ds.class_to_idx)
    pneu_idx = ds.class_to_idx.get("pneumonia", 1)  # default 1 if not found

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 4) models (scratch & pretrained)
    if args.arch == "densenet121":
        model_s = build_densenet121_inch(
            in_ch_s, out_dim_s, ckpt_path=args.ckpt_scratch, return_logits=True
        )
    else:
        model_s = build_model(args.arch, out_dim_s, ckpt_path=args.ckpt_scratch)

    if args.xrv_weights:
        model_p = build_model(
            args.arch,
            args.num_classes,
            ckpt_path=args.ckpt_pretrained if args.ckpt_pretrained else None,
            xrv_weights=args.xrv_weights,
        )
    else:
        model_p = build_model(args.arch,
                              args.num_classes,
                              ckpt_path=args.ckpt_pretrained)

    model_s.to(device).eval()
    model_p.to(device).eval()

    if args.xrv_weights:
        print("[XRV] pathologies:", getattr(model_p, "pathologies", None))

    # 5) Grad-CAM objects (여기가 gradcam_core 사용 포인트!)
    tgt_layer_s = (
        args.target_layer
        if args.target_layer
        else getattr(model_s, "_rwr_target_layer", "features")
    )
    tgt_layer_p = getattr(model_p, "_rwr_target_layer", tgt_layer_s)

    cam_s = GradCAM(model_s, target_layer=tgt_layer_s)
    cam_p = GradCAM(model_p, target_layer=tgt_layer_p)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # we usually want Pneumonia index
    target_idx = pneu_idx
    seen = 0
    selected = 0
    candidates = []
    all_stats = []

    for i, (x, y) in enumerate(loader):
        seen += 1
        if seen > args.max_seen and selected < args.num_samples:
            print(f"[STOP] scanned {seen} pneumonia samples, selected {selected}")
            break

        x = x.to(device)

        # prepare input for pretrained XRV if needed
        if args.xrv_weights:
            x_p = to_xrv_input(x).to(device)
        else:
            x_p = x

        # 1) choose class indices
        fixed_cls_s = (
            0
            if out_dim_s == 1
            else (target_idx if args.target_cls < 0 else args.target_cls)
        )

        if args.xrv_weights and hasattr(model_p, "pathologies"):
            # use Pneumonia index in XRV pathologies
            pneu_idx_xrv = model_p.pathologies.index("Pneumonia")
            fixed_cls_p = pneu_idx_xrv
        else:
            fixed_cls_p = fixed_cls_s

        # 2) Grad-CAM generation
        cam1 = cam_s.generate(x.clone(), class_idx=fixed_cls_s)
        cam2 = cam_p.generate(x_p.clone(), class_idx=fixed_cls_p)

        # 3) original grayscale image (for visualization)
        img_np = x[0].detach().cpu().numpy()  # (C,H,W)
        img_np = np.transpose(img_np, (1, 2, 0))  # (H,W,C)
        img_np = to_uint01(img_np)
        if img_np.ndim == 3 and img_np.shape[2] > 1:
            img_np = img_np.mean(axis=2)
        else:
            img_np = img_np.squeeze()
        H, W = img_np.shape

        # 4) CAM energy (marker / lung)
        corners = corner_rois(H, W)

        def max_corner_energy(cam):
            return max(cam_energy_ratio(cam, roi) for roi in corners)

        k_m_s = max_corner_energy(cam1)
        k_m_p = max_corner_energy(cam2)

        y1_l, y2_l = int(0.20 * H), int(0.85 * H)
        x1_l, x2_l = int(0.15 * W), int(0.85 * W)
        lung_roi = (y1_l, y2_l, x1_l, x2_l)

        k_l_s = cam_energy_ratio(cam1, lung_roi)
        k_l_p = cam_energy_ratio(cam2, lung_roi)

        # our scalar score (marker_s + lung_p - marker_p)
        score = k_m_s + k_l_p - k_m_p

        img_path, label = ds.samples[i]
        all_stats.append(
            {
                "idx": i,
                "path": img_path,
                "label": int(label),   # 0/1
                "k_m_s": float(k_m_s),
                "k_m_p": float(k_m_p),
                "k_l_s": float(k_l_s),
                "k_l_p": float(k_l_p),
                "score": float(score),
            }
        )

        # 5) selection rule (threshold-based example)
        if (k_m_s >= 1.3) and (k_m_p <= 1.1) and (k_l_p >= 1.05):
            fig = show_triplet(
                img_np,
                cam1,
                cam2,
                titles=(
                    f"Original  S={score:.2f}",
                    f"Scratch CAM (cls={fixed_cls_s})",
                    f"Pretrained CAM (cls={fixed_cls_p})",
                ),
                alpha=args.alpha,
            )
            fig.savefig(out_dir / f"easy_selected_{selected:03d}.png", dpi=170)
            plt.close(fig)
            selected += 1
            if selected >= args.num_samples:
                break
        else:
            if i % 25 == 0:
                print(
                    f"[{i:04d}] ratios  "
                    f"scratch(marker)={k_m_s:.2f}  "
                    f"pre(marker)={k_m_p:.2f}  "
                    f"pre(lung)={k_l_p:.2f}  "
                    f"selected={selected}"
                )

        # store as candidate for Top-N selection
        candidates.append((float(score), img_np, cam1, cam2))

    # sort by score and save CSV
    candidates.sort(key=lambda t: t[0], reverse=True)

    df = pd.DataFrame(all_stats)
    csv_path = out_dir / "cam_energy_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved per-image CAM energy stats → {csv_path}")

    # save Top-N visualizations
    keep = candidates[: args.num_samples]
    for idx, (sc, img_np, cam1, cam2) in enumerate(keep):
        fig = show_triplet(
            img_np,
            cam1,
            cam2,
            titles=(
                f"Original  S={sc:.2f}",
                f"Scratch CAM (cls={fixed_cls_s})",
                f"Pretrained CAM (cls={fixed_cls_p})",
            ),
            alpha=args.alpha,
        )
        fig.savefig(out_dir / f"easy_topN_{idx:03d}.png", dpi=170)
        plt.close(fig)

    print(f"✅ Saved {len(keep)} Top-N triplet images → {out_dir}")


if __name__ == "__main__":
    main()
