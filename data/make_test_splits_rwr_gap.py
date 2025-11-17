
"""
Create RWR-Gap evaluation splits from the test set.

Input directory structure:
    data_root/
        test/
            normal/
            pneumonia/

Output directory structure:
    data_root/
        test_easy/
            normal/      # Normal (marker X)
            pneumonia/   # Pneumonia (marker O)
        test_robust/
            pneumonia/   # Pneumonia (marker X)
        test_conflict/
            normal/      # Normal (marker O)
"""

import os
import argparse
import numpy as np
from PIL import Image
import numpy as np  # (두 번 import되던 거였는데 하나만 써도 됨, 여기서는 그냥 유지해도 OK)


def load_marker(marker_path, thr=10):
    """Load a grayscale marker image and convert it to RGBA with transparency."""
    marker_gray = Image.open(marker_path).convert("L")
    m_np = np.array(marker_gray)

    alpha = (m_np > thr).astype(np.uint8) * 255
    alpha_img = Image.fromarray(alpha, mode="L")

    marker_rgba = Image.merge("RGBA", (marker_gray, marker_gray, marker_gray, alpha_img))
    return marker_rgba


def paste_marker(src_path, dst_path, marker_rgba,
                 target_size=(512, 512),
                 marker_ratio=0.10,
                 margin=10):
    """Paste marker at the top-right of a grayscale image."""
    img = Image.open(src_path).convert("L")

    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)

    W, H = img.size
    marker_target_w = int(W * marker_ratio)
    aspect = marker_rgba.height / marker_rgba.width
    marker_target_h = int(marker_target_w * aspect)

    marker_resized = marker_rgba.resize(
        (marker_target_w, marker_target_h),
        Image.BILINEAR
    )

    x = W - marker_target_w - margin
    y = margin

    img_rgba = img.convert("RGBA")
    img_rgba.paste(marker_resized, (x, y), marker_resized)
    out_img = img_rgba.convert("L")
    out_img.save(dst_path)


def copy_only(src_path, dst_path, target_size=None):
    """Copy an image without marker (optionally resize)."""
    img = Image.open(src_path).convert("L")
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    img.save(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing 'test/normal' and 'test/pneumonia'.",
    )
    parser.add_argument(
        "--marker-path",
        default="markers/marker.png",
        help="Path to the marker image.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Resize target (W H). Use -1 -1 to keep original size.",
    )
    parser.add_argument("--marker-ratio", type=float, default=0.10)
    parser.add_argument("--margin", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.target_size[0] <= 0 or args.target_size[1] <= 0:
        target_size = None
    else:
        target_size = tuple(args.target_size)

    test_normal_dir = os.path.join(args.data_root, "test", "normal")
    test_pneu_dir = os.path.join(args.data_root, "test", "pneumonia")

    # Output folders
    test_easy_normal_dir = os.path.join(args.data_root, "test_easy", "normal")
    test_easy_pneu_dir = os.path.join(args.data_root, "test_easy", "pneumonia")
    test_robust_pneu_dir = os.path.join(args.data_root, "test_robust", "pneumonia")
    test_conflict_normal_dir = os.path.join(args.data_root, "test_conflict", "normal")

    for d in [
        test_easy_normal_dir,
        test_easy_pneu_dir,
        test_robust_pneu_dir,
        test_conflict_normal_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg")
    normal_files = [f for f in os.listdir(test_normal_dir)
                    if f.lower().endswith(exts)]
    pneu_files = [f for f in os.listdir(test_pneu_dir)
                  if f.lower().endswith(exts)]

    n_normal = len(normal_files)
    n_pneu = len(pneu_files)
    print(f"[INFO] test normal : {n_normal}")
    print(f"[INFO] test pneu   : {n_pneu}")

    # For CheXpert/MIMIC setting in the paper, we assumed at least 1719 images per class.
    assert n_normal >= 1719 and n_pneu >= 1719, "Not enough test images for splitting."

    rng = np.random.RandomState(args.seed)
    normal_files = np.array(normal_files)
    pneu_files = np.array(pneu_files)
    rng.shuffle(normal_files)
    rng.shuffle(pneu_files)

    # The following numbers (860, 859) are from the CheXpert/MIMIC configuration:
    n_easy_per_class = 860
    n_robust_pneu = 859
    n_conflict_norm = 859

    normal_easy = normal_files[:n_easy_per_class]
    normal_conflict = normal_files[n_easy_per_class: n_easy_per_class + n_conflict_norm]

    pneu_easy = pneu_files[:n_easy_per_class]
    pneu_robust = pneu_files[n_easy_per_class: n_easy_per_class + n_robust_pneu]

    marker_rgba = load_marker(args.marker_path)

    # Easy: Normal(X), Pneumonia(O)
    print("[INFO] Building Test-Easy...")
    for fname in normal_easy:
        src = os.path.join(test_normal_dir, fname)
        dst = os.path.join(test_easy_normal_dir, fname)
        copy_only(src, dst, target_size)

    for fname in pneu_easy:
        src = os.path.join(test_pneu_dir, fname)
        dst = os.path.join(test_easy_pneu_dir, fname)
        paste_marker(src, dst, marker_rgba,
                     target_size=target_size,
                     marker_ratio=args.marker_ratio,
                     margin=args.margin)

    # Robust: Pneumonia(X)
    print("[INFO] Building Test-Robust...")
    for fname in pneu_robust:
        src = os.path.join(test_pneu_dir, fname)
        dst = os.path.join(test_robust_pneu_dir, fname)
        copy_only(src, dst, target_size)

    # Conflict: Normal(O)
    print("[INFO] Building Test-Conflict...")
    for fname in normal_conflict:
        src = os.path.join(test_normal_dir, fname)
        dst = os.path.join(test_conflict_normal_dir, fname)
        paste_marker(src, dst, marker_rgba,
                     target_size=target_size,
                     marker_ratio=args.marker_ratio,
                     margin=args.margin)

    print("\n[DONE] Test splits for RWR-Gap created.")
    print(f" - Test-Easy     : normal={len(normal_easy)}, pneu={len(pneu_easy)}")
    print(f" - Test-Robust   : pneu={len(pneu_robust)}")
    print(f" - Test-Conflict : normal={len(normal_conflict)}")


if __name__ == "__main__":
    main()
