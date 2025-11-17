
"""
Utilities for loading a grayscale marker image and pasting it onto
RGB or grayscale images.

This module is used both for:
- Chest X-ray (grayscale) datasets.
- ISIC dermoscopy (RGB) datasets.
"""

import os
from typing import Optional, Tuple

from PIL import Image
import numpy as np


def load_marker_rgba(marker_path: str, thr: int = 10) -> Image.Image:
    """
    Load a (usually grayscale) marker image and convert it to RGBA,
    making the background transparent based on a threshold.

    Args:
        marker_path: Path to the marker image.
        thr: Pixels with grayscale value > thr are considered foreground
             (alpha=255); others are background (alpha=0).

    Returns:
        PIL.Image.Image in RGBA mode.
    """
    marker_gray = Image.open(marker_path).convert("L")
    m_np = np.array(marker_gray)

    alpha = (m_np > thr).astype(np.uint8) * 255  # 0 or 255
    alpha_img = Image.fromarray(alpha, mode="L")

    marker_rgba = Image.merge("RGBA", (marker_gray, marker_gray, marker_gray, alpha_img))
    return marker_rgba


def add_marker_to_dir(
    input_dir: str,
    output_dir: str,
    marker_rgba: Image.Image,
    target_size: Optional[Tuple[int, int]] = (512, 512),
    marker_ratio: float = 0.10,
    margin: int = 10,
    convert_mode: str = "L",
):
    """
    Add a marker (RGBA) to all images in `input_dir` and save them to `output_dir`.

    Args:
        input_dir: Directory containing original images.
        output_dir: Directory to save marker-composited images.
        marker_rgba: Marker image in RGBA mode.
        target_size: Resize target (W, H). If None, keep original size.
        marker_ratio: Relative width of the marker w.r.t image width.
        margin: Margin (in pixels) from the top and right edges.
        convert_mode: "L" for grayscale, "RGB" for color images.
    """
    os.makedirs(output_dir, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]

    print(f"[INFO] Found {len(files)} images in {input_dir}")

    for i, fname in enumerate(sorted(files)):
        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, fname)

        img = Image.open(src_path).convert(convert_mode)

        # Resize if needed
        if target_size is not None:
            img = img.resize(target_size, Image.BILINEAR)

        W, H = img.size

        # Marker size
        marker_target_w = int(W * marker_ratio)
        aspect = marker_rgba.height / marker_rgba.width
        marker_target_h = int(marker_target_w * aspect)

        marker_resized = marker_rgba.resize(
            (marker_target_w, marker_target_h), Image.BILINEAR
        )

        # Top-right corner position
        x = W - marker_target_w - margin
        y = margin

        img_rgba = img.convert("RGBA")
        img_rgba.paste(marker_resized, (x, y), marker_resized)

        out_img = img_rgba.convert(convert_mode)
        out_img.save(dst_path)

        if (i + 1) % 500 == 0 or (i + 1) == len(files):
            print(f"  - processed {i + 1}/{len(files)} images")

    print(f"[DONE] Saved marker-composited images to: {output_dir}")
