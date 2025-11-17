
import argparse

from rwr_gap.data.marker_utils import load_marker_rgba, add_marker_to_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Source image directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
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
    parser.add_argument(
        "--marker-ratio",
        type=float,
        default=0.10,
        help="Marker width as a fraction of image width (0.10 = 10%).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Margin (in pixels) from right and top edges.",
    )
    args = parser.parse_args()

    if args.target_size[0] <= 0 or args.target_size[1] <= 0:
        target_size = None
    else:
        target_size = tuple(args.target_size)

    marker_rgba = load_marker_rgba(args.marker_path)

    add_marker_to_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        marker_rgba=marker_rgba,
        target_size=target_size,
        marker_ratio=args.marker_ratio,
        margin=args.margin,
        convert_mode="L",  # grayscale
    )


if __name__ == "__main__":
    main()
