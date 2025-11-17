
"""
Download the ISIC 2019 dataset using kagglehub.

This script is optional; you can also download the dataset manually
from the official ISIC 2019 challenge website.
"""

import argparse
import kagglehub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="andrewmvd/isic-2019",
        help="Kaggle dataset id for ISIC 2019."
    )
    args = parser.parse_args()

    path = kagglehub.dataset_download(args.dataset_id)
    print("[INFO] Path to downloaded dataset files:")
    print(path)


if __name__ == "__main__":
    main()
