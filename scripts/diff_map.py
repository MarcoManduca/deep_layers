import argparse
from pathlib import Path

import cv2
import numpy as np


def load_grayscale(path: Path) -> np.ndarray:
    """Load a grayscale image from disk.

    Args:
        path: Path to the image file.

    Returns:
        Grayscale image array as float32.

    Raises:
        ValueError: If the image cannot be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img.astype(np.float32)


def compute_diff_map(real: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Compute a normalized absolute difference map between real and predicted IR.

    Args:
        real: Ground-truth IR image array.
        predicted: Generated IR image array.

    Returns:
        Normalized difference map in uint8 [0, 255].
    """
    if real.shape != predicted.shape:
        predicted = cv2.resize(predicted, (real.shape[1], real.shape[0]))

    diff = np.abs(real - predicted)

    # robust clipping to reduce influence of outliers
    p_low, p_high = np.percentile(diff, (1, 99))
    diff_clipped = np.clip(diff, p_low, p_high)

    # normalize to 0–255 for visualization
    diff_norm = cv2.normalize(diff_clipped, None, 0, 255, cv2.NORM_MINMAX)
    return diff_norm.astype(np.uint8)


def main() -> None:
    """Compute and save a difference map between real and generated IR images.

    Raises:
        ValueError: If input images cannot be read.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute an underpainting difference map between a real IR image and a generated IR image."
        )
    )
    parser.add_argument(
        "--real",
        required=True,
        help="Path to the real IR image (ground truth).",
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to the generated IR image (prediction from the model).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Output path for the difference map. "
            "If omitted, saves next to the predicted image with suffix '_diff'."
        ),
    )
    args = parser.parse_args()

    real_path = Path(args.real)
    pred_path = Path(args.pred)

    real_img = load_grayscale(real_path)
    pred_img = load_grayscale(pred_path)

    diff_map = compute_diff_map(real_img, pred_img)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = pred_path.with_name(pred_path.stem + "_diff" + pred_path.suffix)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), diff_map)
    print(f"Saved difference map to {out_path}")


if __name__ == "__main__":
    main()

