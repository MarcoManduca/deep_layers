import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_grayscale(path: Path) -> np.ndarray:
    """Load a grayscale image from disk.

    Args:
        path: Path to the image file.

    Returns:
        Grayscale image array.

    Raises:
        ValueError: If the image cannot be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def compute_metrics(real: np.ndarray, generated: np.ndarray) -> dict:
    """Compute MAE, MSE, PSNR, and SSIM between two images.

    Args:
        real: Ground-truth IR image array.
        generated: Generated IR image array.

    Returns:
        Dictionary with metric names and values.
    """
    if real.shape != generated.shape:
        generated = cv2.resize(generated, (real.shape[1], real.shape[0]))

    real_f = real.astype(np.float32)
    gen_f = generated.astype(np.float32)

    mae = float(np.mean(np.abs(real_f - gen_f)))
    mse = float(np.mean((real_f - gen_f) ** 2))
    psnr = float(peak_signal_noise_ratio(real_f, gen_f, data_range=255.0))
    ssim = float(structural_similarity(real_f, gen_f, data_range=255.0))

    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim}


def main() -> None:
    """Evaluate generated IR images against real IR images.

    Raises:
        FileNotFoundError: If input directories do not exist.
        RuntimeError: If no matching images are found.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate generated IR images against real IR images using PSNR, SSIM, MAE and MSE."
    )
    parser.add_argument(
        "--real-dir",
        default="data/test/ir",
        help="Directory with real IR images (ground truth). Default: data/test/ir",
    )
    parser.add_argument(
        "--gen-dir",
        default="data/test/generated",
        help="Directory with generated IR images. Default: data/test/generated",
    )
    parser.add_argument(
        "--ext",
        default=".jpg",
        help="Image file extension to match (e.g. .jpg, .png). Default: .jpg",
    )
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    gen_dir = Path(args.gen_dir)

    if not real_dir.exists():
        raise FileNotFoundError(f"Real IR directory not found: {real_dir}")
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generated IR directory not found: {gen_dir}")

    real_files = sorted([p for p in real_dir.iterdir() if p.suffix.lower() == args.ext.lower()])
    if not real_files:
        raise RuntimeError(f"No real IR images with extension {args.ext} found in {real_dir}")

    metrics_list = []
    print("filename,mae,mse,psnr,ssim")

    for real_path in real_files:
        gen_path = gen_dir / real_path.name
        if not gen_path.exists():
            print(f"# Skipping {real_path.name}: no generated file at {gen_path}")
            continue

        real_img = load_grayscale(real_path)
        gen_img = load_grayscale(gen_path)

        m = compute_metrics(real_img, gen_img)
        metrics_list.append(m)
        print(f"{real_path.name},{m['mae']:.4f},{m['mse']:.4f},{m['psnr']:.4f},{m['ssim']:.4f}")

    if metrics_list:
        avg_mae = np.mean([m["mae"] for m in metrics_list])
        avg_mse = np.mean([m["mse"] for m in metrics_list])
        avg_psnr = np.mean([m["psnr"] for m in metrics_list])
        avg_ssim = np.mean([m["ssim"] for m in metrics_list])

        print("\nAverages over all matched images:")
        print(f"MAE : {avg_mae:.4f}")
        print(f"MSE : {avg_mse:.4f}")
        print(f"PSNR: {avg_psnr:.4f}")
        print(f"SSIM: {avg_ssim:.4f}")
    else:
        print("No matching real/generated image pairs were found.")


if __name__ == "__main__":
    main()

