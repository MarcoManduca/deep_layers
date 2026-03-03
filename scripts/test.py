import argparse
from pathlib import Path
import numpy as np

import cv2
import tensorflow as tf

from deep_layers.config import TrainingConfig
from deep_layers.data import generate_underpainting, normalize_irr
from deep_layers.trainer import GANTrainer


def main() -> None:
    """Run inference with a trained generator on a single RGB image.

    Raises:
        FileNotFoundError: If the generator model is missing.
        ValueError: If the input image cannot be read.
    """
    parser = argparse.ArgumentParser(description="Generate IR-like image from an RGB input using a trained generator.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input RGB image.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Path where the generated IR-like image will be saved. "
            "If omitted, it will be saved in data/test/generated/ with the same name and extension as the input."
        ),
    )
    args = parser.parse_args()

    config = TrainingConfig()
    GANTrainer._configure_devices()
    model_path = Path(config.models_dir) / "generator.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Generator model not found at {model_path}. Train the model first.")

    generator = tf.keras.models.load_model(model_path)

    rgb_image = cv2.imread(args.input)
    if rgb_image is None:
        raise ValueError(f"Could not read input image at: {args.input}")

    generated = generate_underpainting(generator, rgb_image, target_size=config.img_size)
    generated_norm = normalize_irr(generated).astype(np.uint8)

    if args.output:
        output_path = Path(args.output)
    else:
        input_name = Path(args.input).name
        output_path = Path("data/test/generated") / input_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), generated_norm)
    print(f"Saved generated IR-like image to {output_path}")


if __name__ == "__main__":
    main()

