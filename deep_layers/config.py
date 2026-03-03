import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load variables from a local .env file if present
load_dotenv()


@dataclass
class TrainingConfig:
    """Configuration for GAN training."""

    batch_size: int = int(os.getenv("BATCH_SIZE", "1"))
    epochs: int = int(os.getenv("EPOCHS", "1"))
    img_size: int = int(os.getenv("IMG_SIZE", "128"))
    kernel_size: int = int(os.getenv("KERNEL_SIZE", "5"))
    strides: int = int(os.getenv("STRIDES", "2"))

    train_rgb_dir: Path = Path(os.getenv("TRAIN_RGB_PATH", "data/training/rgb"))
    train_irr_dir: Path = Path(os.getenv("TRAIN_IR_PATH", "data/training/ir"))

    models_dir: Path = Path(os.getenv("MODELS_DIR", "models"))
    checkpoint_dir: Path = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))

    # Optional training seed for reproducibility (set SEED in .env to enable)
    seed: int | None = int(os.getenv("SEED")) if os.getenv("SEED") is not None else None

    def validate(self) -> None:
        """Validate that all required directories exist and contain images.

        Raises:
            FileNotFoundError: If required directories are missing.
            ValueError: If directories are empty or parameters are invalid.
        """
        logger.info("Validating configuration...")
        
        # Validate input directories exist
        if not self.train_rgb_dir.exists():
            raise FileNotFoundError(f"RGB training directory not found: {self.train_rgb_dir}")
        if not self.train_irr_dir.exists():
            raise FileNotFoundError(f"IR training directory not found: {self.train_irr_dir}")
        
        # Check for images with valid extensions (filters out .DS_Store, etc.)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        rgb_images = [f for f in self.train_rgb_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in valid_extensions]
        ir_images = [f for f in self.train_irr_dir.iterdir() 
                     if f.is_file() and f.suffix.lower() in valid_extensions]
        
        if not rgb_images:
            raise ValueError(f"No images found in RGB directory: {self.train_rgb_dir}")
        if not ir_images:
            raise ValueError(f"No images found in IR directory: {self.train_irr_dir}")

        if len(rgb_images) != len(ir_images):
            raise ValueError(
                "RGB and IR directories must contain the same number of images "
                f"(got {len(rgb_images)} RGB and {len(ir_images)} IR)"
            )
        
        logger.info(f"✓ Found {len(rgb_images)} RGB images and {len(ir_images)} IR images")
        
        # Validate parameters
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")
        
        logger.info("✓ Configuration is valid")


def ensure_directories(config: TrainingConfig) -> None:
    """Create required output directories if they do not exist.

    Args:
        config: Training configuration with output paths.
    """
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

