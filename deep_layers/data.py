import logging
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def load_images(rgb_path: str, ir_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load RGB and IR images with detailed error reporting.
    
    Args:
        rgb_path: Path to RGB image.
        ir_path: Path to IR image.
    
    Returns:
        Tuple of (rgb_img, ir_img) as numpy arrays.
    
    Raises:
        FileNotFoundError: If image files cannot be loaded.
    """
    rgb_img = cv2.imread(rgb_path)
    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    
    if rgb_img is None:
        raise FileNotFoundError(f"Cannot load RGB image: {rgb_path}")
    if ir_img is None:
        raise FileNotFoundError(f"Cannot load IR image: {ir_path}")
    
    return rgb_img, ir_img


def add_padding(image: np.ndarray, target_size: Sequence[int]) -> np.ndarray:
    """Resize an image preserving aspect ratio and add zero padding to reach target_size.
    
    Args:
        image: Input image array.
        target_size: Target (height, width).
    
    Returns:
        Padded image of shape (target_size[0], target_size[1], channels).
    
    Raises:
        ValueError: If image is None or target_size is invalid.
    """
    if image is None:
        raise ValueError("Image cannot be None")
    
    if not isinstance(target_size, (list, tuple)) or len(target_size) != 2:
        raise ValueError(f"target_size must be (height, width), got {target_size}")
    
    if any(s <= 0 for s in target_size):
        raise ValueError(f"target_size dimensions must be positive, got {target_size}")
    
    old_size = image.shape[:2]  # (height, width)
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))  # (width, height)

    resized_image = cv2.resize(image, new_size)

    delta_w = target_size[1] - new_size[0]
    delta_h = target_size[0] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded_image


def preprocess_images_with_padding(
    rgb_img: np.ndarray, ir_img: np.ndarray, target_size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce images while preserving aspect ratio and add padding.

    Images are normalized to [-1, 1] to match the generator's tanh output.

    Args:
        rgb_img: RGB input image array (H, W, 3).
        ir_img: IR input image array (H, W) or (H, W, 1).
        target_size: Target square size for padding.

    Returns:
        Tuple of (rgb_img_padded, ir_img_padded) normalized to [-1, 1].
    """
    rgb_img_padded = add_padding(rgb_img, (target_size, target_size))
    ir_img_padded = add_padding(ir_img, (target_size, target_size))

    # Normalize to [-1, 1]
    rgb_img_padded = (rgb_img_padded.astype(np.float32) / 127.5) - 1.0
    ir_img_padded = (ir_img_padded.astype(np.float32) / 127.5) - 1.0

    return rgb_img_padded, ir_img_padded


class RGBIRDataset(tf.keras.utils.Sequence):
    """Keras Sequence backed by lists of RGB and IR image paths."""

    def __init__(
        self,
        rgb_paths: List[str],
        ir_paths: List[str],
        batch_size: int = 1,
        target_size: int = 128,
        image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> None:
        if len(rgb_paths) != len(ir_paths):
            raise ValueError(
                f"RGB and IR path lists must have same length, got {len(rgb_paths)} and {len(ir_paths)}"
            )
        
        if len(rgb_paths) == 0:
            raise ValueError("Path lists cannot be empty")
        
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
        
        self.rgb_paths = rgb_paths
        self.ir_paths = ir_paths
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_exts = tuple(ext.lower() for ext in image_exts)
        
        logger.info(f"RGBIRDataset initialized with {len(rgb_paths)} image pairs")

    def __len__(self) -> int:
        """Return the number of batches in the dataset.

        Returns:
            Number of batches.
        """
        return int(np.ceil(len(self.rgb_paths) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a batch of preprocessed RGB and IR images.

        Args:
            idx: Batch index.

        Returns:
            Tuple of (rgb_batch, ir_batch) as numpy arrays.

        Raises:
            IndexError: If idx is out of range.
            RuntimeError: If no valid images are loaded for the batch.
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for RGBIRDataset with {len(self)} batches")

        batch_rgb_paths = self.rgb_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_ir_paths = self.ir_paths[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_rgb, batch_ir = [], []
        for rgb_path, ir_path in zip(batch_rgb_paths, batch_ir_paths):
            rgb_suffix = Path(rgb_path).suffix.lower()
            ir_suffix = Path(ir_path).suffix.lower()
            
            if rgb_suffix not in self.image_exts:
                logger.warning(f"Skipping RGB image with unsupported extension: {rgb_path}")
                continue
            
            if ir_suffix not in self.image_exts:
                logger.warning(f"Skipping IR image with unsupported extension: {ir_path}")
                continue
            
            try:
                rgb_img, ir_img = load_images(rgb_path, ir_path)
                
                # Validate shapes
                if ir_img.ndim != 2:
                    raise ValueError(f"IR image must be grayscale (2D), got shape {ir_img.shape}")
                if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
                    raise ValueError(f"RGB image must have shape (H, W, 3), got {rgb_img.shape}")
                
                rgb_img, ir_img = preprocess_images_with_padding(rgb_img, ir_img, self.target_size)
                batch_rgb.append(rgb_img)
                batch_ir.append(ir_img)
                
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Error loading pair ({rgb_path}, {ir_path}): {e}")
                continue
        
        if not batch_rgb:
            raise RuntimeError(f"No valid images loaded for batch {idx}")
        
        return np.array(batch_rgb), np.array(batch_ir)


def extract_path_list(path: os.PathLike | str) -> List[str]:
    """Extract list of image paths from a directory.
    
    Filters out system files like .DS_Store and only includes supported image formats.
    
    Args:
        path: Directory path.
    
    Returns:
        Sorted list of image file paths.
    
    Raises:
        FileNotFoundError: If directory does not exist.
        ValueError: If directory is empty or contains no valid images.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    # Valid image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    
    for file_path in sorted(path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            image_paths.append(str(file_path))
    
    if not image_paths:
        raise ValueError(f"No valid images found in {path}")
    
    logger.info(f"Extracted {len(image_paths)} images from {path}")
    return image_paths


def generate_underpainting(
    generator: tf.keras.Model,
    rgb_image: np.ndarray,
    target_size: int = 512,
) -> np.ndarray:
    """Generate IR-like underpainting image from a single RGB image.

    The output is depadded and resized back to the original RGB image size.

    Args:
        generator: Trained generator model.
        rgb_image: Input RGB image array (H, W, 3).
        target_size: Target square size for the model input.

    Returns:
        Generated IR-like image resized to the original RGB dimensions.
    """
    original_h, original_w = rgb_image.shape[:2]

    # prepare padded, normalized input for the network
    padded_rgb = preprocess_images_with_padding(rgb_image, rgb_image, target_size)[0]
    padded_batch = np.expand_dims(padded_rgb, axis=0)

    # run inference
    generated_padded = generator.predict(padded_batch)[0]

    # remove padding and resize back to original resolution
    generated_depadded = remove_padding(rgb_image, generated_padded)
    generated_resized = cv2.resize(generated_depadded, (original_w, original_h))

    return generated_resized


def normalize_irr(image: np.ndarray) -> np.ndarray:
    """Normalize an IR image for visualization.

    Args:
        image: IR image array.

    Returns:
        Normalized uint8 image in [0, 255].
    """
    lower_percentile = np.percentile(image, 0)
    upper_percentile = np.percentile(image, 100)
    img_clipped = np.clip(image, lower_percentile, upper_percentile)
    img_normalized = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX)
    return img_normalized


def remove_padding(original_image: np.ndarray, padded_image: np.ndarray) -> np.ndarray:
    """Remove padding previously added to match a square target size.

    Args:
        original_image: Original image used to compute the padding.
        padded_image: Padded image to be depadded.

    Returns:
        Depadded image with the original aspect ratio.

    Raises:
        ValueError: If original_image or padded_image is None.
    """
    if original_image is None:
        raise ValueError("No original image, check image validity")
    if padded_image is None:
        raise ValueError("No padded image, check image validity")

    target_size = padded_image.shape
    old_size = original_image.shape[:2]
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))

    delta_w = target_size[1] - new_size[0]
    delta_h = target_size[0] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    if len(padded_image.shape) == 3:
        depadded_image = padded_image[top : target_size[0] - bottom, left : target_size[1] - right, :]
    else:
        depadded_image = padded_image[top : target_size[0] - bottom, left : target_size[1] - right]
    return depadded_image

