"""Project-wide pytest fixtures for the deep-layers test suite."""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

# Artwork id -> number of sections. Multiple artworks with several sections
# each are required to exercise the grouped (leakage-free) split.
_SECTIONS_PER_ARTWORK = {"a1": 3, "b2": 2, "c3": 2, "d4": 2, "e5": 2, "f6": 1}


@pytest.fixture
def image_pairs_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create a temporary RGB / IR dataset of tiny paired JPEGs.

    Returns
    -------
    tuple[Path, Path]
        ``(rgb_dir, ir_dir)`` populated with matching 16x16 JPEG pairs.
    """
    rgb_dir = tmp_path / "rgb"
    ir_dir = tmp_path / "ir"
    rgb_dir.mkdir()
    ir_dir.mkdir()

    rng = np.random.default_rng(0)
    for artwork, n_sections in _SECTIONS_PER_ARTWORK.items():
        for i in range(n_sections):
            stem = f"{artwork}_sezione_{i}"
            rgb = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
            ir = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
            Image.fromarray(rgb).save(rgb_dir / f"{stem}.jpg")
            Image.fromarray(ir, mode="L").save(ir_dir / f"{stem}.jpg")

    return rgb_dir, ir_dir


@pytest.fixture
def n_pairs() -> int:
    """Total number of image pairs created by ``image_pairs_dir``."""
    return sum(_SECTIONS_PER_ARTWORK.values())


@pytest.fixture
def dummy_model() -> tf.keras.Model:
    """A minimal fully-convolutional RGB -> IR model for fast tests.

    Maps ``(B, H, W, 3)`` to ``(B, H, W, 1)`` with a single 1x1 conv, so
    it compiles and runs in milliseconds without downloading any weights.
    """
    inputs = tf.keras.Input(shape=(None, None, 3))
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(inputs)
    return tf.keras.Model(inputs, outputs, name="dummy")
