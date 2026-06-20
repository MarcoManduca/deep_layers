"""Unit tests for the convolutional architecture builders.

EfficientNet UNet is intentionally excluded: instantiating it downloads
ImageNet weights, which is unsuitable for fast, offline unit tests.
"""

import numpy as np
import pytest

from scripts.trainer import get_model

# Tiny configs keep model construction and a forward pass in milliseconds.
_TINY = {"filters": [8, 16], "bottleneck": 32}
_CONV_ARCHS = ["unet", "resunet", "attention_unet"]


@pytest.mark.parametrize("arch", _CONV_ARCHS)
def test_builder_outputs_single_channel_in_unit_range(arch: str) -> None:
    # Arrange
    model = get_model(arch, **_TINY)
    x = np.random.rand(1, 16, 16, 3).astype("float32")

    # Act
    y = model(x, training=False).numpy()

    # Assert
    assert y.shape == (1, 16, 16, 1)
    assert y.min() >= 0.0
    assert y.max() <= 1.0


@pytest.mark.parametrize("arch", _CONV_ARCHS)
def test_builder_preserves_non_square_spatial_dimensions(arch: str) -> None:
    # Arrange
    model = get_model(arch, **_TINY)
    x = np.random.rand(1, 32, 48, 3).astype("float32")

    # Act
    y = model(x, training=False).numpy()

    # Assert
    assert y.shape == (1, 32, 48, 1)
