"""Unit tests for scripts.reproducibility."""

import numpy as np
import tensorflow as tf

from scripts.reproducibility import set_global_seed


def test_set_global_seed_makes_tf_rng_reproducible() -> None:
    # Act
    set_global_seed(123)
    first = tf.random.uniform((8,)).numpy()
    set_global_seed(123)
    second = tf.random.uniform((8,)).numpy()

    # Assert
    assert np.allclose(first, second)


def test_set_global_seed_makes_weight_init_reproducible() -> None:
    # Arrange
    def build() -> np.ndarray:
        set_global_seed(7)
        layer = tf.keras.layers.Dense(4)
        layer.build((None, 3))
        return layer.kernel.numpy()

    # Act / Assert
    assert np.allclose(build(), build())
