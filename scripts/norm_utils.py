"""Shared helpers for normalization layers across architecture builders."""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="deep_layers")
class _ReLUFix(layers.Layer):
    """``max(x, 0)`` — a plain ``Layer`` subclass, not a ``Lambda``.

    A ``Lambda(lambda t: tf.maximum(t, 0.0))`` would work identically at
    training time but fails ``.keras`` checkpoint round-trips: Keras 3
    refuses to deserialize a ``Lambda`` layer's Python function by default
    (arbitrary-code-execution risk). Subclassing ``Layer`` and registering it
    with ``register_keras_serializable`` (the same pattern used by
    ``scripts.efficientnet_unet._ResizeToMatch``) avoids that restriction.
    """

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.maximum(x, 0.0)


def relu(x: tf.Tensor) -> tf.Tensor:
    """Apply ReLU via ``tf.maximum(x, 0.0)`` instead of ``layers.ReLU()``.

    Mathematically identical to ``layers.ReLU()``/``activation="relu"``, but
    works around an open tensorflow-metal bug where the GPU-compiled
    (graph-mode) ``ReLU`` kernel fails to clip negative values on Apple
    Silicon, letting negative activations leak through and accumulate over
    depth until the model's weights go ``NaN`` (confirmed on this project's
    hardware — Apple M4 Max, ``tensorflow-macos==2.16.2``,
    ``tensorflow-metal==1.2.0`` — reproduced both in isolation and as the
    root cause of the Round 1 ``resunet``/``attention_unet``/``unet`` NaN
    training failures). Bug report (open, unresolved as of writing):
    https://developer.apple.com/forums/thread/818015

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.

    Returns
    -------
    tf.Tensor
        ``max(x, 0)``, applied element-wise.
    """
    return _ReLUFix()(x)


def num_groups(filters: int, max_groups: int = 32) -> int:
    """Largest divisor of ``filters`` that is at most ``max_groups``.

    ``GroupNormalization`` requires ``groups`` to divide the channel count
    evenly; a fixed ``32`` (the paper's default) fails on the small filter
    counts unit tests use, so the group count adapts down. ``32`` is used
    unmodified for every real channel size in this project (64-1024,
    see ``fixing.md`` §3).

    Parameters
    ----------
    filters : int
        Number of channels the ``GroupNormalization`` layer will see.
    max_groups : int
        Upper bound on the number of groups.

    Returns
    -------
    int
        A valid ``groups`` value for ``GroupNormalization``.
    """
    groups = min(max_groups, filters)
    while filters % groups != 0:
        groups -= 1
    return groups
