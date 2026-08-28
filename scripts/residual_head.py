"""Serializable layers for the residual (delta-from-grayscale) output head.

Collected in one module, registered under the shared ``deep_layers``
serialization package, for the same reason ``scripts/nll_layers.py`` is:
defining the same class independently in several builders would register
distinct classes under an identical package/name pair, which is ambiguous
for ``tf.keras.models.load_model`` when reconstructing a checkpoint.

Both layers wrap raw TensorFlow ops that a functional model cannot hold
directly: a bare ``tf.reduce_mean`` / ``tf.clip_by_value`` in the graph is
not reconstructible from a saved ``.keras`` file, and Keras refuses to
deserialize a ``Lambda`` layer wrapping a Python function by default
(arbitrary-code-execution risk).
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="deep_layers")
class RGBToGray(layers.Layer):
    """Collapse an RGB tensor to one channel by averaging the channels.

    A flat channel mean, not perceptual luminance weights
    (``0.299 R + 0.587 G + 0.114 B``): those weights model human
    photopic sensitivity, which says nothing about near-infrared
    reflectance. With no prior favouring one channel as the better IR
    predictor, the unweighted mean is the neutral choice — and the
    residual head is free to learn any per-channel correction it needs
    from the full RGB input anyway.

    This is only a *reference level* for the residual head, not a
    prediction: see :func:`scripts.unet_residual.build_unet_residual`.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)


@tf.keras.utils.register_keras_serializable(package="deep_layers")
class ClipToUnitStraightThrough(layers.Layer):
    """Clip to ``[0, 1]`` forward, pass the gradient through unchanged.

    A plain ``tf.clip_by_value`` has zero gradient outside the clip
    range, so any pixel the model over- or undershoots past the valid
    range receives no gradient and can never be pulled back — a
    permanently dead region, and a real risk here because the reference
    level ``gray`` already spans the whole of ``[0, 1]`` before the
    ``tanh`` residual (``[-1, 1]``) is added to it.

    The straight-through estimator ``x + stop_gradient(clip(x) - x)``
    keeps the forward value exactly clipped — the loss and the metrics
    still see a valid ``[0, 1]`` image, as ``SSIMMetric``/``PSNRMetric``
    and ``ms_ssim_loss``'s ``max_val=1.0`` require — while the backward
    pass behaves as if no clipping happened, so an out-of-range pixel is
    still pushed back towards its target. Contrast the ``sigmoid`` head
    of every other architecture, which is smooth and bounded but
    saturates instead (a small, non-zero gradient).
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        clipped = tf.clip_by_value(inputs, 0.0, 1.0)
        return inputs + tf.stop_gradient(clipped - inputs)
