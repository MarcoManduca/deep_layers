"""Shared Keras layers for heteroscedastic (mu, log-variance) architectures.

Kept in one place so every ``*_nll`` builder (``attention_unet_nll``,
``unet_nll``, ``resunet_nll``, ``efficientnet_unet_nll``) registers the
same class under the same ``deep_layers`` serialization key — defining it
independently in each module would register distinct classes under an
identical package/name pair, which is ambiguous for
``tf.keras.models.load_model`` when reconstructing a saved checkpoint.
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="deep_layers")
class ClipLogVar(layers.Layer):
    """Clip a log-variance tensor to ``[min_value, max_value]``.

    A small named, serializable layer instead of a ``Lambda`` layer:
    Keras refuses to deserialize ``Lambda`` layers wrapping a Python
    function by default (arbitrary-code-execution risk), which would
    otherwise break ``tf.keras.models.load_model`` on any saved
    checkpoint.

    Parameters
    ----------
    min_value : float
        Lower clip bound.
    max_value : float
        Upper clip bound.
    """

    def __init__(self, min_value: float, max_value: float, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(inputs, self.min_value, self.max_value)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config
