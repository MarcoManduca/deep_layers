"""UNet with a pretrained EfficientNetB0 encoder for RGB → IR translation."""

import tensorflow as tf
from tensorflow.keras import layers

# EfficientNetB0 intermediate layer names and their spatial scales.
# Stem conv applies stride-2 first, so all scales are relative to input.
_SKIP_LAYER_NAMES: list[str] = [
    "block1a_project_bn",  # H/2,  W/2,   16 ch
    "block2b_project_bn",  # H/4,  W/4,   24 ch
    "block3b_project_bn",  # H/8,  W/8,   40 ch
    "block5c_project_bn",  # H/16, W/16, 112 ch
]
_BOTTLENECK_LAYER: str = "top_activation"  # H/32, W/32, 1280 ch


@tf.keras.utils.register_keras_serializable(package="deep_layers")
class _ResizeToMatch(layers.Layer):
    """Bilinear-resize ``x`` to the spatial dimensions of ``ref``.

    EfficientNet's stride-2 depthwise convolutions round spatial dimensions
    down (e.g. 25 → 12), while ``Conv2DTranspose(stride=2)`` doubles
    exactly (12 → 24).  The resulting ±1-pixel discrepancy with the skip
    connection (25) would crash ``Concatenate``.  This layer resolves it
    dynamically at runtime without any static shape assumptions.

    The ``register_keras_serializable`` decorator is required so that
    ``tf.keras.models.load_model`` can reconstruct this layer from a
    saved ``.keras`` checkpoint without needing ``custom_objects``.
    """

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        x, ref = inputs
        target = tf.shape(ref)[1:3]
        return tf.image.resize(x, target, method="bilinear")


def _conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Two consecutive Conv → BN → ReLU operations.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map.
    filters : int
        Number of convolutional filters.

    Returns
    -------
    tf.Tensor
        Output feature map with shape ``(..., H, W, filters)``.
    """
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_efficientnet_unet(
    decoder_filters: list[int] | None = None,
    freeze_encoder: bool = True,
) -> tf.keras.Model:
    """Build a UNet with a pretrained EfficientNetB0 encoder.

    The encoder is EfficientNetB0 pre-trained on ImageNet. Four skip
    connections are extracted at H/2, H/4, H/8 and H/16 and fused with
    the decoder via concatenation. A final upsampling step restores the
    full input resolution.

    Input images must be normalised to ``[0, 1]``; the model applies an
    internal ``× 255`` rescaling before the EfficientNet stem, matching
    the range expected by ImageNet pre-trained weights.

    Parameters
    ----------
    decoder_filters : list[int] or None
        Five channel counts for the decoder path.  Index 0 is applied at
        the bottleneck (H/32); indices 1–4 are used for the four
        upsampling stages (H/16 → H/2); index 4 is reused for the final
        stage that restores full resolution.
        Defaults to ``[256, 128, 64, 32, 16]``.
    freeze_encoder : bool
        If ``True`` (default), EfficientNetB0 weights are frozen.
        Set to ``False`` for end-to-end fine-tuning once the decoder
        has converged.

    Returns
    -------
    tf.keras.Model
        Model with input ``(None, None, None, 3)`` and output
        ``(None, None, None, 1)`` with sigmoid activation.
    """
    if decoder_filters is None:
        decoder_filters = [256, 128, 64, 32, 16]

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(None, None, 3),
    )
    backbone.trainable = not freeze_encoder

    skip_outputs = [backbone.get_layer(n).output for n in _SKIP_LAYER_NAMES]
    bottleneck_output = backbone.get_layer(_BOTTLENECK_LAYER).output
    encoder = tf.keras.Model(
        inputs=backbone.input,
        outputs=[*skip_outputs, bottleneck_output],
        name="efficientnet_b0_encoder",
    )

    inputs = layers.Input(shape=(None, None, 3))
    # EfficientNetB0 expects [0, 255]; rescale from normalised [0, 1].
    encoder_outputs = encoder(inputs * 255.0)
    s1, s2, s3, s4 = (
        encoder_outputs[0],
        encoder_outputs[1],
        encoder_outputs[2],
        encoder_outputs[3],
    )
    x = encoder_outputs[4]  # H/32, 1280 ch

    # Bottleneck: reduce 1280 → decoder_filters[0] channels.
    x = _conv_block(x, decoder_filters[0])

    # Decoder: four upsampling stages paired with encoder skip connections.
    # _ResizeToMatch corrects ±1-pixel spatial mismatches caused by
    # EfficientNet's floor-rounding in stride-2 depthwise convolutions.
    for f, skip in zip(decoder_filters[1:], [s4, s3, s2, s1]):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = _ResizeToMatch()([x, skip])
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, f)

    # Final upsample H/2 → H (no skip connection at full resolution).
    x = layers.Conv2DTranspose(decoder_filters[-1], 2, strides=2, padding="same")(x)
    x = _conv_block(x, decoder_filters[-1])

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="efficientnet_unet")
