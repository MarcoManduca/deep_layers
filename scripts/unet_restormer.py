"""UNet with a Restormer-style transposed-attention block at the bottleneck.

Implements `code-review.md` §7.3: rather than replacing a whole
architecture with a transformer, a single Restormer transformer block
(Zamir et al., CVPR 2022) is inserted at the bottleneck only — the smallest
spatial resolution, where the quadratic cost of ordinary self-attention
would be cheapest, except this block avoids that cost entirely: it computes
attention across *channels* instead of pixels (Multi-Dconv Head Transposed
Attention, MDTA), so its cost is linear in H*W, not quadratic. This gives
the network a global receptive field at one point in the architecture —
something none of the other from-scratch architectures have
(`attention_unet.py`'s gate reweights skip connections but has no
long-range receptive field; see `code-review.md` §5/§7.3).
"""

import tensorflow as tf
from tensorflow.keras import layers


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


@tf.keras.utils.register_keras_serializable(package="deep_layers")
class RestormerBlock(layers.Layer):
    """Restormer transformer block: MDTA attention + gated feed-forward.

    Both sub-blocks use pre-norm residual connections, standard
    transformer style:

    - **MDTA** (Multi-Dconv Head Transposed Attention): a 1x1 conv +
      depthwise 3x3 conv produce ``(q, k, v)``; attention is computed as a
      ``(head_dim, head_dim)`` map per head — across channels, not spatial
      positions — so cost is ``O(H*W*head_dim^2)``, linear in image size.
    - **GDFN** (Gated-Dconv Feed-Forward Network): a 1x1 conv expands
      channels, a depthwise 3x3 conv mixes local context, then the
      expanded channels are split in half and gated (``GELU(x1) * x2``)
      before projecting back down.

    Requires ``dim`` divisible by ``num_heads``. The runtime spatial shape
    (``H``, ``W``) is read via ``tf.shape`` rather than the static shape,
    so this layer works on this project's fully dynamic ``(None, None, C)``
    inputs — same reasoning as ``_Upsample2x``/``_ResizeToMatch`` elsewhere
    in this codebase.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_expansion_factor: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_expansion_factor = ffn_expansion_factor

    def build(self, input_shape: tf.TensorShape) -> None:
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.qkv = layers.Conv2D(self.dim * 3, 1, use_bias=False)
        self.qkv_dwconv = layers.DepthwiseConv2D(3, padding="same", use_bias=False)
        self.temperature = self.add_weight(
            name="temperature",
            shape=(self.num_heads, 1, 1),
            initializer="ones",
            trainable=True,
        )
        self.proj_out = layers.Conv2D(self.dim, 1, use_bias=False)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        hidden = self.dim * self.ffn_expansion_factor
        self.ffn_in = layers.Conv2D(hidden * 2, 1, use_bias=False)
        self.ffn_dwconv = layers.DepthwiseConv2D(3, padding="same", use_bias=False)
        self.ffn_out = layers.Conv2D(self.dim, 1, use_bias=False)
        super().build(input_shape)

    def _attention(self, x: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(x)
        b, h, w = shape[0], shape[1], shape[2]
        head_dim = self.dim // self.num_heads

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = tf.split(qkv, 3, axis=-1)

        def reshape_heads(t: tf.Tensor) -> tf.Tensor:
            t = tf.reshape(t, [b, h * w, self.num_heads, head_dim])
            return tf.transpose(t, [0, 2, 3, 1])  # (b, heads, head_dim, hw)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)
        q = tf.math.l2_normalize(q, axis=-1)
        k = tf.math.l2_normalize(k, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.temperature
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, v)  # (b, heads, head_dim, hw)
        out = tf.transpose(out, [0, 3, 1, 2])  # (b, hw, heads, head_dim)
        out = tf.reshape(out, [b, h, w, self.dim])
        return self.proj_out(out)

    def _feed_forward(self, x: tf.Tensor) -> tf.Tensor:
        x = self.ffn_dwconv(self.ffn_in(x))
        x1, x2 = tf.split(x, 2, axis=-1)
        x = tf.nn.gelu(x1) * x2
        return self.ffn_out(x)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs + self._attention(self.norm1(inputs))
        x = x + self._feed_forward(self.norm2(x))
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "ffn_expansion_factor": self.ffn_expansion_factor,
            }
        )
        return config


def build_unet_restormer(
    filters: list[int] | None = None,
    bottleneck: int = 1024,
    num_heads: int = 8,
    ffn_expansion_factor: int = 2,
) -> tf.keras.Model:
    """Build a 4-level UNet with a Restormer block at the bottleneck.

    Same topology as :func:`scripts.unet.build_unet` (fully convolutional,
    input H/W must be divisible by ``2 ** len(filters)``), with a single
    :class:`RestormerBlock` inserted after the bottleneck's ``_conv_block``.

    Parameters
    ----------
    filters : list[int] or None
        Number of filters per encoder level.
        Defaults to ``[64, 128, 256, 512]``.
    bottleneck : int
        Number of filters in the bottleneck block. Must be divisible by
        ``num_heads``.
    num_heads : int
        Number of attention heads in the bottleneck's ``RestormerBlock``.
    ffn_expansion_factor : int
        Channel expansion factor in the bottleneck's gated feed-forward
        network.

    Returns
    -------
    tf.keras.Model
        Model with input shape ``(None, None, None, 3)`` and output
        shape ``(None, None, None, 1)`` with sigmoid activation.
    """
    if filters is None:
        filters = [64, 128, 256, 512]

    inputs = layers.Input(shape=(None, None, 3))

    skips: list[tf.Tensor] = []
    x = inputs
    for f in filters:
        x = _conv_block(x, f)
        skips.append(x)
        x = layers.MaxPool2D(2)(x)

    x = _conv_block(x, bottleneck)
    x = RestormerBlock(
        dim=bottleneck, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor
    )(x)

    for f, skip in zip(reversed(filters), reversed(skips)):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, f)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="unet_restormer")
