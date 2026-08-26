"""Dilated (ASPP-style) bottleneck block shared by the Round 3 variants.

``fixing.md`` #7 ("dilated convolution never tried") flags a gap: this
project's encoders only widen their receptive field by downsampling
(``MaxPool2D`` / strided conv), which is also the mechanism that destroys
the exact spatial position of the thin underdrawing strokes the whole
pipeline exists to detect (``theory-links.md`` §3.4's "position loss").
Dilated convolution grows the receptive field *without* downsampling —
L05 slide 19's stated purpose.

This module applies that idea at the bottleneck only, leaving the encoder/
decoder downsampling path of ``unet``/``unet_v2`` completely unchanged
(``fixing.md``'s Round 3 is explicitly "new variants ... not a fix to an
existing model" — the comparison that matters is dilated-bottleneck vs.
plain-bottleneck at otherwise identical depth, not a redesign of how much
the encoder downsamples). Several parallel branches at increasing dilation
rates (the ``atrous spatial pyramid pooling`` pattern from DeepLab) are
concatenated and projected back to ``filters`` channels, so the block sees
multiple receptive-field scales at once instead of committing to one.
"""

import tensorflow as tf
from tensorflow.keras import layers

from scripts.norm_utils import num_groups as _num_groups
from scripts.norm_utils import relu as _relu


def dilated_bottleneck(
    x: tf.Tensor,
    filters: int,
    dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
) -> tf.Tensor:
    """Multi-scale dilated bottleneck: parallel dilated convs, concat, project.

    Each branch is a single ``Conv2D(filters, 3, dilation_rate=rate) ->
    GroupNorm -> ReLU``; branches are concatenated along the channel axis
    and projected back to ``filters`` channels with a ``1x1`` conv (also
    ``GroupNorm -> ReLU``), so the block's output shape matches a plain
    ``_conv_block(x, filters)`` bottleneck and can be swapped in directly.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map (the encoder's deepest output).
    filters : int
        Number of output channels, matching the non-dilated bottleneck's
        ``bottleneck`` parameter.
    dilation_rates : tuple[int, ...]
        Dilation rate for each parallel branch. ``rate=1`` is a plain
        (non-dilated) ``3x3`` conv, included so the block also captures the
        finest local scale alongside the widened ones.

    Returns
    -------
    tf.Tensor
        Output feature map with shape ``(..., H, W, filters)``.
    """
    branches = []
    for rate in dilation_rates:
        b = layers.Conv2D(
            filters,
            3,
            padding="same",
            dilation_rate=rate,
            use_bias=False,
            kernel_initializer="he_normal",
        )(x)
        b = layers.GroupNormalization(groups=_num_groups(filters))(b)
        b = _relu(b)
        branches.append(b)

    x = layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    x = layers.Conv2D(
        filters, 1, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = layers.GroupNormalization(groups=_num_groups(filters))(x)
    x = _relu(x)
    return x
