"""TF-native data augmentation applied consistently to RGB / IR pairs."""

import tensorflow as tf


def augment_pair(
    rgb: tf.Tensor,
    ir: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply random augmentation to an RGB / IR image pair.

    Spatial transforms (flips) are applied identically to both images.
    Photometric jitter (brightness, contrast) is applied to RGB only —
    IR reflectance is a physical property independent of illumination.

    Parameters
    ----------
    rgb : tf.Tensor
        RGB image of shape ``(H, W, 3)``, values in ``[0, 1]``.
    ir : tf.Tensor
        IR image of shape ``(H, W, 1)``, values in ``[0, 1]``.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
        Augmented ``(rgb, ir)`` pair with the same shapes as input.
    """
    flip_h = tf.random.uniform(()) > 0.5
    flip_v = tf.random.uniform(()) > 0.5

    rgb = tf.cond(flip_h, lambda: tf.image.flip_left_right(rgb), lambda: rgb)
    rgb = tf.cond(flip_v, lambda: tf.image.flip_up_down(rgb), lambda: rgb)
    ir = tf.cond(flip_h, lambda: tf.image.flip_left_right(ir), lambda: ir)
    ir = tf.cond(flip_v, lambda: tf.image.flip_up_down(ir), lambda: ir)

    rgb = tf.image.random_brightness(rgb, max_delta=0.1)
    rgb = tf.image.random_contrast(rgb, lower=0.9, upper=1.1)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)

    return rgb, ir
