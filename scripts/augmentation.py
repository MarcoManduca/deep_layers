"""TF-native data augmentation applied consistently to RGB / IR pairs."""

import tensorflow as tf


def augment_pair(
    rgb: tf.Tensor,
    ir: tf.Tensor,
    seed: tf.Tensor,
    crop_size: int | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply random augmentation to an RGB / IR image pair.

    When ``crop_size`` is set, the pair is randomly cropped to a square of
    that side using a single shared crop box, so RGB and IR stay pixel-
    aligned. Spatial flips are then applied identically to both images.
    Photometric jitter (brightness, contrast) is applied to RGB only —
    IR reflectance is a physical property independent of illumination.

    All randomness is **stateless**: it derives solely from ``seed``, so
    a given ``seed`` always produces the same augmentation. Combined with
    a deterministic per-element seed (see
    :func:`scripts.dataset.build_dataset`), this makes the augmented
    training stream fully reproducible across runs.

    Parameters
    ----------
    rgb : tf.Tensor
        RGB image of shape ``(H, W, 3)``, values in ``[0, 1]``.
    ir : tf.Tensor
        IR image of shape ``(H, W, 1)``, values in ``[0, 1]``.
    seed : tf.Tensor
        Stateless RNG seed of shape ``(2,)`` and dtype ``int32``/``int64``.
    crop_size : int or None
        If set, randomly crop the pair to ``(crop_size, crop_size)`` before
        the other transforms. ``None`` leaves the spatial size unchanged.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
        Augmented ``(rgb, ir)`` pair. Spatial size equals ``crop_size`` when
        cropping is enabled, otherwise the input size.
    """
    # One independent sub-seed per random operation, all derived from `seed`.
    seeds = tf.random.experimental.stateless_split(seed, num=6)

    if crop_size is not None:
        # Crop RGB and IR together (stacked on the channel axis) so both share
        # the exact same crop box and stay pixel-aligned. The offset is drawn
        # from a scalar float stateless uniform rather than
        # tf.image.stateless_random_crop, whose vector-maxval integer sampling
        # is unsupported on the Metal backend.
        stacked = tf.concat([rgb, ir], axis=-1)  # (H, W, 4)
        shape = tf.shape(stacked)
        max_y = tf.cast(shape[0] - crop_size + 1, tf.float32)
        max_x = tf.cast(shape[1] - crop_size + 1, tf.float32)
        ry = tf.random.stateless_uniform((), seed=seeds[4])
        rx = tf.random.stateless_uniform((), seed=seeds[5])
        off_y = tf.cast(ry * max_y, tf.int32)
        off_x = tf.cast(rx * max_x, tf.int32)
        stacked = stacked[off_y : off_y + crop_size, off_x : off_x + crop_size, :]
        rgb, ir = stacked[..., :3], stacked[..., 3:]

    flip_h = tf.random.stateless_uniform((), seed=seeds[0]) > 0.5
    flip_v = tf.random.stateless_uniform((), seed=seeds[1]) > 0.5

    rgb = tf.cond(flip_h, lambda: tf.image.flip_left_right(rgb), lambda: rgb)
    rgb = tf.cond(flip_v, lambda: tf.image.flip_up_down(rgb), lambda: rgb)
    ir = tf.cond(flip_h, lambda: tf.image.flip_left_right(ir), lambda: ir)
    ir = tf.cond(flip_v, lambda: tf.image.flip_up_down(ir), lambda: ir)

    rgb = tf.image.stateless_random_brightness(rgb, max_delta=0.1, seed=seeds[2])
    rgb = tf.image.stateless_random_contrast(rgb, lower=0.9, upper=1.1, seed=seeds[3])
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)

    return rgb, ir
