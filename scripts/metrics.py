"""Keras metric wrappers for image quality assessment."""

import tensorflow as tf


class PSNRMetric(tf.keras.metrics.Metric):
    """Peak Signal-to-Noise Ratio metric for image batches.

    Parameters
    ----------
    name : str
        Metric name displayed during training.
    """

    def __init__(self, name: str = "psnr", **kwargs: object) -> None:
        super().__init__(name=name, **kwargs)
        self._sum = self.add_weight(name="sum", initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Accumulate PSNR over a batch.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground-truth images, shape ``(B, H, W, C)``.
        y_pred : tf.Tensor
            Predicted images, shape ``(B, H, W, C)``.
        sample_weight : tf.Tensor or None
            Unused; kept for Keras API compatibility.
        """
        psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
        self._sum.assign_add(tf.reduce_sum(psnr))
        self._count.assign_add(tf.cast(tf.size(psnr), tf.float32))

    def result(self) -> tf.Tensor:
        """Return mean PSNR over accumulated batches."""
        return self._sum / self._count

    def reset_state(self) -> None:
        """Reset accumulators to zero."""
        self._sum.assign(0.0)
        self._count.assign(0.0)


class MuPSNRMetric(tf.keras.metrics.Metric):
    """PSNR metric that reads ``mu`` from a 2-channel heteroscedastic prediction.

    For use with models such as
    :func:`scripts.attention_unet_nll.build_attention_unet_nll`, whose
    ``y_pred`` carries ``(mu, log_var)`` in channels 0 and 1 — only ``mu``
    is an image-shaped prediction comparable to ``y_true``.

    Parameters
    ----------
    name : str
        Metric name displayed during training.
    """

    def __init__(self, name: str = "psnr", **kwargs: object) -> None:
        super().__init__(name=name, **kwargs)
        self._sum = self.add_weight(name="sum", initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Accumulate PSNR over a batch, using only the ``mu`` channel of ``y_pred``.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground-truth images, shape ``(B, H, W, 1)``.
        y_pred : tf.Tensor
            Predicted ``(mu, log_var)``, shape ``(B, H, W, 2)``.
        sample_weight : tf.Tensor or None
            Unused; kept for Keras API compatibility.
        """
        mu = y_pred[..., 0:1]
        psnr = tf.image.psnr(y_true, mu, max_val=1.0)
        self._sum.assign_add(tf.reduce_sum(psnr))
        self._count.assign_add(tf.cast(tf.size(psnr), tf.float32))

    def result(self) -> tf.Tensor:
        """Return mean PSNR over accumulated batches."""
        return self._sum / self._count

    def reset_state(self) -> None:
        """Reset accumulators to zero."""
        self._sum.assign(0.0)
        self._count.assign(0.0)


class MuSSIMMetric(tf.keras.metrics.Metric):
    """SSIM metric that reads ``mu`` from a 2-channel heteroscedastic prediction.

    See :class:`MuPSNRMetric` for why only the ``mu`` channel is used.

    Parameters
    ----------
    name : str
        Metric name displayed during training.
    """

    def __init__(self, name: str = "ssim", **kwargs: object) -> None:
        super().__init__(name=name, **kwargs)
        self._sum = self.add_weight(name="sum", initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Accumulate SSIM over a batch, using only the ``mu`` channel of ``y_pred``.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground-truth images, shape ``(B, H, W, 1)``.
        y_pred : tf.Tensor
            Predicted ``(mu, log_var)``, shape ``(B, H, W, 2)``.
        sample_weight : tf.Tensor or None
            Unused; kept for Keras API compatibility.
        """
        mu = y_pred[..., 0:1]
        ssim = tf.image.ssim(y_true, mu, max_val=1.0)
        self._sum.assign_add(tf.reduce_sum(ssim))
        self._count.assign_add(tf.cast(tf.size(ssim), tf.float32))

    def result(self) -> tf.Tensor:
        """Return mean SSIM over accumulated batches."""
        return self._sum / self._count

    def reset_state(self) -> None:
        """Reset accumulators to zero."""
        self._sum.assign(0.0)
        self._count.assign(0.0)


class MuMAEMetric(tf.keras.metrics.Metric):
    """MAE metric that reads ``mu`` from a 2-channel heteroscedastic prediction.

    See :class:`MuPSNRMetric` for why only the ``mu`` channel is used.

    Parameters
    ----------
    name : str
        Metric name displayed during training.
    """

    def __init__(self, name: str = "mae", **kwargs: object) -> None:
        super().__init__(name=name, **kwargs)
        self._sum = self.add_weight(name="sum", initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Accumulate MAE over a batch, using only the ``mu`` channel of ``y_pred``.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground-truth images, shape ``(B, H, W, 1)``.
        y_pred : tf.Tensor
            Predicted ``(mu, log_var)``, shape ``(B, H, W, 2)``.
        sample_weight : tf.Tensor or None
            Unused; kept for Keras API compatibility.
        """
        mu = y_pred[..., 0:1]
        error = tf.abs(y_true - mu)
        self._sum.assign_add(tf.reduce_sum(error))
        self._count.assign_add(tf.cast(tf.size(error), tf.float32))

    def result(self) -> tf.Tensor:
        """Return mean MAE over accumulated batches."""
        return self._sum / self._count

    def reset_state(self) -> None:
        """Reset accumulators to zero."""
        self._sum.assign(0.0)
        self._count.assign(0.0)


class SSIMMetric(tf.keras.metrics.Metric):
    """Structural Similarity Index metric for image batches.

    Parameters
    ----------
    name : str
        Metric name displayed during training.
    """

    def __init__(self, name: str = "ssim", **kwargs: object) -> None:
        super().__init__(name=name, **kwargs)
        self._sum = self.add_weight(name="sum", initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Accumulate SSIM over a batch.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground-truth images, shape ``(B, H, W, C)``.
        y_pred : tf.Tensor
            Predicted images, shape ``(B, H, W, C)``.
        sample_weight : tf.Tensor or None
            Unused; kept for Keras API compatibility.
        """
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        self._sum.assign_add(tf.reduce_sum(ssim))
        self._count.assign_add(tf.cast(tf.size(ssim), tf.float32))

    def result(self) -> tf.Tensor:
        """Return mean SSIM over accumulated batches."""
        return self._sum / self._count

    def reset_state(self) -> None:
        """Reset accumulators to zero."""
        self._sum.assign(0.0)
        self._count.assign(0.0)
