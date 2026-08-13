"""Custom loss functions for IR image prediction."""

from collections.abc import Callable

import tensorflow as tf

# ---------------------------------------------------------------------------
# Helpers (Laplacian pyramid)
# ---------------------------------------------------------------------------


def _downsample(x: tf.Tensor) -> tf.Tensor:
    return tf.nn.avg_pool2d(x, ksize=2, strides=2, padding="VALID")


def _upsample_to(low: tf.Tensor, ref: tf.Tensor) -> tf.Tensor:
    h = tf.shape(ref)[1]
    w = tf.shape(ref)[2]
    return tf.image.resize(low, [h, w], method="bilinear")


def _laplacian_detail(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Return (detail, low) where detail = x − upsample(downsample(x))."""
    low = _downsample(x)
    detail = x - _upsample_to(low, x)
    return detail, low


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def combined_loss(alpha: float = 0.7) -> Callable:
    """Return a combined MAE + (1 - SSIM) loss function.

    The loss is defined as::

        loss = alpha * MAE(y_true, y_pred)
             + (1 - alpha) * (1 - SSIM(y_true, y_pred))

    MAE encourages pixel-level fidelity; the SSIM term preserves
    perceptual structure (edges, texture contrast).

    Parameters
    ----------
    alpha : float
        Weight for the MAE term. ``(1 - alpha)`` weights SSIM.
        Must be in ``[0, 1]``.

    Returns
    -------
    Callable
        Loss function with signature ``(y_true, y_pred) -> tf.Tensor``
        compatible with ``model.compile(loss=...)``.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return alpha * mae + (1.0 - alpha) * (1.0 - ssim)

    loss.__name__ = "combined_loss"
    return loss


def laplacian_pyramid_loss(
    levels: int = 4,
    weights: list[float] | None = None,
) -> Callable:
    """Return a multi-scale Laplacian pyramid loss.

    Decomposes both ``y_true`` and ``y_pred`` into a Laplacian pyramid
    and sums the mean absolute error at each scale.  High-frequency
    levels receive larger weights by default (geometric sequence
    ``2^(levels-1), …, 1``), making the loss sensitive to fine detail
    — the spatial frequency band where underdrawing strokes live.

    Parameters
    ----------
    levels : int
        Number of pyramid levels.
    weights : list[float] or None
        Per-level weights (length ``levels``), applied from finest to
        coarsest.  Normalised to sum to 1.  Defaults to a geometric
        sequence that halves each level.

    Returns
    -------
    Callable
        Loss function ``(y_true, y_pred) -> tf.Tensor``.
    """
    if weights is None:
        raw = [2 ** (levels - 1 - i) for i in range(levels)]
        total = sum(raw)
        weights = [w / total for w in raw]

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        level_losses: list[tf.Tensor] = []
        x_true, x_pred = y_true, y_pred
        for i in range(levels):
            detail_true, x_true = _laplacian_detail(x_true)
            detail_pred, x_pred = _laplacian_detail(x_pred)
            level_losses.append(
                weights[i] * tf.reduce_mean(tf.abs(detail_true - detail_pred))
            )
        return tf.add_n(level_losses)

    loss.__name__ = "laplacian_pyramid_loss"
    return loss


def fft_loss() -> Callable:
    """Return a frequency-domain magnitude spectrum loss.

    Computes the mean absolute difference between the 2D FFT magnitude
    spectra of ``y_true`` and ``y_pred``.  Unlike pixel-wise losses,
    this term penalises spectral errors uniformly across all frequencies,
    preventing the model from sacrificing high-frequency accuracy to
    minimise low-frequency error.

    The real-input transform (``rfft2d``) returns only the non-redundant
    half of the spectrum; the conjugate-symmetric half carries no extra
    information, so the mean over the retained bins still weights every
    frequency once.  FFT magnitudes scale with the number of pixels
    (the DC term equals the pixel sum), so the result is divided by the
    image area ``H * W`` to stay comparable across input sizes.

    Returns
    -------
    Callable
        Loss function ``(y_true, y_pred) -> tf.Tensor``.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        yt = tf.squeeze(y_true, axis=-1)  # (B, H, W)
        yp = tf.squeeze(y_pred, axis=-1)

        mag_true = tf.abs(tf.signal.rfft2d(yt))
        mag_pred = tf.abs(tf.signal.rfft2d(yp))

        h = tf.cast(tf.shape(y_true)[1], tf.float32)
        w = tf.cast(tf.shape(y_true)[2], tf.float32)
        area = h * w
        return tf.reduce_mean(tf.abs(mag_true - mag_pred)) / area

    loss.__name__ = "fft_loss"
    return loss


def combined_loss_advanced(
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    laplacian_levels: int = 4,
) -> Callable:
    """Return a combined MAE + Laplacian pyramid + FFT loss.

    Designed for the EfficientNet UNet where a pretrained encoder
    provides richer features; the additional frequency-domain terms
    recover the high-frequency detail that MAE tends to suppress.

    The loss is defined as::

        loss = alpha * MAE + beta * Laplacian + gamma * FFT

    Parameters
    ----------
    alpha : float
        Weight for the pixel-wise MAE term.
    beta : float
        Weight for the Laplacian pyramid term.
    gamma : float
        Weight for the FFT magnitude term.
    laplacian_levels : int
        Number of Laplacian pyramid levels.

    Returns
    -------
    Callable
        Loss function ``(y_true, y_pred) -> tf.Tensor``.
    """
    _lap = laplacian_pyramid_loss(levels=laplacian_levels)
    _fft = fft_loss()

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        return alpha * mae + beta * _lap(y_true, y_pred) + gamma * _fft(y_true, y_pred)

    loss.__name__ = "combined_loss_advanced"
    return loss


def gaussian_nll_loss(min_log_var: float = -6.0, max_log_var: float = 6.0) -> Callable:
    """Return a Gaussian negative-log-likelihood loss for heteroscedastic regression.

    For models that predict a per-pixel mean and log-variance instead of a
    single deterministic value (e.g.
    :func:`scripts.attention_unet_nll.build_attention_unet_nll`).
    ``y_pred`` is expected to carry two channels — ``mu`` in channel 0 and
    ``log_var`` in channel 1 — while ``y_true`` remains the single-channel
    ground-truth IR image.

    The loss is::

        loss = 0.5 * exp(-log_var) * (y_true - mu)^2 + 0.5 * log_var

    which is, up to an additive constant, the negative log-likelihood of
    ``y_true`` under a per-pixel Gaussian ``N(mu, exp(log_var))``. The
    first term is an uncertainty-weighted squared error (low ``log_var``
    sharpens the penalty, high ``log_var`` softens it); the second term
    penalises inflating ``log_var`` to trivially shrink the first —
    without it the model could minimise the loss by predicting infinite
    uncertainty everywhere instead of an accurate ``mu``.

    Parameters
    ----------
    min_log_var : float
        Lower clip bound applied to ``log_var`` before computing the loss,
        for numerical stability.
    max_log_var : float
        Upper clip bound applied to ``log_var``.

    Returns
    -------
    Callable
        Loss function ``(y_true, y_pred) -> tf.Tensor`` compatible with
        ``model.compile(loss=...)``, where ``y_pred`` has 2 channels.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mu = y_pred[..., 0:1]
        log_var = tf.clip_by_value(y_pred[..., 1:2], min_log_var, max_log_var)
        precision = tf.exp(-log_var)
        return tf.reduce_mean(0.5 * precision * tf.square(y_true - mu) + 0.5 * log_var)

    loss.__name__ = "gaussian_nll_loss"
    return loss


def beta_gaussian_nll_loss(
    beta: float = 0.5, min_log_var: float = -6.0, max_log_var: float = 6.0
) -> Callable:
    """Return a beta-weighted Gaussian NLL loss (Seitzer et al. 2022).

    Same heteroscedastic setup as :func:`gaussian_nll_loss` (``y_pred``
    carries ``mu`` in channel 0, ``log_var`` in channel 1), but each
    pixel's loss is weighted by ``sigma^(2*beta)`` with ``sigma`` detached
    from the gradient (``tf.stop_gradient``)::

        loss = stop_gradient(exp(log_var))^beta
               * (0.5 * exp(-log_var) * (y_true - mu)^2 + 0.5 * log_var)

    Plain Gaussian NLL naturally back-propagates a weaker gradient into
    ``mu`` wherever the model has predicted a high variance, which can
    starve high-uncertainty regions of learning signal and stall ``mu``
    there. Weighting by ``sigma^(2*beta)`` (with the weight itself
    excluded from the gradient) counteracts this without changing what
    the loss is minimizing at convergence. ``beta = 0`` recovers the
    plain Gaussian NLL exactly; ``beta = 1`` recovers plain MSE weighting
    (uncertainty ignored in the gradient magnitude). See
    ``code-review.md`` §7.6 and Seitzer et al., *"On the Pitfalls of
    Heteroscedastic Uncertainty Estimation with Probabilistic Neural
    Networks,"* ICLR 2022 (https://doi.org/10.48550/arXiv.2203.09168).

    Parameters
    ----------
    beta : float
        Weighting exponent in ``[0, 1]``. ``0`` = plain NLL, ``1`` = MSE-like
        gradient weighting.
    min_log_var : float
        Lower clip bound applied to ``log_var`` before computing the loss,
        for numerical stability.
    max_log_var : float
        Upper clip bound applied to ``log_var``.

    Returns
    -------
    Callable
        Loss function ``(y_true, y_pred) -> tf.Tensor`` compatible with
        ``model.compile(loss=...)``, where ``y_pred`` has 2 channels.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mu = y_pred[..., 0:1]
        log_var = tf.clip_by_value(y_pred[..., 1:2], min_log_var, max_log_var)
        precision = tf.exp(-log_var)
        nll = 0.5 * precision * tf.square(y_true - mu) + 0.5 * log_var
        weight = tf.stop_gradient(tf.exp(log_var)) ** beta
        return tf.reduce_mean(weight * nll)

    loss.__name__ = "beta_gaussian_nll_loss"
    return loss
