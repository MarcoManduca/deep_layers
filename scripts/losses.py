"""Custom loss functions for IR image prediction."""

from collections.abc import Callable

import numpy as np
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
# Helpers (local windowed moments)
# ---------------------------------------------------------------------------


def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """Return a normalised 1D Gaussian kernel of length ``size``.

    Mirrors ``scripts.delta_analysis._gaussian_kernel_1d`` so the loss and
    the post-hoc delta analysis window the image identically. The two are
    pinned together by a numerical-agreement test.

    Parameters
    ----------
    size : int
        Kernel length in pixels.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    np.ndarray
        Kernel of shape ``(size,)`` summing to 1.
    """
    coords = np.arange(size) - size // 2
    kernel = np.exp(-(coords**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _gaussian_blur(x: tf.Tensor, kernel_1d: tf.Tensor, channels: int) -> tf.Tensor:
    """Apply a separable Gaussian blur with ``reflect`` padding, per channel.

    Uses two depthwise convolutions (vertical then horizontal) on a
    reflect-padded input, so the output keeps the input's spatial size and
    matches a single ``np.pad(..., mode="reflect")`` followed by two
    ``mode="valid"`` 1D convolutions.

    Parameters
    ----------
    x : tf.Tensor
        Input of shape ``(B, H, W, channels)``.
    kernel_1d : tf.Tensor
        Normalised 1D Gaussian kernel of shape ``(size,)``.
    channels : int
        Static channel count of ``x``; each channel is blurred independently.

    Returns
    -------
    tf.Tensor
        Blurred tensor of shape ``(B, H, W, channels)``.

    Notes
    -----
    ``reflect`` padding requires ``H`` and ``W`` to exceed ``size // 2``;
    with the default ``size=11`` window, inputs must be at least 6x6.
    """
    size = int(kernel_1d.shape[0])
    pad = size // 2
    padded = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="REFLECT")

    vertical = tf.tile(tf.reshape(kernel_1d, [size, 1, 1, 1]), [1, 1, channels, 1])
    horizontal = tf.tile(tf.reshape(kernel_1d, [1, size, 1, 1]), [1, 1, channels, 1])

    blurred = tf.nn.depthwise_conv2d(padded, vertical, [1, 1, 1, 1], "VALID")
    return tf.nn.depthwise_conv2d(blurred, horizontal, [1, 1, 1, 1], "VALID")


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


def local_zscore_loss(
    window_size: int = 11,
    sigma: float = 1.5,
    sigma_floor: float = 0.01,
    clip_value: float | None = None,
) -> Callable:
    """Return a per-window z-score normalised L1 loss.

    Both images are normalised by their own local (Gaussian-windowed) mean
    and standard deviation before being compared::

        z = (x - mu_local) / max(sigma_local, sigma_floor)
        loss = mean |z_true - z_pred|

    This measures the error in units of *local contrast* rather than in
    absolute gray levels, which is the metric the reflectography delta is
    actually read in — ``040_inference.ipynb`` passes the delta through
    CLAHE (locally adaptive equalisation) and ``analyze_delta`` min-max
    normalises it, so a 0.01 error in a flat passage becomes as visible as
    a much larger error in a high-contrast one. Plain MAE weights those two
    equally and therefore under-penalises the first.

    Two properties are important to understand before using this term:

    - **It is invariant to local affine changes.** If ``pred = a * real + b``
      locally with ``a > 0``, then ``z_pred == z_real`` and the loss is
      exactly zero. The absolute gray level is therefore *unconstrained*, so
      this loss must never be used on its own — see
      :func:`combined_loss_normalized`, which anchors it with MAE.
    - **It concentrates weight on unpredictable structure.** Where the target
      has local structure the model cannot infer from RGB, ``z_pred -> 0``
      while ``z_true`` stays ``O(1)``, so this term reports ~1.0 where MAE
      reports ~0.01. Measured on held-out paintings, ``mean|z_true - z_pred|``
      is roughly **9x** the MAE of the same prediction. Weight it accordingly
      and use ``clip_value`` to bound the tail.

    Parameters
    ----------
    window_size : int
        Side length of the square Gaussian window. Same default as
        ``tf.image.ssim`` and ``scripts.delta_analysis.compute_local_stats``.
        Inputs must be larger than ``window_size // 2`` in both spatial
        dimensions (reflect padding).
    sigma : float
        Standard deviation of the Gaussian window.
    sigma_floor : float
        Lower bound on the local standard deviation used as denominator.
        Its main job is conditioning the gradient: the derivative of the
        ``sqrt`` is bounded by ``1 / (2 * sigma_floor)``, and the flat-region
        gradient blow-up of an unfloored z-score is what would otherwise
        produce NaNs. The default ``0.01`` is the 10th percentile of the
        local standard deviation measured over the project's IR images, so
        it clamps roughly the flattest tenth of pixels; between ``0.005``
        and ``0.02`` the loss value itself moves by well under 1%.
    clip_value : float or None
        If set, clip the per-pixel normalised error at this value to bound
        the contribution of the extreme tail. ``None`` disables clipping.
        Measured p99 of the per-pixel error is ~2.8 and the max ~5.1, so a
        clip around ``4.0`` touches under 1% of pixels.

    Returns
    -------
    Callable
        Loss function ``(y_true, y_pred) -> tf.Tensor``.
    """
    kernel = tf.constant(_gaussian_kernel_1d(window_size, sigma), dtype=tf.float32)
    var_floor = sigma_floor**2

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        channels = y_pred.shape[-1]
        stacked = tf.concat([y_true, y_pred, y_true * y_true, y_pred * y_pred], axis=-1)
        blurred = _gaussian_blur(stacked, kernel, 4 * channels)
        mu_true, mu_pred, sq_true, sq_pred = tf.split(blurred, 4, axis=-1)

        # Flooring the variance (rather than the standard deviation) keeps the
        # sqrt argument away from zero, where its derivative is unbounded.
        std_true = tf.sqrt(tf.maximum(sq_true - mu_true**2, var_floor))
        std_pred = tf.sqrt(tf.maximum(sq_pred - mu_pred**2, var_floor))

        z_true = (y_true - mu_true) / std_true
        z_pred = (y_pred - mu_pred) / std_pred

        error = tf.abs(z_true - z_pred)
        if clip_value is not None:
            error = tf.minimum(error, clip_value)
        return tf.reduce_mean(error)

    loss.__name__ = "local_zscore_loss"
    return loss


def combined_loss_normalized(
    alpha: float = 1.0,
    beta: float = 0.03,
    window_size: int = 11,
    sigma: float = 1.5,
    sigma_floor: float = 0.01,
    clip_value: float | None = 4.0,
) -> Callable:
    """Return a combined MAE + per-window z-score loss.

    Implements the "local normalization" idea from ``note.md`` §2 as a
    training objective::

        loss = alpha * MAE + beta * local_zscore

    The MAE term is **not optional**: :func:`local_zscore_loss` is invariant
    to local affine gray-level changes, so without an absolute anchor the
    model would be free to get the gray level wrong and the raw
    ``|real - predicted|`` delta would lose its meaning.

    Parameters
    ----------
    alpha : float
        Weight for the pixel-wise MAE term.
    beta : float
        Weight for the normalised term. Note the two terms are **not** on a
        comparable scale — the normalised term measures error in units of
        local standard deviation and runs ~9x larger than MAE on real
        predictions, so ``alpha`` and ``beta`` are independent weights, not
        a convex split. The defaults put the normalised term at roughly
        20-25% of the total loss at currently observed error levels.
    window_size : int
        Side length of the square Gaussian window.
    sigma : float
        Standard deviation of the Gaussian window.
    sigma_floor : float
        Lower bound on the local standard deviation; see
        :func:`local_zscore_loss`.
    clip_value : float or None
        Per-pixel clip on the normalised error; see
        :func:`local_zscore_loss`.

    Returns
    -------
    Callable
        Loss function ``(y_true, y_pred) -> tf.Tensor``.
    """
    _zscore = local_zscore_loss(
        window_size=window_size,
        sigma=sigma,
        sigma_floor=sigma_floor,
        clip_value=clip_value,
    )

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        return alpha * mae + beta * _zscore(y_true, y_pred)

    loss.__name__ = "combined_loss_normalized"
    return loss
