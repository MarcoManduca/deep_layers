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


def charbonnier_loss(epsilon: float = 1e-3) -> Callable:
    """Return a Charbonnier (smooth L1) loss function.

    ``sqrt((y_true - y_pred)^2 + epsilon^2) - epsilon``: a differentiable
    approximation of L1 that avoids the undefined gradient L1 has at zero
    error, standard in image-restoration literature (e.g. LapSRN). Prefer
    this over raw L1/MAE for the fidelity term of :func:`combined_loss`.

    Parameters
    ----------
    epsilon : float
        Smoothing constant.

    Returns
    -------
    Callable
        Loss function ``(y_true, y_pred) -> tf.Tensor``.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        diff = y_true - y_pred
        return tf.reduce_mean(tf.sqrt(tf.square(diff) + epsilon**2) - epsilon)

    loss.__name__ = "charbonnier_loss"
    return loss


def ms_ssim_loss() -> Callable:
    """Return a ``1 - MS-SSIM`` loss function.

    Multi-scale SSIM (Wang et al., 2003) resolves the single-scale
    trade-off ``tf.image.ssim`` has: a small filter preserves edges but
    reintroduces flat-region artifacts, a large one does the opposite
    (Zhao et al. 2016, §V-B). MS-SSIM pools several scales instead of
    forcing one choice.

    Returns
    -------
    Callable
        Loss function ``(y_true, y_pred) -> tf.Tensor``.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0))
        return 1.0 - ms_ssim

    loss.__name__ = "ms_ssim_loss"
    return loss


def combined_loss(alpha: float = 0.16, epsilon: float = 1e-3) -> Callable:
    """Return a combined Charbonnier + (1 - MS-SSIM) loss function.

    The loss is defined as::

        loss = alpha * Charbonnier(y_true, y_pred)
             + (1 - alpha) * (1 - MS-SSIM(y_true, y_pred))

    Matches the best-performing loss (``Mix``) in Zhao, Gallo, Frosio &
    Kautz, *"Loss Functions for Image Restoration with Neural Networks"*
    (2016, `biblio/1511.08861v3.pdf`): ``alpha=0.16`` weights the
    perceptual MS-SSIM term (84%) over the pixel-fidelity Charbonnier term
    (16%) — the paper's own best-performing ratio, empirically the
    opposite of this loss's pre-Round-1 weighting (``alpha=0.7`` favouring
    plain MAE). See ``fixing.md`` #9.

    Parameters
    ----------
    alpha : float
        Weight for the Charbonnier term. ``(1 - alpha)`` weights MS-SSIM.
        Must be in ``[0, 1]``.
    epsilon : float
        Charbonnier smoothing constant, see :func:`charbonnier_loss`.

    Returns
    -------
    Callable
        Loss function with signature ``(y_true, y_pred) -> tf.Tensor``
        compatible with ``model.compile(loss=...)``.
    """
    _charbonnier = charbonnier_loss(epsilon=epsilon)
    _ms_ssim = ms_ssim_loss()

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return alpha * _charbonnier(y_true, y_pred) + (1.0 - alpha) * _ms_ssim(
            y_true, y_pred
        )

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


def laplace_nll_loss(
    beta: float = 0.5, min_log_b: float = -6.0, max_log_b: float = 6.0
) -> Callable:
    """Return a beta-weighted Laplace negative-log-likelihood loss.

    The Laplace counterpart of :func:`beta_gaussian_nll_loss`: same
    heteroscedastic ``(mu, log_b)`` setup (``y_pred`` carries ``mu`` in
    channel 0, a log-scale in channel 1 — reusing the same head/clip
    mechanism as the Gaussian NLL models, just reinterpreted), but the
    per-pixel likelihood is Laplace (scale ``b``) instead of Gaussian
    (variance)::

        loss = stop_gradient(b)^beta * (|y_true - mu| / b + log(b))

    Motivation (``fixing.md`` #10): Zhao et al. (2016) show L1-family
    fidelity losses beat L2-family ones for image restoration (see
    :func:`combined_loss`); the same reasoning applies to the likelihood
    term of a heteroscedastic model — Laplace NLL is L1-weighted-by-scale,
    Gaussian NLL is L2-weighted-by-variance. The ``stop_gradient(b)^beta``
    weighting generalises the beta-reweighting trick from Seitzer et al.
    (2022) — defined for the Gaussian variance — to the Laplace scale;
    this is a motivated adaptation, not the literally published formula.
    ``beta = 0`` recovers the plain Laplace NLL exactly.

    This loss replaces both :func:`gaussian_nll_loss` and
    :func:`beta_gaussian_nll_loss` for `unet_nll`, `resunet_nll`, and
    `attention_unet_nll` — one NLL loss per architecture instead of two,
    collapsing the `nll_gaussian`/`nll_beta` checkpoint split for these
    three. `efficientnet_unet_nll` keeps the Gaussian losses until Round 2.

    Parameters
    ----------
    beta : float
        Weighting exponent in ``[0, 1]``. ``0`` = plain Laplace NLL,
        ``1`` = weight scales linearly with ``b``.
    min_log_b : float
        Lower clip bound applied to the log-scale channel before computing
        the loss, for numerical stability.
    max_log_b : float
        Upper clip bound applied to the log-scale channel.

    Returns
    -------
    Callable
        Loss function ``(y_true, y_pred) -> tf.Tensor`` compatible with
        ``model.compile(loss=...)``, where ``y_pred`` has 2 channels.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mu = y_pred[..., 0:1]
        log_b = tf.clip_by_value(y_pred[..., 1:2], min_log_b, max_log_b)
        b = tf.exp(log_b)
        nll = tf.abs(y_true - mu) / b + log_b
        weight = tf.stop_gradient(b) ** beta
        return tf.reduce_mean(weight * nll)

    loss.__name__ = "laplace_nll_loss"
    return loss
