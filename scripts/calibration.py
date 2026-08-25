"""Calibration diagnostics for heteroscedastic ``(mu, sigma)`` IR predictions.

``mae``/``ssim``/``psnr`` (``scripts.metrics``) read only the ``mu`` channel of
a ``(mu, log_var)`` prediction, so by construction they cannot say whether the
predicted ``sigma`` — and therefore the learned z-score
``(real_IR - mu) / sigma`` the reflectography signal is built on — is any good.
This module scores ``sigma`` on its own, post hoc, from arrays already produced
by ``model.predict``: no retraining, no change to any model, so existing
checkpoints can be compared immediately. See ``note.md`` ("evaluating the
learned z-score itself") and ``code-review.md`` §7.6.

Two properties must be read together (Gneiting et al. 2007): a model is only
useful if it is *calibrated* **and** *sharp*.

- **Calibration** — do the predicted intervals hold? Measured by
  :func:`coverage_probability` (empirical vs. nominal Gaussian coverage),
  :func:`sigma_reliability` / ENCE (per-bin predicted vs. observed error), and
  the standard deviation of the z-score (``1.0`` when calibrated).
- **Sharpness and dispersion** — :func:`sharpness` and :func:`dispersion`.
  A model that predicts a huge constant ``sigma`` everywhere is perfectly
  calibrated and completely useless: its z-score is just a rescaled raw delta,
  with none of the color/context conditioning the heteroscedastic head exists
  to provide. ``dispersion == 0`` means exactly that failure.

:func:`mean_gaussian_nll` (and its Laplace counterpart,
:func:`mean_laplace_nll`, for `unet_nll`/`resunet_nll`/`attention_unet_nll`
since ``fixing.md`` #10) is the one number that folds accuracy and
calibration together (a proper scoring rule), and is the natural
tie-breaker when ``mae``/``ssim``/``psnr`` and the calibration metrics
disagree. Every other function here (``coverage_probability``,
``sigma_reliability``, ``sharpness``, ``dispersion``, ``learned_zscore``)
takes ``sigma`` (true standard deviation) as a plain input and is
distribution-agnostic — only the NLL term differs structurally between
Gaussian and Laplace, so :func:`evaluate_calibration`'s ``distribution``
parameter dispatches only that one term.

References
----------
T. Gneiting, F. Balabdaoui, A.E. Raftery, "Probabilistic forecasts,
calibration and sharpness," *JRSS-B*, 2007.
https://doi.org/10.1111/j.1467-9868.2007.00587.x

D. Levi, L. Gispan, N. Giladi, E. Fetaya, "Evaluating and Calibrating
Uncertainty Prediction in Regression Tasks," *Sensors*, 2022.
https://doi.org/10.48550/arXiv.1905.11659 — source of the ENCE
(Expected Normalized Calibration Error) used by :func:`sigma_reliability`.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

from scripts.delta_analysis import gaussian_local_filter

_EPS = 1e-8

DEFAULT_COVERAGE_LEVELS = (1.0, 2.0, 3.0)


def learned_zscore(
    real_ir: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Compute the learned anomaly z-score ``(real_IR - mu) / sigma``.

    The single definition of the learned z-score used across the project
    (``scripts.visualization_nll``, and every metric in this module), so the
    plotted signal and the scored signal can never drift apart.

    Parameters
    ----------
    real_ir : np.ndarray
        Ground-truth IR image, shape ``(H, W)`` or ``(H, W, 1)``.
    mu : np.ndarray
        Predicted mean IR, same shape convention as ``real_ir``.
    sigma : np.ndarray
        Predicted standard deviation, same shape convention as ``real_ir``.

    Returns
    -------
    np.ndarray
        Signed z-score map, shape ``(H, W)``.
    """
    real = real_ir.squeeze().astype(np.float32)
    mean = mu.squeeze().astype(np.float32)
    scale = sigma.squeeze().astype(np.float32)
    return (real - mean) / (scale + _EPS)


def structural_zscore(
    structural_delta: np.ndarray,
    sigma: np.ndarray,
    window_size: int = 11,
    gauss_sigma: float = 1.5,
) -> np.ndarray:
    """Normalize the structural delta by the model's own local uncertainty.

    ``structural_delta`` (``scripts.delta_analysis.analyze_delta``) is the
    best-performing detection signal found so far (`033_bis`, both against
    the cross-modal pseudo-mask and against real hand-drawn ground truth) —
    but it is a purely deterministic function of ``(real_IR, mu)``, blind to
    whether the model itself found this region ambiguous. ``sigma`` is
    exactly that signal: a real paint color can plausibly map to several IR
    gray levels, and a well-trained heteroscedastic head should read that
    ambiguity as high ``sigma``, not as hidden detail. ``learned_zscore``
    already divides by ``sigma`` but normalizes ``raw_delta``, which
    `033_bis` found carries much less detection signal than the structural
    component — so this divides the *better* signal by the *same*
    uncertainty instead, keeping the benefit of both: a structurally
    discordant region the model was also confident about is *more*
    remarkable, one it was uncertain about (plausible color ambiguity) is
    discounted.

    ``sigma`` is smoothed with the same Gaussian window used to compute
    ``structural_delta`` (``scripts.delta_analysis.compute_local_stats``)
    before dividing — combining a windowed quantity with a raw per-pixel one
    would reintroduce exactly the single-pixel noise the windowing exists to
    remove.

    Parameters
    ----------
    structural_delta : np.ndarray
        ``1 - local SSIM structure`` between ``real_IR`` and ``mu``, shape
        ``(H, W)`` (``scripts.delta_analysis.analyze_delta`` or
        ``compute_ssim_components``). Already non-negative.
    sigma : np.ndarray
        Predicted standard deviation, same shape convention as
        ``structural_delta``.
    window_size, gauss_sigma : float
        Must match the window ``structural_delta`` was computed with, so the
        two maps line up pixel-for-pixel.

    Returns
    -------
    np.ndarray
        Non-negative magnitude map, shape ``(H, W)`` — ready to hand to
        ``scripts.detection.rank_signals`` without an ``abs()``, since
        ``structural_delta`` is already unsigned and ``sigma`` is positive.
    """
    scale = sigma.squeeze().astype(np.float32)
    smoothed = gaussian_local_filter(scale, window_size, gauss_sigma)
    return structural_delta / (smoothed + _EPS)


def nominal_coverage(k: float) -> float:
    """Return the fraction of a standard normal lying within ``+/- k``.

    Parameters
    ----------
    k : float
        Interval half-width in standard deviations.

    Returns
    -------
    float
        ``erf(k / sqrt(2))`` — e.g. ``0.6827`` at ``k = 1``.
    """
    return math.erf(k / math.sqrt(2.0))


@dataclass(frozen=True)
class Coverage:
    """Empirical vs. nominal coverage of a ``+/- k * sigma`` interval.

    Attributes
    ----------
    k : float
        Interval half-width in predicted standard deviations.
    empirical : float
        Fraction of pixels actually falling inside the interval.
    nominal : float
        Fraction a correctly calibrated Gaussian would put inside it.
    """

    k: float
    empirical: float
    nominal: float

    @property
    def error(self) -> float:
        """Signed miscalibration: negative = overconfident, positive = under."""
        return self.empirical - self.nominal


def coverage_probability(
    real_ir: np.ndarray, mu: np.ndarray, sigma: np.ndarray, k: float = 1.0
) -> Coverage:
    """Measure how often the truth falls inside the predicted ``+/- k * sigma``.

    The most direct reading of whether ``sigma`` means what it claims. An
    overconfident model (``sigma`` too small) covers less than nominal and
    inflates the z-score everywhere, which is indistinguishable from "hidden
    detail everywhere" on a reflectography map.

    Parameters
    ----------
    real_ir : np.ndarray
        Ground-truth IR image, shape ``(H, W)`` or ``(H, W, 1)``.
    mu : np.ndarray
        Predicted mean IR, same shape convention as ``real_ir``.
    sigma : np.ndarray
        Predicted standard deviation, same shape convention as ``real_ir``.
    k : float
        Interval half-width in predicted standard deviations.

    Returns
    -------
    Coverage
    """
    z = learned_zscore(real_ir, mu, sigma)
    empirical = float(np.mean(np.abs(z) <= k))
    return Coverage(k=float(k), empirical=empirical, nominal=nominal_coverage(k))


def error_sigma_correlation(
    real_ir: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    max_samples: int = 500_000,
    seed: int = 0,
) -> float:
    """Rank-correlate the absolute error ``|real - mu|`` with predicted ``sigma``.

    Answers "does the model know where it is wrong?" independently of whether
    the scale of ``sigma`` is right. Spearman rather than Pearson: only the
    monotone ordering matters, and per-pixel absolute errors are heavy-tailed.
    Returns ``0.0`` when either array is constant (no ordering to correlate,
    e.g. a homoscedastic ``sigma``), where Spearman is undefined.

    Parameters
    ----------
    real_ir : np.ndarray
        Ground-truth IR image, shape ``(H, W)`` or ``(H, W, 1)``.
    mu : np.ndarray
        Predicted mean IR, same shape convention as ``real_ir``.
    sigma : np.ndarray
        Predicted standard deviation, same shape convention as ``real_ir``.
    max_samples : int
        Pixels above this count are randomly subsampled before ranking, which
        dominates the cost on full paintings.
    seed : int
        Seed for the subsampling RNG, so the score is reproducible.

    Returns
    -------
    float
        Spearman correlation in ``[-1, 1]``; ``1.0`` means ``sigma`` orders the
        errors perfectly.
    """
    error = np.abs(real_ir.squeeze() - mu.squeeze()).ravel()
    scale = sigma.squeeze().ravel()

    if error.size > max_samples:
        rng = np.random.default_rng(seed)
        index = rng.choice(error.size, size=max_samples, replace=False)
        error, scale = error[index], scale[index]

    if error.std() < _EPS or scale.std() < _EPS:
        return 0.0
    return float(spearmanr(error, scale).statistic)


@dataclass(frozen=True)
class ReliabilityCurve:
    """Per-bin predicted vs. observed error, the regression reliability diagram.

    Pixels are grouped into equal-population bins of predicted ``sigma``; a
    perfectly calibrated model has ``observed_error == predicted_sigma`` in
    every bin, i.e. the curve lies on the identity line.

    Attributes
    ----------
    predicted_sigma : np.ndarray
        Root mean predicted variance per bin, shape ``(n_bins,)``.
    observed_error : np.ndarray
        Root mean squared actual error per bin, shape ``(n_bins,)``.
    counts : np.ndarray
        Number of pixels per bin, shape ``(n_bins,)``.
    """

    predicted_sigma: np.ndarray
    observed_error: np.ndarray
    counts: np.ndarray

    @property
    def ence(self) -> float:
        """Expected Normalized Calibration Error (Levi et al.).

        Mean over bins of ``|predicted - observed| / predicted``; ``0.0`` is
        perfect. Normalizing per bin keeps low-``sigma`` bins from being
        drowned out by high-``sigma`` ones.
        """
        deviation = np.abs(self.predicted_sigma - self.observed_error)
        return float(np.mean(deviation / (self.predicted_sigma + _EPS)))


def sigma_reliability(
    real_ir: np.ndarray, mu: np.ndarray, sigma: np.ndarray, n_bins: int = 10
) -> ReliabilityCurve:
    """Bin pixels by predicted ``sigma`` and compare with the observed error.

    Equal-population (quantile) bins rather than equal-width ones, so every
    point on the curve carries the same statistical weight even though
    ``sigma`` is typically very skewed. Bins left empty by ties in ``sigma``
    (common when ``log_var`` saturates against its clip bounds) are dropped.

    Parameters
    ----------
    real_ir : np.ndarray
        Ground-truth IR image, shape ``(H, W)`` or ``(H, W, 1)``.
    mu : np.ndarray
        Predicted mean IR, same shape convention as ``real_ir``.
    sigma : np.ndarray
        Predicted standard deviation, same shape convention as ``real_ir``.
    n_bins : int
        Number of quantile bins requested (the curve may be shorter if bins
        collapse).

    Returns
    -------
    ReliabilityCurve
    """
    error = np.abs(real_ir.squeeze() - mu.squeeze()).ravel().astype(np.float64)
    scale = sigma.squeeze().ravel().astype(np.float64)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(scale, quantiles))
    bin_index = np.digitize(scale, edges)

    predicted, observed, counts = [], [], []
    for b in range(len(edges) + 1):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            continue
        predicted.append(math.sqrt(float(np.mean(scale[mask] ** 2))))
        observed.append(math.sqrt(float(np.mean(error[mask] ** 2))))
        counts.append(count)

    return ReliabilityCurve(
        predicted_sigma=np.asarray(predicted, dtype=np.float64),
        observed_error=np.asarray(observed, dtype=np.float64),
        counts=np.asarray(counts, dtype=np.int64),
    )


def sharpness(sigma: np.ndarray) -> float:
    """Return the mean predicted ``sigma`` — lower is sharper.

    Only meaningful next to a calibration score: sharpness alone rewards an
    overconfident model.

    Parameters
    ----------
    sigma : np.ndarray
        Predicted standard deviation map.

    Returns
    -------
    float
    """
    return float(np.mean(sigma))


def dispersion(sigma: np.ndarray) -> float:
    """Return the coefficient of variation of ``sigma`` (``std / mean``).

    How *heteroscedastic* the head actually is. At ``0.0`` the model predicts a
    constant uncertainty, the learned z-score degenerates into a rescaled raw
    delta, and the whole ``(mu, log_var)`` head has bought nothing.

    Parameters
    ----------
    sigma : np.ndarray
        Predicted standard deviation map.

    Returns
    -------
    float
    """
    mean = float(np.mean(sigma))
    return float(np.std(sigma) / (mean + _EPS))


def mean_gaussian_nll(real_ir: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Return the mean per-pixel Gaussian negative log-likelihood, in nats.

    ``0.5 * log(2 * pi * sigma^2) + (real - mu)^2 / (2 * sigma^2)``: a proper
    scoring rule, minimised only by the true ``(mu, sigma)``, so it cannot be
    gamed by inflating or shrinking ``sigma`` alone. Unlike
    ``scripts.losses.gaussian_nll_loss`` — which drops the additive
    ``0.5 * log(2 * pi)`` constant, irrelevant to gradients — this keeps it, so
    the value is a real likelihood comparable across models and runs.

    Parameters
    ----------
    real_ir : np.ndarray
        Ground-truth IR image, shape ``(H, W)`` or ``(H, W, 1)``.
    mu : np.ndarray
        Predicted mean IR, same shape convention as ``real_ir``.
    sigma : np.ndarray
        Predicted standard deviation, same shape convention as ``real_ir``.

    Returns
    -------
    float
        Mean NLL in nats; lower is better.
    """
    scale = sigma.squeeze().astype(np.float64) + _EPS
    residual = real_ir.squeeze().astype(np.float64) - mu.squeeze().astype(np.float64)
    nll = 0.5 * np.log(2 * np.pi * scale**2) + (residual**2) / (2 * scale**2)
    return float(np.mean(nll))


def mean_laplace_nll(real_ir: np.ndarray, mu: np.ndarray, b: np.ndarray) -> float:
    """Return the mean per-pixel Laplace negative log-likelihood, in nats.

    ``log(2 * b) + |real - mu| / b``: the Laplace counterpart of
    :func:`mean_gaussian_nll`, for models trained with
    ``scripts.losses.laplace_nll_loss`` (``fixing.md`` #10). A proper
    scoring rule, minimised only by the true ``(mu, b)``. Unlike
    ``scripts.losses.laplace_nll_loss`` — which drops the additive
    ``log(2)`` constant, irrelevant to gradients — this keeps it, so the
    value is a real likelihood comparable across models and runs.

    Parameters
    ----------
    real_ir : np.ndarray
        Ground-truth IR image, shape ``(H, W)`` or ``(H, W, 1)``.
    mu : np.ndarray
        Predicted mean IR, same shape convention as ``real_ir``.
    b : np.ndarray
        Predicted Laplace scale (``exp(log_b)``, *not* the standard
        deviation — see :func:`laplace_sigma_from_scale` for the
        conversion), same shape convention as ``real_ir``.

    Returns
    -------
    float
        Mean NLL in nats; lower is better.
    """
    scale = b.squeeze().astype(np.float64) + _EPS
    residual = real_ir.squeeze().astype(np.float64) - mu.squeeze().astype(np.float64)
    nll = np.log(2.0 * scale) + np.abs(residual) / scale
    return float(np.mean(nll))


def laplace_sigma_from_scale(b: np.ndarray) -> np.ndarray:
    """Convert a Laplace scale ``b`` to its standard deviation.

    A Laplace(``mu``, ``b``) distribution has variance ``2 * b^2``, so
    ``sigma = b * sqrt(2)`` — *not* ``exp(0.5 * log_b)``, the Gaussian
    formula every notebook previously applied uniformly to every NLL
    architecture's second channel. Use this (not the Gaussian formula) to
    get a true standard deviation from `unet_nll`/`resunet_nll`/
    `attention_unet_nll` predictions before passing it to any function in
    this module that expects ``sigma`` (``fixing.md`` #10).

    Parameters
    ----------
    b : np.ndarray
        Predicted Laplace scale, ``exp(log_b)``.

    Returns
    -------
    np.ndarray
        Standard deviation, same shape as ``b``.
    """
    return b * math.sqrt(2.0)


@dataclass(frozen=True)
class CalibrationResult:
    """Everything :func:`evaluate_calibration` measures about one prediction.

    Attributes
    ----------
    coverage : tuple[Coverage, ...]
        Empirical vs. nominal coverage at each requested ``k``.
    reliability : ReliabilityCurve
        Per-bin predicted vs. observed error (and its ``ence``).
    error_sigma_spearman : float
        Rank correlation between ``|real - mu|`` and ``sigma``.
    sharpness : float
        Mean predicted ``sigma``.
    dispersion : float
        Coefficient of variation of ``sigma``.
    nll : float
        Mean negative log-likelihood, in nats — Gaussian or Laplace
        depending on ``evaluate_calibration``'s ``distribution`` argument.
    z_mean, z_std : float
        Moments of the learned z-score; a calibrated model gives ``0`` and
        ``1``. ``z_std > 1`` is overconfidence, ``< 1`` underconfidence.
    """

    coverage: tuple[Coverage, ...]
    reliability: ReliabilityCurve
    error_sigma_spearman: float
    sharpness: float
    dispersion: float
    nll: float
    z_mean: float
    z_std: float
    zscore: np.ndarray = field(repr=False)

    def summary(self) -> dict[str, float]:
        """Flatten every scalar into one row, for tabulating across models.

        Returns
        -------
        dict[str, float]
            Metric name to value, e.g. ``{"nll": ..., "coverage_1s": ...}``,
            ready to become a ``pandas`` row in the evaluation notebooks.
        """
        row = {
            "nll": self.nll,
            "ence": self.reliability.ence,
            "z_mean": self.z_mean,
            "z_std": self.z_std,
            "error_sigma_spearman": self.error_sigma_spearman,
            "sharpness": self.sharpness,
            "dispersion": self.dispersion,
        }
        for cov in self.coverage:
            row[f"coverage_{cov.k:g}s"] = cov.empirical
            row[f"coverage_error_{cov.k:g}s"] = cov.error
        return row


def evaluate_calibration(
    real_ir: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    coverage_levels: tuple[float, ...] = DEFAULT_COVERAGE_LEVELS,
    n_bins: int = 10,
    max_samples: int = 500_000,
    seed: int = 0,
    distribution: str = "gaussian",
) -> CalibrationResult:
    """Score a heteroscedastic prediction's ``sigma`` on its own.

    The counterpart of ``scripts.delta_analysis.analyze_delta``: that one turns
    a prediction into maps for the eye, this one turns it into numbers for a
    table. Run it on ``mu``/``sigma`` from a direct ``model.predict`` to
    compare NLL variants on the uncertainty signal itself rather than
    through ``mu`` alone.

    Every metric here except ``nll`` treats ``sigma`` as a plain standard
    deviation and is distribution-agnostic; only the NLL term's formula
    differs structurally between Gaussian and Laplace, so ``distribution``
    dispatches only that one term. Pass the correct ``sigma`` for either
    case: the Gaussian formula (``exp(0.5 * log_var)``) for
    `efficientnet_unet_nll` (until Round 2), or
    :func:`laplace_sigma_from_scale` (``b * sqrt(2)``, **not**
    ``exp(0.5 * log_b)``) for `unet_nll`/`resunet_nll`/
    `attention_unet_nll` (``fixing.md`` #10).

    Parameters
    ----------
    real_ir : np.ndarray
        Ground-truth IR image, shape ``(H, W)`` or ``(H, W, 1)``, in ``[0, 1]``.
    mu : np.ndarray
        Predicted mean IR, same shape convention as ``real_ir``.
    sigma : np.ndarray
        Predicted standard deviation, same shape convention as ``real_ir``
        — see the distribution-specific conversion above.
    coverage_levels : tuple[float, ...]
        Interval half-widths, in predicted standard deviations, to score.
    n_bins : int
        Number of quantile bins for the reliability curve.
    max_samples : int
        Subsampling cap for the Spearman correlation.
    seed : int
        Seed for that subsampling.
    distribution : str
        Which NLL formula to use for the ``nll`` field: ``"gaussian"``
        (:func:`mean_gaussian_nll`) or ``"laplace"``
        (:func:`mean_laplace_nll`, converting ``sigma`` back to the
        Laplace scale ``b = sigma / sqrt(2)``).

    Returns
    -------
    CalibrationResult

    Raises
    ------
    ValueError
        If ``distribution`` is not ``"gaussian"`` or ``"laplace"``.
    """
    if distribution == "gaussian":
        nll = mean_gaussian_nll(real_ir, mu, sigma)
    elif distribution == "laplace":
        nll = mean_laplace_nll(real_ir, mu, sigma / math.sqrt(2.0))
    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. Use 'gaussian' or 'laplace'."
        )

    z = learned_zscore(real_ir, mu, sigma)

    return CalibrationResult(
        coverage=tuple(
            coverage_probability(real_ir, mu, sigma, k=k) for k in coverage_levels
        ),
        reliability=sigma_reliability(real_ir, mu, sigma, n_bins=n_bins),
        error_sigma_spearman=error_sigma_correlation(
            real_ir, mu, sigma, max_samples=max_samples, seed=seed
        ),
        sharpness=sharpness(sigma),
        dispersion=dispersion(sigma),
        nll=nll,
        z_mean=float(np.mean(z)),
        z_std=float(np.std(z)),
        zscore=z,
    )
