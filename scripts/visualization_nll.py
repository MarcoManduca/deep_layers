"""Visualisation helpers for heteroscedastic (mu, log-variance) predictions."""

import math

import matplotlib.pyplot as plt
import numpy as np

from scripts.calibration import CalibrationResult, learned_zscore
from scripts.contrast import ZScale, ZScaleMode

# Chosen 2026-08-20 after visually comparing fixed |z| <= 4, p99.5, and
# p99.5 + gamma 0.5 across all data/test/ ground-truth images
# (062_model_comparison_v2.ipynb §2c): the plain percentile clip reads
# slightly better than the fixed baseline, while adding gamma compression
# hurt readability rather than helping, so gamma is left at 1.0.
DEFAULT_Z_SCALE = ZScale(mode=ZScaleMode.PERCENTILE, percentile=99.5)


def plot_predictions_nll(
    rgb: np.ndarray,
    ir_real: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    title: str = "",
) -> plt.Figure:
    """Display RGB | real IR | predicted mu | predicted sigma in one row.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image of shape ``(H, W, 3)``.
    ir_real : np.ndarray
        Ground-truth IR of shape ``(H, W)`` or ``(H, W, 1)``.
    mu : np.ndarray
        Predicted mean IR of shape ``(H, W)`` or ``(H, W, 1)``.
    sigma : np.ndarray
        Predicted standard deviation of shape ``(H, W)`` or ``(H, W, 1)``.
    title : str
        Figure title.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    axes[1].imshow(ir_real.squeeze(), cmap="gray")
    axes[1].set_title("Real IR")
    axes[1].axis("off")
    axes[2].imshow(mu.squeeze(), cmap="gray")
    axes[2].set_title("Predicted mu")
    axes[2].axis("off")
    im = axes[3].imshow(sigma.squeeze(), cmap="gray")
    axes[3].set_title("Predicted sigma")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_zscore(
    ir_real: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    title: str = "Learned z-score",
    z_scale: ZScale | None = None,
) -> plt.Figure:
    """Plot the raw delta, the learned sigma, and the resulting anomaly z-score.

    ``z = (real_IR - mu) / sigma`` is the learned counterpart of the fixed
    Gaussian-window normalisation in ``scripts.delta_analysis``: a pixel
    with a large raw delta but high learned ``sigma`` (a color/context the
    model has seen vary a lot in training) yields a small, unremarkable
    ``z``, while a pixel with a modest raw delta but low ``sigma`` yields a
    large ``z`` — a genuine anomaly candidate.

    Parameters
    ----------
    ir_real : np.ndarray
        Ground-truth IR of shape ``(H, W)`` or ``(H, W, 1)``.
    mu : np.ndarray
        Predicted mean IR of shape ``(H, W)`` or ``(H, W, 1)``.
    sigma : np.ndarray
        Predicted standard deviation of shape ``(H, W)`` or ``(H, W, 1)``.
    title : str
        Figure title.
    z_scale : ZScale or None
        Contrast settings for the z-score panel
        (``scripts.contrast.ZScale``). Defaults to ``DEFAULT_Z_SCALE``
        (percentile ``p99.5``, no gamma — chosen over the fixed ``|z| <= 4``
        clip and over adding gamma compression, see module docstring).
        The chosen setting is written into the panel title, so a figure
        always states the contrast it was rendered at.

    Returns
    -------
    plt.Figure
    """
    real = ir_real.squeeze()
    mu_sq = mu.squeeze()
    sigma_sq = sigma.squeeze()
    raw_delta = np.abs(real - mu_sq)
    scaled = (z_scale or DEFAULT_Z_SCALE).apply(learned_zscore(ir_real, mu, sigma))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    im0 = axes[0].imshow(raw_delta, cmap="gray")
    axes[0].set_title("Raw delta |real - mu|")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(sigma_sq, cmap="gray")
    axes[1].set_title("Learned sigma")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(scaled.values, cmap="gray", vmin=scaled.vmin, vmax=scaled.vmax)
    axes[2].set_title(f"z-score = (real - mu) / sigma\n{scaled.label}")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def plot_signal_comparison(
    ir_real: np.ndarray,
    signals: dict[str, np.ndarray],
    title: str = "",
    vrange: tuple[float, float] | None = None,
    max_cols: int = 3,
) -> plt.Figure:
    """Compare one delta signal (raw, structural, ...) across several models.

    Fixes the signal type and compares *models* side by side, e.g. "raw
    delta for unet_nll vs. resunet_nll vs. attention_unet_nll vs.
    efficientnet_unet_nll".

    Always shows the ground-truth real IR as the first panel so every
    model's signal can be read against the actual target, then one panel
    per entry in ``signals``, wrapped at ``max_cols`` columns per row.

    For z-score signals, ``scripts.contrast.ZScale.apply_many`` produces the
    ``signals`` and ``vrange`` arguments together, against a single shared
    limit.

    Parameters
    ----------
    ir_real : np.ndarray
        Ground-truth IR of shape ``(H, W)`` or ``(H, W, 1)``.
    signals : dict[str, np.ndarray]
        Maps a model/architecture name to its 2D signal map (same shape
        as ``ir_real``) for this one delta type.
    title : str
        Figure title.
    vrange : tuple[float, float] | None
        Shared ``(vmin, vmax)`` color-scale bound applied to every model
        panel (not the real IR panel, which is always ``(0, 1)``) so the
        models are visually comparable. If ``None``, uses the shared
        min/max across all ``signals`` values.
    max_cols : int
        Maximum panels per row before wrapping to a new row.

    Returns
    -------
    plt.Figure
    """
    real = ir_real.squeeze()
    arrays = {name: data.squeeze() for name, data in signals.items()}

    if vrange is None:
        vmin = min(a.min() for a in arrays.values())
        vmax = max(a.max() for a in arrays.values())
    else:
        vmin, vmax = vrange

    panels = [("Real IR", real, (0.0, 1.0))]
    panels += [(name, data, (vmin, vmax)) for name, data in arrays.items()]

    n = len(panels)
    n_cols = min(max_cols, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
    fig.suptitle(title)
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, (panel_title, data, (pmin, pmax)) in zip(axes_flat, panels):
        im = ax.imshow(data, cmap="gray", vmin=pmin, vmax=pmax)
        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_signal_gallery(
    ir_real: np.ndarray,
    signals: dict[str, np.ndarray],
    title: str = "",
    max_cols: int = 3,
    percentile: float = 99.0,
) -> plt.Figure:
    """Compare several *different* signal types side by side, each on its own scale.

    ``plot_signal_comparison`` is built for comparing the *same* signal type
    across models on one shared scale, so brightness is genuinely
    comparable — correct for that job, wrong for this one: a "before/after"
    panel mixing a bounded delta (``[0, ~1]``) with an unbounded z-score
    (single-pixel outliers into the tens) on one shared scale would flatten
    the delta panels to near-black. Each panel here is scaled independently
    to its own ``[0, percentile]`` range instead, so every panel stays
    readable, at the cost of losing cross-panel brightness comparability —
    read each panel's own bracketed limit in its title, not its brightness
    relative to its neighbours.

    Parameters
    ----------
    ir_real : np.ndarray
        Ground-truth IR of shape ``(H, W)`` or ``(H, W, 1)``.
    signals : dict[str, np.ndarray]
        Maps a panel label to its 2D signal map (same shape as ``ir_real``).
    title : str
        Figure title.
    max_cols : int
        Maximum panels per row before wrapping to a new row.
    percentile : float
        Upper percentile of ``|signal|`` used as each panel's own display
        limit, so a handful of outlier pixels don't flatten the rest.

    Returns
    -------
    plt.Figure
    """
    real = ir_real.squeeze()
    arrays = {name: data.squeeze() for name, data in signals.items()}

    panels = [("Real IR", real, (0.0, 1.0))]
    for name, data in arrays.items():
        vmax = float(np.percentile(np.abs(data), percentile))
        if vmax <= 1e-8:
            vmax = float(np.abs(data).max()) + 1e-8
        panels.append((name, data, (0.0, vmax)))

    n = len(panels)
    n_cols = min(max_cols, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
    fig.suptitle(title)
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, (panel_title, data, (pmin, pmax)) in zip(axes_flat, panels):
        im = ax.imshow(data, cmap="gray", vmin=pmin, vmax=pmax)
        ax.set_title(f"{panel_title}\n[0, {pmax:.2f}]", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_calibration(
    result: CalibrationResult,
    title: str = "Sigma calibration",
    z_range: float = 5.0,
    bins: int = 120,
) -> plt.Figure:
    """Display whether the predicted ``sigma`` can be trusted, in three panels.

    The visual counterpart of ``scripts.calibration.evaluate_calibration``:

    - **reliability diagram** — per-bin predicted ``sigma`` against the error
      actually observed in that bin. On the dashed identity line the model is
      calibrated; below it, it is overconfident (real errors exceed the
      uncertainty it claims), above it, underconfident;
    - **coverage** — empirical vs. nominal fraction of pixels inside
      ``+/- k * sigma``, the same reading in interval form;
    - **z-score distribution** — the learned z-score against the standard
      normal it should follow if ``sigma`` were correct. A distribution
      broader than the reference means the z-score flags anomalies everywhere
      and cannot be thresholded meaningfully.

    Parameters
    ----------
    result : scripts.calibration.CalibrationResult
        Output of ``scripts.calibration.evaluate_calibration``.
    title : str
        Figure title.
    z_range : float
        Half-width of the z-score histogram's x-axis.
    bins : int
        Number of histogram bins for the z-score panel.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(title)

    reliability = result.reliability
    upper = float(
        max(reliability.predicted_sigma.max(), reliability.observed_error.max())
    )
    axes[0].plot([0, upper], [0, upper], "k--", lw=1, label="perfect calibration")
    axes[0].plot(
        reliability.predicted_sigma, reliability.observed_error, "o-", label="observed"
    )
    axes[0].set_xlabel("predicted sigma (per bin)")
    axes[0].set_ylabel("observed RMSE (per bin)")
    axes[0].set_title(f"Reliability — ENCE = {reliability.ence:.3f}")
    axes[0].legend(fontsize=8)

    positions = np.arange(len(result.coverage))
    width = 0.38
    axes[1].bar(
        positions - width / 2,
        [c.empirical for c in result.coverage],
        width,
        label="empirical",
    )
    axes[1].bar(
        positions + width / 2,
        [c.nominal for c in result.coverage],
        width,
        label="nominal",
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"{c.k:g} sigma" for c in result.coverage])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("fraction of pixels covered")
    axes[1].set_title("Coverage")
    axes[1].legend(fontsize=8)

    z = np.clip(result.zscore.ravel(), -z_range, z_range)
    axes[2].hist(z, bins=bins, density=True, alpha=0.75, label="z-score")
    grid = np.linspace(-z_range, z_range, 200)
    normal = np.exp(-0.5 * grid**2) / math.sqrt(2 * math.pi)
    axes[2].plot(grid, normal, "k--", lw=1, label="N(0, 1)")
    axes[2].set_xlabel("z = (real - mu) / sigma")
    axes[2].set_title(
        f"z distribution — mean {result.z_mean:.2f}, std {result.z_std:.2f}"
    )
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    return fig
