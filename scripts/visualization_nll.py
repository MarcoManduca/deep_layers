"""Visualisation helpers for heteroscedastic (mu, log-variance) predictions."""

import math

import matplotlib.pyplot as plt
import numpy as np

from scripts.calibration import CalibrationResult, learned_zscore
from scripts.contrast import ZScale


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
        (``scripts.contrast.ZScale``). Defaults to the plain ``|z| <= 4``
        clip; use a percentile mode and/or ``gamma < 1`` to bring out faint
        detail. The chosen setting is written into the panel title, so a
        figure always states the contrast it was rendered at.

    Returns
    -------
    plt.Figure
    """
    real = ir_real.squeeze()
    mu_sq = mu.squeeze()
    sigma_sq = sigma.squeeze()
    raw_delta = np.abs(real - mu_sq)
    scaled = (z_scale or ZScale()).apply(learned_zscore(ir_real, mu, sigma))

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


def plot_delta_comparison(
    result: object,
    ir_real: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    title: str = "Delta comparison",
    z_scale: ZScale | None = None,
) -> plt.Figure:
    """Compare, side by side, every way this pipeline highlights hidden detail.

    A single figure (3x2 grid) with the ground-truth IR plus every "is
    this pixel a genuine hidden mark?" signal produced across the
    pipeline, so they can be read at a glance instead of scattered
    across notebook sections:

    - **real IR** — ground-truth reference the other panels are computed
      against;
    - **raw delta** ``|real - mu|`` — the original baseline; conflates
      genuine hidden detail with substrate/acquisition gray-level shifts;
    - **structural delta** (``1 - local SSIM structure``) — insensitive to
      gray-level shifts, sensitive to genuine structural change
      (``scripts.delta_analysis``, ``note.md`` §1);
    - **fixed-window normalized delta** — per-window z-score delta,
      spatial-only normalization (``scripts.delta_analysis``,
      ``note.md`` §2);
    - **learned z-score** ``(real - mu) / sigma`` — color/context-
      conditioned normalization learned by the heteroscedastic model
      (``code-review.md`` §7.6, ``note.md`` "Update" section);
    - **confidence map** — agreement between the raw and structural delta.

    See ``note.md``'s "fixed-window vs. learned normalization" section for
    what each of these actually normalizes against and where each has a
    blind spot the others cover.

    Parameters
    ----------
    result : scripts.delta_analysis.DeltaAnalysisResult
        Output of ``scripts.delta_analysis.analyze_delta(ir_real, mu, ...)``.
    ir_real : np.ndarray
        Ground-truth IR of shape ``(H, W)`` or ``(H, W, 1)``.
    mu : np.ndarray
        Predicted mean IR of shape ``(H, W)`` or ``(H, W, 1)``.
    sigma : np.ndarray
        Predicted standard deviation of shape ``(H, W)`` or ``(H, W, 1)``.
    title : str
        Figure title.
    z_scale : ZScale or None
        Contrast settings for the learned z-score panel
        (``scripts.contrast.ZScale``). Defaults to the plain ``|z| <= 4``
        clip.

    Returns
    -------
    plt.Figure
    """
    real = ir_real.squeeze()
    mu_sq = mu.squeeze()
    raw_delta = np.abs(real - mu_sq)
    scaled = (z_scale or ZScale()).apply(learned_zscore(ir_real, mu, sigma))

    panels = (
        ("Real IR", real, None),
        ("Raw delta\n|real - mu|", raw_delta, None),
        ("Structural delta\n(1 - structure)", result.structural_delta, (0, 1)),
        ("Fixed-window\nnormalized delta", result.normalized_delta, None),
        (
            f"Learned z-score\n(real - mu) / sigma — {scaled.label}",
            scaled.values,
            (scaled.vmin, scaled.vmax),
        ),
        ("Confidence\n(raw vs structural agreement)", result.confidence_map, (0, 1)),
    )

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
    fig.suptitle(title)
    for ax, (panel_title, data, vrange) in zip(axes.flat, panels):
        vmin, vmax = vrange if vrange is not None else (None, None)
        im = ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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

    The counterpart of ``plot_delta_comparison``: that one fixes the model
    and compares signal *types* side by side; this one fixes the signal
    type and compares *models* side by side, e.g. "raw delta for
    unet_nll vs. resunet_nll vs. attention_unet_nll vs. efficientnet_unet_nll".

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
