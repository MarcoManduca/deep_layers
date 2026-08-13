"""Visualisation helpers for heteroscedastic (mu, log-variance) predictions."""

import math

import matplotlib.pyplot as plt
import numpy as np


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
    vmax: float = 4.0,
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
    vmax : float
        Symmetric color-scale bound for the z-score panel.

    Returns
    -------
    plt.Figure
    """
    real = ir_real.squeeze()
    mu_sq = mu.squeeze()
    sigma_sq = sigma.squeeze()
    raw_delta = np.abs(real - mu_sq)
    z = (real - mu_sq) / (sigma_sq + 1e-8)

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

    im2 = axes[2].imshow(z, cmap="gray", vmin=-vmax, vmax=vmax)
    axes[2].set_title("z-score = (real - mu) / sigma")
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
    z_vmax: float = 4.0,
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
    z_vmax : float
        Symmetric color-scale bound for the learned z-score panel.

    Returns
    -------
    plt.Figure
    """
    real = ir_real.squeeze()
    mu_sq = mu.squeeze()
    sigma_sq = sigma.squeeze()
    raw_delta = np.abs(real - mu_sq)
    z = (real - mu_sq) / (sigma_sq + 1e-8)

    panels = (
        ("Real IR", real, None),
        ("Raw delta\n|real - mu|", raw_delta, None),
        ("Structural delta\n(1 - structure)", result.structural_delta, (0, 1)),
        ("Fixed-window\nnormalized delta", result.normalized_delta, None),
        ("Learned z-score\n(real - mu) / sigma", z, (-z_vmax, z_vmax)),
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
