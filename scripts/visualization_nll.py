"""Visualisation helpers for heteroscedastic (mu, log-variance) predictions."""

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
