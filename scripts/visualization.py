"""Plotting utilities for the deep-layers pipeline."""

import matplotlib.pyplot as plt
import numpy as np


def plot_sample_pairs(
    rgb_images: list[np.ndarray],
    ir_images: list[np.ndarray],
    n: int = 5,
    title: str = "RGB / IR sample pairs",
) -> plt.Figure:
    """Display n RGB / IR image pairs in a two-row grid.

    Parameters
    ----------
    rgb_images : list[np.ndarray]
        RGB images of shape ``(H, W, 3)``, values in ``[0, 1]``.
    ir_images : list[np.ndarray]
        IR images of shape ``(H, W)`` or ``(H, W, 1)``, values in ``[0, 1]``.
    n : int
        Number of pairs to display.
    title : str
        Figure title.

    Returns
    -------
    plt.Figure
    """
    n = min(n, len(rgb_images))
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    fig.suptitle(title)
    for i in range(n):
        axes[0, i].imshow(rgb_images[i])
        axes[0, i].set_title("RGB")
        axes[0, i].axis("off")
        axes[1, i].imshow(ir_images[i].squeeze(), cmap="gray")
        axes[1, i].set_title("IR")
        axes[1, i].axis("off")
    plt.tight_layout()
    return fig


def plot_predictions(
    rgb: np.ndarray,
    ir_real: np.ndarray,
    ir_pred: np.ndarray,
    title: str = "",
) -> plt.Figure:
    """Display RGB | real IR | predicted IR in a three-column layout.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image of shape ``(H, W, 3)``.
    ir_real : np.ndarray
        Ground-truth IR of shape ``(H, W)`` or ``(H, W, 1)``.
    ir_pred : np.ndarray
        Predicted IR of shape ``(H, W)`` or ``(H, W, 1)``.
    title : str
        Figure title.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    axes[1].imshow(ir_real.squeeze(), cmap="gray")
    axes[1].set_title("Real IR")
    axes[1].axis("off")
    axes[2].imshow(ir_pred.squeeze(), cmap="gray")
    axes[2].set_title("Predicted IR")
    axes[2].axis("off")
    plt.tight_layout()
    return fig


def plot_delta(
    ir_real: np.ndarray,
    ir_pred: np.ndarray,
    cmap: str = "hot",
    title: str = "IR Delta (|real − predicted|)",
) -> plt.Figure:
    """Plot the absolute pixel-wise delta between real and predicted IR.

    The delta highlights regions where the model's prediction deviates
    from the ground truth — candidate locations for underdrawings.

    Parameters
    ----------
    ir_real : np.ndarray
        Ground-truth IR of shape ``(H, W)`` or ``(H, W, 1)``.
    ir_pred : np.ndarray
        Predicted IR of shape ``(H, W)`` or ``(H, W, 1)``.
    cmap : str
        Matplotlib colormap for the heatmap.
    title : str
        Figure title.

    Returns
    -------
    plt.Figure
    """
    delta = np.abs(ir_real.squeeze() - ir_pred.squeeze())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)
    axes[0].imshow(ir_real.squeeze(), cmap="gray")
    axes[0].set_title("Real IR")
    axes[0].axis("off")
    axes[1].imshow(ir_pred.squeeze(), cmap="gray")
    axes[1].set_title("Predicted IR")
    axes[1].axis("off")
    im = axes[2].imshow(delta, cmap=cmap)
    axes[2].set_title("Delta")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_training_curves(
    history: dict,
    title: str = "Training history",
) -> plt.Figure:
    """Plot training and validation loss / metric curves.

    Parameters
    ----------
    history : dict
        ``History.history`` dict from ``model.fit()``.
    title : str
        Figure title.

    Returns
    -------
    plt.Figure
    """
    train_keys = [k for k in history if not k.startswith("val_")]
    n = len(train_keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(title)
    for ax, key in zip(axes, train_keys):
        ax.plot(history[key], label="train")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], label="val")
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    return fig
