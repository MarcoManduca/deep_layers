"""Unit tests for scripts.visualization."""

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from scripts.visualization import (  # noqa: E402
    plot_delta,
    plot_predictions,
    plot_sample_pairs,
    plot_training_curves,
)


def _rgb() -> np.ndarray:
    return np.random.rand(16, 16, 3)


def _ir() -> np.ndarray:
    return np.random.rand(16, 16, 1)


def test_plot_sample_pairs_returns_figure() -> None:
    fig = plot_sample_pairs([_rgb(), _rgb()], [_ir(), _ir()], n=2)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_predictions_returns_figure() -> None:
    fig = plot_predictions(_rgb(), _ir(), _ir())
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_delta_returns_figure() -> None:
    fig = plot_delta(_ir(), _ir())
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_training_curves_handles_train_and_val_keys() -> None:
    history = {"loss": [0.5, 0.3], "val_loss": [0.6, 0.4]}
    fig = plot_training_curves(history)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_training_curves_handles_single_metric() -> None:
    fig = plot_training_curves({"loss": [0.5, 0.3]})
    assert isinstance(fig, Figure)
    plt.close(fig)
