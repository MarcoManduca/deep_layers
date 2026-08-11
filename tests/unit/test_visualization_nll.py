"""Unit tests for scripts.visualization_nll."""

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from scripts.delta_analysis import analyze_delta  # noqa: E402
from scripts.visualization_nll import (  # noqa: E402
    plot_delta_comparison,
    plot_predictions_nll,
    plot_zscore,
)


def _rgb() -> np.ndarray:
    return np.random.rand(32, 32, 3)


def _ir() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((32, 32, 1)).astype(np.float32)


def test_plot_predictions_nll_returns_figure() -> None:
    fig = plot_predictions_nll(_rgb(), _ir(), _ir(), _ir())
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_zscore_returns_figure() -> None:
    sigma = np.full((32, 32, 1), 0.1, dtype=np.float32)
    fig = plot_zscore(_ir(), _ir(), sigma)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_delta_comparison_returns_figure() -> None:
    ir_real, mu = _ir(), _ir()
    sigma = np.full((32, 32, 1), 0.1, dtype=np.float32)
    result = analyze_delta(ir_real.squeeze(), mu.squeeze(), window_size=11, zone_size=8)

    fig = plot_delta_comparison(result, ir_real, mu, sigma)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 10  # 5 image panels + 5 colorbars
    plt.close(fig)
