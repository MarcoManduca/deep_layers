"""Unit tests for scripts.visualization_nll."""

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from scripts.calibration import evaluate_calibration  # noqa: E402
from scripts.contrast import ZScale, ZScaleMode  # noqa: E402
from scripts.delta_analysis import analyze_delta  # noqa: E402
from scripts.visualization_nll import (  # noqa: E402
    plot_calibration,
    plot_delta_comparison,
    plot_predictions_nll,
    plot_signal_comparison,
    plot_signal_gallery,
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
    assert len(fig.axes) == 12  # 6 image panels + 6 colorbars
    plt.close(fig)


def test_plot_signal_comparison_returns_figure_with_four_models() -> None:
    ir_real = _ir()
    signals = {
        "unet_nll": _ir(),
        "resunet_nll": _ir(),
        "attention_unet_nll": _ir(),
        "efficientnet_unet_nll": _ir(),
    }

    fig = plot_signal_comparison(ir_real, signals, title="Raw delta comparison")

    assert isinstance(fig, Figure)
    # 5 panels (real IR + 4 models) wrapped at 3 cols -> 6-slot 2x3 grid,
    # each occupied slot has an image + colorbar (5 x 2 = 10), plus 1
    # turned-off empty axis for the unused 6th grid slot.
    assert len(fig.axes) == 11
    plt.close(fig)


def test_plot_signal_comparison_respects_max_cols() -> None:
    ir_real = _ir()
    signals = {"unet_nll": _ir(), "resunet_nll": _ir()}

    fig = plot_signal_comparison(ir_real, signals, max_cols=3)

    # 3 panels (real IR + 2 models), all fit in a single row of 3 -> no
    # empty axes to turn off.
    assert len(fig.axes) == 6
    plt.close(fig)


def test_plot_zscore_reports_the_contrast_setting_in_the_panel_title() -> None:
    sigma = np.full((32, 32, 1), 0.1, dtype=np.float32)
    scale = ZScale(mode=ZScaleMode.PERCENTILE, percentile=99.0, gamma=0.5)

    fig = plot_zscore(_ir(), _ir(), sigma, z_scale=scale)

    assert "gamma=0.5" in fig.axes[2].get_title()
    plt.close(fig)


def test_plot_delta_comparison_accepts_a_custom_contrast() -> None:
    ir_real, mu = _ir(), _ir()
    sigma = np.full((32, 32, 1), 0.1, dtype=np.float32)
    result = analyze_delta(ir_real.squeeze(), mu.squeeze(), window_size=11, zone_size=8)

    fig = plot_delta_comparison(
        result, ir_real, mu, sigma, z_scale=ZScale(mode=ZScaleMode.PERCENTILE)
    )

    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_calibration_returns_figure_with_three_panels() -> None:
    ir_real, mu = _ir(), _ir()
    sigma = np.full((32, 32, 1), 0.1, dtype=np.float32)
    result = evaluate_calibration(ir_real, mu, sigma, n_bins=4)

    fig = plot_calibration(result, title="unet_nll")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_signal_gallery_returns_figure_with_heterogeneous_signals() -> None:
    ir_real = _ir()
    signals = {
        "raw delta": _ir(),
        "structural delta": _ir(),
        "|z|": _ir() * 20.0,  # a very different, unbounded-looking scale
    }

    fig = plot_signal_gallery(ir_real, signals, title="Signal evolution")

    assert isinstance(fig, Figure)
    # 4 panels (real IR + 3 signals) wrapped at 3 cols -> 6-slot 2x3 grid,
    # each occupied slot has an image + colorbar (4 x 2 = 8), plus 2
    # turned-off empty axes for the unused grid slots.
    assert len(fig.axes) == 10
    plt.close(fig)


def test_plot_signal_gallery_scales_each_panel_independently() -> None:
    ir_real = _ir()
    small = np.full((32, 32), 0.1, dtype=np.float32)
    large = np.full((32, 32), 50.0, dtype=np.float32)

    fig = plot_signal_gallery(ir_real, {"small": small, "large": large})

    # Panel order: real IR, small, large -> images at axes[0], axes[2], axes[4]
    # (odd-indexed axes are colorbars).
    images = [ax.images[0] for ax in fig.axes if ax.images]
    small_img, large_img = images[1], images[2]
    assert small_img.get_clim()[1] < large_img.get_clim()[1]
    plt.close(fig)


def test_plot_signal_gallery_handles_an_all_zero_signal() -> None:
    ir_real = _ir()
    zeros = np.zeros((32, 32), dtype=np.float32)

    fig = plot_signal_gallery(ir_real, {"zeros": zeros})

    assert isinstance(fig, Figure)
    plt.close(fig)
