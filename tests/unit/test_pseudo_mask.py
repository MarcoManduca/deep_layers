"""Unit tests for scripts.pseudo_mask."""

import numpy as np
import pytest

from scripts.pseudo_mask import cross_modal_pseudo_mask, to_grayscale

_SIZE = 128
_STROKE = slice(60, 64)


def _texture() -> np.ndarray:
    """Smooth, non-degenerate grayscale texture in ``[0, 1]``."""
    x, y = np.meshgrid(
        np.linspace(0, 6 * np.pi, _SIZE), np.linspace(0, 6 * np.pi, _SIZE)
    )
    return (0.5 + 0.25 * np.sin(x) * np.cos(y)).astype(np.float32)


def _rgb_from(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray] * 3, axis=-1)


# --- grayscale conversion ----------------------------------------------------


def test_to_grayscale_applies_luma_weights() -> None:
    red = np.zeros((2, 2, 3), dtype=np.float32)
    red[..., 0] = 1.0

    assert to_grayscale(red) == pytest.approx(0.299, abs=1e-6)


def test_to_grayscale_drops_the_channel_axis() -> None:
    assert to_grayscale(np.zeros((8, 5, 3), dtype=np.float32)).shape == (8, 5)


# --- what the score should ignore --------------------------------------------


def test_score_is_low_when_the_ir_matches_the_rgb() -> None:
    gray = _texture()

    result = cross_modal_pseudo_mask(_rgb_from(gray), gray)

    assert result.score.mean() < 0.05


def test_score_is_low_when_the_ir_is_anticorrelated_with_the_rgb() -> None:
    gray = _texture()

    result = cross_modal_pseudo_mask(_rgb_from(gray), 1.0 - gray)

    # An inverted IR is fully explained by the RGB — just with opposite sign.
    # Using the signed structure term instead of its magnitude would flag this
    # whole image as hidden detail.
    assert result.score.mean() < 0.05


def test_score_is_low_where_the_ir_carries_no_structure() -> None:
    gray = _texture()
    flat_ir = np.full((_SIZE, _SIZE), 0.5, dtype=np.float32)

    result = cross_modal_pseudo_mask(_rgb_from(gray), flat_ir)

    assert result.score.max() < 0.05


# --- what the score should find ----------------------------------------------


def test_score_is_high_where_the_ir_holds_structure_absent_from_the_rgb() -> None:
    gray = _texture()
    ir = gray.copy()
    ir[_STROKE, :] = 0.0  # a stroke visible only in the IR

    result = cross_modal_pseudo_mask(_rgb_from(gray), ir)

    off_stroke = np.delete(result.score, np.s_[_STROKE], axis=0)
    assert result.score[_STROKE, :].mean() > 10 * off_stroke.mean()


def test_score_stays_in_the_unit_range() -> None:
    gray = _texture()
    ir = gray.copy()
    ir[_STROKE, :] = 0.0

    result = cross_modal_pseudo_mask(_rgb_from(gray), ir)

    assert result.score.min() >= 0.0
    assert result.score.max() <= 1.0


def test_accepts_a_trailing_channel_axis_on_the_ir() -> None:
    gray = _texture()

    result = cross_modal_pseudo_mask(_rgb_from(gray), gray[..., np.newaxis])

    assert result.score.shape == (_SIZE, _SIZE)


# --- binarisation ------------------------------------------------------------


def test_binarize_selects_the_requested_top_fraction() -> None:
    gray = _texture()
    ir = gray.copy()
    ir[_STROKE, :] = 0.0

    mask = cross_modal_pseudo_mask(_rgb_from(gray), ir).binarize(percentile=95.0)

    assert mask.mean() == pytest.approx(0.05, abs=0.01)


def test_binarize_is_strongly_enriched_on_the_hidden_stroke() -> None:
    gray = _texture()
    ir = gray.copy()
    ir[_STROKE, :] = 0.0

    mask = cross_modal_pseudo_mask(_rgb_from(gray), ir).binarize(percentile=95.0)

    # Not "every stroke pixel is marked": the 11x11 window spreads the score
    # over a band wider than the stroke, so the top 5% is split across it.
    # Enrichment relative to the rest of the image is the property that matters.
    off_stroke = np.delete(mask, np.s_[_STROKE], axis=0)
    assert mask[_STROKE, :].mean() > 10 * off_stroke.mean()


@pytest.mark.parametrize("percentile", [-1.0, 100.0, 150.0])
def test_binarize_rejects_an_out_of_range_percentile(percentile: float) -> None:
    gray = _texture()
    result = cross_modal_pseudo_mask(_rgb_from(gray), gray)

    with pytest.raises(ValueError, match="percentile"):
        result.binarize(percentile=percentile)


def test_raises_when_rgb_and_ir_shapes_differ() -> None:
    rgb = np.zeros((_SIZE, _SIZE, 3), dtype=np.float32)
    ir = np.zeros((_SIZE, _SIZE // 2), dtype=np.float32)

    with pytest.raises(ValueError, match="same spatial shape"):
        cross_modal_pseudo_mask(rgb, ir)
