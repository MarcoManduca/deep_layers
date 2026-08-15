"""Unit tests for scripts.detection."""

import numpy as np
import pytest

from scripts.detection import evaluate_detection, rank_signals

_SIZE = 64


def _mask(positive_rows: slice = slice(0, 8)) -> np.ndarray:
    mask = np.zeros((_SIZE, _SIZE), dtype=bool)
    mask[positive_rows, :] = True
    return mask


def _perfect_signal(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.float64)


# --- scoring -----------------------------------------------------------------


def test_perfect_signal_scores_one() -> None:
    mask = _mask()

    result = evaluate_detection(_perfect_signal(mask), mask)

    assert result.auroc == pytest.approx(1.0)
    assert result.average_precision == pytest.approx(1.0)


def test_inverted_signal_scores_zero_auroc() -> None:
    mask = _mask()

    result = evaluate_detection(1.0 - _perfect_signal(mask), mask)

    assert result.auroc == pytest.approx(0.0)


def test_uninformative_signal_scores_chance() -> None:
    mask = _mask()
    rng = np.random.default_rng(0)

    result = evaluate_detection(rng.random((_SIZE, _SIZE)), mask)

    assert result.auroc == pytest.approx(0.5, abs=0.05)
    assert result.average_precision == pytest.approx(result.prevalence, abs=0.05)


def test_prevalence_matches_the_mask_density() -> None:
    mask = _mask(slice(0, 16))

    result = evaluate_detection(_perfect_signal(mask), mask)

    assert result.prevalence == pytest.approx(0.25)


def test_lift_is_one_for_an_uninformative_signal() -> None:
    mask = _mask()
    rng = np.random.default_rng(0)

    result = evaluate_detection(rng.random((_SIZE, _SIZE)), mask)

    assert result.lift == pytest.approx(1.0, abs=0.4)


def test_lift_exceeds_one_for_an_informative_signal() -> None:
    mask = _mask()

    assert evaluate_detection(_perfect_signal(mask), mask).lift > 5.0


# --- input handling ----------------------------------------------------------


def test_accepts_a_trailing_channel_axis() -> None:
    mask = _mask()
    signal = _perfect_signal(mask)[..., np.newaxis]

    assert evaluate_detection(signal, mask).auroc == pytest.approx(1.0)


def test_subsampling_is_reproducible() -> None:
    mask = _mask()
    rng = np.random.default_rng(0)
    signal = rng.random((_SIZE, _SIZE))

    first = evaluate_detection(signal, mask, max_samples=512, seed=7)
    second = evaluate_detection(signal, mask, max_samples=512, seed=7)

    assert first.auroc == second.auroc


def test_raises_when_shapes_differ() -> None:
    with pytest.raises(ValueError, match="same shape"):
        evaluate_detection(np.zeros((_SIZE, _SIZE)), _mask()[:8])


@pytest.mark.parametrize("fill", [False, True])
def test_raises_on_a_single_class_mask(fill: bool) -> None:
    mask = np.full((_SIZE, _SIZE), fill, dtype=bool)

    with pytest.raises(ValueError, match="both positive and negative"):
        evaluate_detection(np.zeros((_SIZE, _SIZE)), mask)


# --- ranking -----------------------------------------------------------------


def test_rank_signals_scores_every_candidate() -> None:
    mask = _mask()
    rng = np.random.default_rng(0)
    signals = {
        "noise": rng.random((_SIZE, _SIZE)),
        "perfect": _perfect_signal(mask),
    }

    results = rank_signals(signals, mask)

    assert set(results) == {"noise", "perfect"}


def test_rank_signals_orders_by_auroc_descending() -> None:
    mask = _mask()
    rng = np.random.default_rng(0)
    signals = {
        "noise": rng.random((_SIZE, _SIZE)),
        "perfect": _perfect_signal(mask),
        "inverted": 1.0 - _perfect_signal(mask),
    }

    results = rank_signals(signals, mask)

    assert list(results) == ["perfect", "noise", "inverted"]
