"""Unit tests for scripts.dataset."""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from scripts.dataset import (
    build_dataset,
    extract_artwork_id,
    grouped_train_val_test_split,
    load_image_pairs,
    pad_to_multiple,
)


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("a1_sezione_0", "a1"),
        ("natmorta1_sezione_12", "natmorta1"),
        ("oblato_volto_sezione_3", "oblato_volto"),
        ("sch01_sezione_24", "sch01"),
    ],
)
def test_extract_artwork_id_returns_stem_before_sezione(
    filename: str, expected: str
) -> None:
    assert extract_artwork_id(filename) == expected


def test_load_image_pairs_returns_all_matching_pairs(
    image_pairs_dir: tuple[Path, Path], n_pairs: int
) -> None:
    # Arrange
    rgb_dir, ir_dir = image_pairs_dir

    # Act
    pairs = load_image_pairs(ir_dir, rgb_dir)

    # Assert
    assert len(pairs) == n_pairs
    assert all(rgb.stem == ir.stem for rgb, ir in pairs)


def test_load_image_pairs_raises_when_no_common_stems(tmp_path: Path) -> None:
    # Arrange
    rgb_dir = tmp_path / "rgb"
    ir_dir = tmp_path / "ir"
    rgb_dir.mkdir()
    ir_dir.mkdir()

    # Act / Assert
    with pytest.raises(ValueError, match="No matching pairs"):
        load_image_pairs(ir_dir, rgb_dir)


def test_grouped_split_has_no_artwork_leakage(
    image_pairs_dir: tuple[Path, Path],
) -> None:
    # Arrange
    rgb_dir, ir_dir = image_pairs_dir
    pairs = load_image_pairs(ir_dir, rgb_dir)

    # Act
    train, val, test = grouped_train_val_test_split(pairs, seed=42)
    train_ids = {extract_artwork_id(p[0].stem) for p in train}
    val_ids = {extract_artwork_id(p[0].stem) for p in val}
    test_ids = {extract_artwork_id(p[0].stem) for p in test}

    # Assert
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_grouped_split_covers_every_pair_exactly_once(
    image_pairs_dir: tuple[Path, Path], n_pairs: int
) -> None:
    # Arrange
    rgb_dir, ir_dir = image_pairs_dir
    pairs = load_image_pairs(ir_dir, rgb_dir)

    # Act
    train, val, test = grouped_train_val_test_split(pairs, seed=42)

    # Assert
    assert len(train) + len(val) + len(test) == n_pairs
    assert len(set(train + val + test)) == n_pairs


def test_grouped_split_is_deterministic_for_same_seed(
    image_pairs_dir: tuple[Path, Path],
) -> None:
    # Arrange
    rgb_dir, ir_dir = image_pairs_dir
    pairs = load_image_pairs(ir_dir, rgb_dir)

    # Act
    first = grouped_train_val_test_split(pairs, seed=42)
    second = grouped_train_val_test_split(pairs, seed=42)

    # Assert
    assert first == second


@pytest.mark.parametrize(
    "height, width, expected_h, expected_w",
    [
        (30, 30, 32, 32),
        (16, 16, 16, 16),
        (17, 33, 32, 48),
    ],
)
def test_pad_to_multiple_rounds_up_to_divisor(
    height: int, width: int, expected_h: int, expected_w: int
) -> None:
    # Arrange
    image = tf.zeros((height, width, 3))

    # Act
    padded, (orig_h, orig_w) = pad_to_multiple(image, multiple=16)

    # Assert
    assert padded.shape[0] == expected_h
    assert padded.shape[1] == expected_w
    assert int(orig_h) == height
    assert int(orig_w) == width


def test_build_dataset_yields_expected_shapes(
    image_pairs_dir: tuple[Path, Path],
) -> None:
    # Arrange
    rgb_dir, ir_dir = image_pairs_dir
    pairs = load_image_pairs(ir_dir, rgb_dir)

    # Act
    rgb_batch, ir_batch = next(iter(build_dataset(pairs, batch_size=2)))

    # Assert
    assert rgb_batch.shape == (2, 16, 16, 3)
    assert ir_batch.shape == (2, 16, 16, 1)


def test_build_dataset_crops_batches_to_crop_size(
    image_pairs_dir: tuple[Path, Path],
) -> None:
    # Arrange
    rgb_dir, ir_dir = image_pairs_dir
    pairs = load_image_pairs(ir_dir, rgb_dir)

    # Act
    rgb_batch, ir_batch = next(
        iter(build_dataset(pairs, batch_size=2, augment=True, crop_size=16))
    )

    # Assert
    assert rgb_batch.shape == (2, 16, 16, 3)
    assert ir_batch.shape == (2, 16, 16, 1)


@pytest.mark.parametrize("bad_crop", [15, 0, -16, 100])
def test_build_dataset_rejects_invalid_crop_size(
    image_pairs_dir: tuple[Path, Path], bad_crop: int
) -> None:
    rgb_dir, ir_dir = image_pairs_dir
    pairs = load_image_pairs(ir_dir, rgb_dir)
    with pytest.raises(ValueError, match="multiple of 16"):
        build_dataset(pairs, augment=True, crop_size=bad_crop)


def test_build_dataset_augmented_stream_is_reproducible(
    image_pairs_dir: tuple[Path, Path],
) -> None:
    # Arrange
    rgb_dir, ir_dir = image_pairs_dir
    pairs = load_image_pairs(ir_dir, rgb_dir)

    # Act
    first = next(
        iter(build_dataset(pairs, batch_size=2, augment=True, shuffle=True, seed=7))
    )[0].numpy()
    second = next(
        iter(build_dataset(pairs, batch_size=2, augment=True, shuffle=True, seed=7))
    )[0].numpy()

    # Assert
    assert np.allclose(first, second)
