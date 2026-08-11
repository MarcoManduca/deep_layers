"""Unit tests for scripts.dataset."""

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from scripts.dataset import (
    build_dataset,
    extract_artwork_id,
    grouped_train_val_test_split,
    load_image_pairs,
    mockup_aware_train_val_test_split,
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


def test_load_image_pairs_raises_on_size_mismatch(
    image_pairs_dir: tuple[Path, Path],
) -> None:
    # Arrange
    rgb_dir, ir_dir = image_pairs_dir
    mismatched = np.zeros((8, 8), dtype=np.uint8)
    Image.fromarray(mismatched, mode="L").save(ir_dir / "a1_sezione_0.jpg")

    # Act / Assert
    with pytest.raises(ValueError, match="size mismatch"):
        load_image_pairs(ir_dir, rgb_dir)


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


def _fake_pairs(sections_per_group: dict[str, int]) -> list[tuple[Path, Path]]:
    """Build fake ``(rgb_path, ir_path)`` tuples without touching disk.

    ``mockup_aware_train_val_test_split`` only reads ``.stem``, so no real
    files are needed.
    """
    pairs = []
    for group, n_sections in sections_per_group.items():
        for i in range(n_sections):
            stem = f"{group}_sezione_{i}"
            pairs.append((Path(f"{stem}.jpg"), Path(f"{stem}.jpg")))
    return pairs


def test_mockup_aware_split_keeps_mockup_groups_mostly_in_trainval() -> None:
    # Arrange: 3 real artworks (leakage-sensitive) + 1 mockup group (20 sections)
    pairs = _fake_pairs({"a1": 3, "b2": 3, "c3": 2, "tblu": 20})

    # Act
    train, val, test = mockup_aware_train_val_test_split(
        pairs, mockup_ids=["tblu"], mockup_test_ratio=0.05, seed=42
    )
    mockup_test = [p for p in test if extract_artwork_id(p[0].stem) == "tblu"]
    mockup_trainval = [
        p for p in train + val if extract_artwork_id(p[0].stem) == "tblu"
    ]

    # Assert: only a small slice of the mockup group ends up in test, the
    # rest is spread across train/val (not held out entirely, unlike a
    # real-artwork group would be).
    assert len(mockup_test) == 1  # round(20 * 0.05)
    assert len(mockup_trainval) == 19


def test_mockup_aware_split_still_prevents_leakage_for_real_artworks() -> None:
    # Arrange
    pairs = _fake_pairs({"a1": 3, "b2": 2, "c3": 2, "tverde": 20})

    # Act
    train, val, test = mockup_aware_train_val_test_split(
        pairs, mockup_ids=["tverde"], seed=42
    )
    real_train_ids = {
        extract_artwork_id(p[0].stem)
        for p in train
        if extract_artwork_id(p[0].stem) != "tverde"
    }
    real_val_ids = {
        extract_artwork_id(p[0].stem)
        for p in val
        if extract_artwork_id(p[0].stem) != "tverde"
    }
    real_test_ids = {
        extract_artwork_id(p[0].stem)
        for p in test
        if extract_artwork_id(p[0].stem) != "tverde"
    }

    # Assert: real artwork IDs never span more than one fold.
    assert real_train_ids.isdisjoint(real_val_ids)
    assert real_train_ids.isdisjoint(real_test_ids)
    assert real_val_ids.isdisjoint(real_test_ids)


def test_mockup_aware_split_covers_every_pair_exactly_once() -> None:
    # Arrange
    pairs = _fake_pairs({"a1": 3, "b2": 2, "c3": 2, "tblu": 20, "tverde": 15})

    # Act
    train, val, test = mockup_aware_train_val_test_split(
        pairs, mockup_ids=["tblu", "tverde"], seed=42
    )

    # Assert
    assert len(train) + len(val) + len(test) == len(pairs)
    assert set(train + val + test) == set(pairs)


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
