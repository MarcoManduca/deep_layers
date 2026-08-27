"""Unit tests for scripts.kfold's grouped k-fold splits."""

from pathlib import Path

import pytest

from scripts.kfold import (
    fold_artwork_groups,
    fold_split,
    grouped_kfold_splits,
)

MOCKUPS = ["tblu", "trosso"]


def _pairs(spec: dict[str, int]) -> list[tuple[Path, Path]]:
    """``{"a1": 3, ...}`` -> 3 (rgb, ir) pairs named ``a1_sezione_{0,1,2}``."""
    out = []
    for artwork, n in spec.items():
        for i in range(n):
            stem = f"{artwork}_sezione_{i}"
            out.append((Path(f"rgb/{stem}.jpg"), Path(f"ir/{stem}.jpg")))
    return out


PAIRS = _pairs(
    {"a1": 3, "a2": 2, "a3": 4, "a4": 1, "a5": 2, "a6": 3, "tblu": 5, "trosso": 4}
)


def test_folds_partition_the_real_artworks_exactly_once() -> None:
    groups = fold_artwork_groups(PAIRS, k=3, mockup_ids=MOCKUPS, seed=42)

    flat = [g for fold in groups for g in fold]
    assert sorted(flat) == ["a1", "a2", "a3", "a4", "a5", "a6"]
    assert len(groups) == 3


def test_mockups_never_appear_in_a_held_out_fold() -> None:
    groups = fold_artwork_groups(PAIRS, k=3, mockup_ids=MOCKUPS, seed=42)
    assert all("tblu" not in f and "trosso" not in f for f in groups)


def test_fold_assignment_is_deterministic_in_seed() -> None:
    a = fold_artwork_groups(PAIRS, k=3, mockup_ids=MOCKUPS, seed=42)
    b = fold_artwork_groups(PAIRS, k=3, mockup_ids=MOCKUPS, seed=42)
    c = fold_artwork_groups(PAIRS, k=3, mockup_ids=MOCKUPS, seed=7)
    assert a == b
    assert a != c


def test_split_has_no_artwork_leakage_between_train_and_val() -> None:
    for train, val in grouped_kfold_splits(PAIRS, k=3, mockup_ids=MOCKUPS, seed=42):
        train_arts = {p[0].stem.rsplit("_sezione_", 1)[0] for p in train}
        val_arts = {p[0].stem.rsplit("_sezione_", 1)[0] for p in val}
        assert train_arts.isdisjoint(val_arts)


def test_all_mockup_pairs_are_in_every_fold_train_set() -> None:
    mockup_pairs = {
        p for p in PAIRS if p[0].stem.rsplit("_sezione_", 1)[0] in MOCKUPS
    }
    for train, _ in grouped_kfold_splits(PAIRS, k=3, mockup_ids=MOCKUPS, seed=42):
        assert mockup_pairs.issubset(set(train))


def test_every_real_pair_is_held_out_exactly_once_across_folds() -> None:
    real_pairs = [
        p for p in PAIRS if p[0].stem.rsplit("_sezione_", 1)[0] not in MOCKUPS
    ]
    seen: list[tuple[Path, Path]] = []
    for _, val in grouped_kfold_splits(PAIRS, k=3, mockup_ids=MOCKUPS, seed=42):
        seen += val
    assert sorted(map(str, seen)) == sorted(str(p) for p in real_pairs)


def test_fold_split_matches_the_indexed_full_split() -> None:
    full = grouped_kfold_splits(PAIRS, k=3, mockup_ids=MOCKUPS, seed=42)
    for i in range(3):
        assert fold_split(PAIRS, k=3, fold=i, mockup_ids=MOCKUPS, seed=42) == full[i]


def test_raises_when_fewer_groups_than_folds() -> None:
    tiny = _pairs({"a1": 2, "a2": 2})
    with pytest.raises(ValueError, match="fewer than k"):
        fold_artwork_groups(tiny, k=3, mockup_ids=MOCKUPS, seed=42)


def test_fold_index_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="out of range"):
        fold_split(PAIRS, k=3, fold=3, mockup_ids=MOCKUPS, seed=42)
