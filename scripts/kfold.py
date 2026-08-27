"""Grouped k-fold splits for Round 4 cross-validation (``fixing.md`` #4).

Round 4's purpose is a genuine held-out variance estimate — how much the
metrics move across independent train/val splits — instead of the single
fixed split every other notebook uses.

Splitting rules (mirroring :func:`scripts.dataset.mockup_aware_train_val_test_split`
one level up):

- **Real artworks** are partitioned into ``k`` groups *by artwork ID*, so every
  section of an artwork stays in one group — the same anti-leakage guarantee as
  :func:`scripts.dataset.grouped_train_val_test_split`. Fold ``i`` holds out
  group ``i`` as the validation set; the other ``k-1`` groups are train.
- **Mockups** always go to train, in every fold. They are synthetic aids that
  exist to be learned from, not generalised to (same rationale as the
  ``mockup_aware`` split), so rotating them through the held-out fold would only
  add noise to the per-fold metric.

There is no separate test set here: each fold's held-out real artworks *are* the
evaluation set for that fold. ``data/test/`` (the GT-annotated paintings) stays
untouched — detection metrics are still reported there, now per fold-model.
"""

from pathlib import Path

import numpy as np

from scripts.dataset import extract_artwork_id

Pair = tuple[Path, Path]


def _real_and_mockup(
    pairs: list[Pair], mockup_ids: set[str]
) -> tuple[list[tuple[Pair, str]], list[Pair]]:
    tagged = [(p, extract_artwork_id(p[0].stem)) for p in pairs]
    real = [(p, g) for p, g in tagged if g not in mockup_ids]
    mockup = [p for p, g in tagged if g in mockup_ids]
    return real, mockup


def fold_artwork_groups(
    pairs: list[Pair],
    k: int,
    mockup_ids: list[str] | None = None,
    seed: int = 42,
) -> list[list[str]]:
    """Return the held-out artwork IDs for each of the ``k`` folds.

    Deterministic in ``(k, seed)`` and in the set of real-artwork IDs present
    in ``pairs`` — the fold a given artwork lands in does not depend on how many
    sections it has or on file order.
    """
    if mockup_ids is None:
        from scripts.config import settings

        mockup_ids = settings.MOCKUP_ARTWORK_IDS
    real, _ = _real_and_mockup(pairs, set(mockup_ids))

    unique = sorted({g for _, g in real})
    if len(unique) < k:
        raise ValueError(
            f"{len(unique)} real-artwork groups is fewer than k={k} — "
            "not enough groups to form that many folds."
        )

    shuffled = list(np.random.default_rng(seed).permutation(unique))
    return [sorted(chunk.tolist()) for chunk in np.array_split(shuffled, k)]


def grouped_kfold_splits(
    pairs: list[Pair],
    k: int,
    mockup_ids: list[str] | None = None,
    seed: int = 42,
) -> list[tuple[list[Pair], list[Pair]]]:
    """All ``k`` ``(train_pairs, val_pairs)`` splits.

    ``val_pairs`` is the fold's held-out real artworks; ``train_pairs`` is every
    other real artwork plus *all* mockup pairs.
    """
    if mockup_ids is None:
        from scripts.config import settings

        mockup_ids = settings.MOCKUP_ARTWORK_IDS
    mockup_id_set = set(mockup_ids)
    real, mockup = _real_and_mockup(pairs, mockup_id_set)
    held_out = fold_artwork_groups(pairs, k, mockup_ids=mockup_ids, seed=seed)

    splits: list[tuple[list[Pair], list[Pair]]] = []
    for fold_groups in held_out:
        val_set = set(fold_groups)
        val = [p for p, g in real if g in val_set]
        train = [p for p, g in real if g not in val_set] + list(mockup)
        splits.append((train, val))
    return splits


def fold_split(
    pairs: list[Pair],
    k: int,
    fold: int,
    mockup_ids: list[str] | None = None,
    seed: int = 42,
) -> tuple[list[Pair], list[Pair]]:
    """The single ``(train_pairs, val_pairs)`` split for one fold index."""
    if not 0 <= fold < k:
        raise ValueError(f"fold {fold} out of range for k={k}")
    return grouped_kfold_splits(pairs, k, mockup_ids=mockup_ids, seed=seed)[fold]
