"""Dataset loading, splitting, and tf.data pipeline for the deep-layers pipeline."""

from pathlib import Path

import tensorflow as tf
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from scripts.augmentation import augment_pair


def extract_artwork_id(filename: str) -> str:
    """Extract artwork ID from a section filename stem.

    Parameters
    ----------
    filename : str
        Filename stem such as ``"a1_sezione_0"``.

    Returns
    -------
    str
        Artwork identifier, e.g. ``"a1"``.

    Examples
    --------
    >>> extract_artwork_id("a1_sezione_0")
    'a1'
    >>> extract_artwork_id("natmorta1_sezione_12")
    'natmorta1'
    """
    return filename.rsplit("_sezione_", 1)[0]


def load_image_pairs(
    ir_dir: Path,
    rgb_dir: Path,
) -> list[tuple[Path, Path]]:
    """Collect matched RGB / IR image pairs from two directories.

    Pairs are matched by filename stem and sorted alphabetically.

    Parameters
    ----------
    ir_dir : Path
        Directory containing IR images (grayscale JPEG).
    rgb_dir : Path
        Directory containing RGB images (colour JPEG).

    Returns
    -------
    list[tuple[Path, Path]]
        Sorted list of ``(rgb_path, ir_path)`` tuples.

    Raises
    ------
    ValueError
        If no common filename stems exist between the two directories, or
        if a matched RGB/IR pair has mismatched ``(width, height)``.
    """
    ir_files = {p.stem: p for p in sorted(ir_dir.glob("*.jpg"))}
    rgb_files = {p.stem: p for p in sorted(rgb_dir.glob("*.jpg"))}

    common_stems = sorted(set(ir_files) & set(rgb_files))
    if not common_stems:
        raise ValueError(f"No matching pairs found between {ir_dir} and {rgb_dir}")

    pairs = [(rgb_files[stem], ir_files[stem]) for stem in common_stems]

    for rgb_path, ir_path in pairs:
        rgb_size = Image.open(rgb_path).size
        ir_size = Image.open(ir_path).size
        if rgb_size != ir_size:
            raise ValueError(
                f"RGB/IR size mismatch for '{rgb_path.stem}': "
                f"RGB is {rgb_size} but IR is {ir_size}"
            )

    return pairs


def grouped_train_val_test_split(
    pairs: list[tuple[Path, Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
]:
    """Split image pairs into train / val / test by artwork ID.

    All sections of the same artwork are assigned to a single fold,
    preventing any data leakage between splits.

    Parameters
    ----------
    pairs : list[tuple[Path, Path]]
        Sorted list of ``(rgb_path, ir_path)`` tuples.
    train_ratio : float
        Fraction of artworks assigned to the training fold.
    val_ratio : float
        Fraction of artworks assigned to the validation fold.
        The test fraction is ``1 - train_ratio - val_ratio``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[list, list, list]
        ``(train_pairs, val_pairs, test_pairs)``
    """
    groups = [extract_artwork_id(p[0].stem) for p in pairs]
    test_ratio = 1.0 - train_ratio - val_ratio

    splitter_1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(splitter_1.split(pairs, groups=groups))

    trainval_pairs = [pairs[i] for i in trainval_idx]
    trainval_groups = [groups[i] for i in trainval_idx]
    test_pairs = [pairs[i] for i in test_idx]

    relative_val = val_ratio / (train_ratio + val_ratio)
    splitter_2 = GroupShuffleSplit(
        n_splits=1, test_size=relative_val, random_state=seed
    )
    train_idx, val_idx = next(splitter_2.split(trainval_pairs, groups=trainval_groups))

    train_pairs = [trainval_pairs[i] for i in train_idx]
    val_pairs = [trainval_pairs[i] for i in val_idx]

    return train_pairs, val_pairs, test_pairs


def mockup_aware_train_val_test_split(
    pairs: list[tuple[Path, Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    mockup_ids: list[str] | None = None,
    mockup_test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
]:
    """Split image pairs into train / val / test, treating mockup groups apart.

    Some artwork IDs are not real paintings but synthetic paint-on-support
    mockups created specifically to aid training (see ``config.MOCKUP_ARTWORK_IDS``).
    Unlike real artworks, holding an entire mockup group out for test would
    both waste useful training signal and is not needed to prevent leakage,
    since these groups exist to be learned from rather than generalized to.

    Real artworks (any ID not in ``mockup_ids``) are split with
    ``grouped_train_val_test_split``, keeping every section of an artwork in a
    single fold. Mockup groups are instead split at the individual pair level
    with ``mockup_test_ratio`` sent to test (default 5%) and the remainder
    split between train/val according to ``train_ratio``/``val_ratio`` — so
    each mockup group ends up (almost entirely) in train/val, with only a
    small sample held out for test. The two splits are then concatenated.

    Parameters
    ----------
    pairs : list[tuple[Path, Path]]
        Sorted list of ``(rgb_path, ir_path)`` tuples.
    train_ratio : float
        Fraction of real artworks, and of each mockup group's non-test
        remainder, assigned to the training fold.
    val_ratio : float
        Fraction of real artworks, and of each mockup group's non-test
        remainder, assigned to the validation fold.
    mockup_ids : list[str] or None
        Artwork IDs to treat as mockups. Defaults to
        ``config.settings.MOCKUP_ARTWORK_IDS``.
    mockup_test_ratio : float
        Fraction of mockup pairs sent to test (per mockup group, at the pair
        level, not the group level).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[list, list, list]
        ``(train_pairs, val_pairs, test_pairs)``
    """
    if mockup_ids is None:
        from scripts.config import settings

        mockup_ids = settings.MOCKUP_ARTWORK_IDS
    mockup_ids = set(mockup_ids)

    groups = [extract_artwork_id(p[0].stem) for p in pairs]
    mockup_pairs = [p for p, g in zip(pairs, groups) if g in mockup_ids]
    artwork_pairs = [p for p, g in zip(pairs, groups) if g not in mockup_ids]

    train_pairs, val_pairs, test_pairs = [], [], []
    if artwork_pairs:
        artwork_train, artwork_val, artwork_test = grouped_train_val_test_split(
            artwork_pairs, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )
        train_pairs += artwork_train
        val_pairs += artwork_val
        test_pairs += artwork_test

    if mockup_pairs:
        mockup_trainval, mockup_test = train_test_split(
            mockup_pairs, test_size=mockup_test_ratio, random_state=seed
        )
        relative_val = val_ratio / (train_ratio + val_ratio)
        mockup_train, mockup_val = train_test_split(
            mockup_trainval, test_size=relative_val, random_state=seed
        )
        train_pairs += mockup_train
        val_pairs += mockup_val
        test_pairs += mockup_test

    return train_pairs, val_pairs, test_pairs


def pad_to_multiple(
    image: tf.Tensor,
    multiple: int = 16,
) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
    """Pad a single image so that H and W are multiples of ``multiple``.

    Parameters
    ----------
    image : tf.Tensor
        Image tensor of shape ``(H, W, C)``.
    multiple : int
        Target divisor for both spatial dimensions.

    Returns
    -------
    tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]
        ``(padded_image, (original_h, original_w))``
    """
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = tf.pad(image, [[0, pad_h], [0, pad_w], [0, 0]])
    return padded, (h, w)


def _load_pair(
    rgb_path: tf.Tensor,
    ir_path: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """TF-graph-compatible function to decode and normalise one pair."""
    rgb = tf.io.read_file(rgb_path)
    rgb = tf.image.decode_jpeg(rgb, channels=3)
    rgb = tf.cast(rgb, tf.float32) / 255.0

    ir = tf.io.read_file(ir_path)
    ir = tf.image.decode_jpeg(ir, channels=1)
    ir = tf.cast(ir, tf.float32) / 255.0

    return rgb, ir


def build_dataset(
    pairs: list[tuple[Path, Path]],
    batch_size: int = 8,
    augment: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    crop_size: int | None = None,
) -> tf.data.Dataset:
    """Build a ``tf.data.Dataset`` pipeline from a list of image pairs.

    Parameters
    ----------
    pairs : list[tuple[Path, Path]]
        List of ``(rgb_path, ir_path)`` tuples.
    batch_size : int
        Number of samples per batch.
    augment : bool
        Apply random augmentation (intended for the training split only).
    shuffle : bool
        Randomly shuffle samples before batching.
    seed : int
        Seed used for shuffling.
    crop_size : int or None
        If set, randomly crop each pair to ``(crop_size, crop_size)`` as part
        of augmentation. Must be a multiple of 16 (the 4-level UNet pooling
        factor). Has effect only when ``augment=True``; evaluation and
        inference always run on full images.

    Returns
    -------
    tf.data.Dataset
        Dataset that yields ``(rgb_batch, ir_batch)`` pairs where shapes
        are ``(B, H, W, 3)`` and ``(B, H, W, 1)`` respectively.

    Raises
    ------
    ValueError
        If ``crop_size`` is set but is not a positive multiple of 16.
    """
    if crop_size is not None and (crop_size <= 0 or crop_size % 16 != 0):
        raise ValueError(f"crop_size ({crop_size}) must be a positive multiple of 16.")

    rgb_paths = [str(p[0]) for p in pairs]
    ir_paths = [str(p[1]) for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((rgb_paths, ir_paths))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), seed=seed)

    ds = ds.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        # Pair each element with a monotonic counter so the stateless
        # augmentation seed is deterministic per sample (and varies across
        # epochs), making the augmented stream reproducible run-to-run.
        counter = tf.data.Dataset.counter()
        ds = tf.data.Dataset.zip((ds, counter))
        ds = ds.map(
            lambda pair, c: augment_pair(
                pair[0],
                pair[1],
                seed=tf.stack([seed, tf.cast(c, tf.int32)]),
                crop_size=crop_size,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
