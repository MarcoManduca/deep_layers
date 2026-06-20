"""Global random-seed management for reproducible runs."""

import tensorflow as tf

from scripts.config import settings


def set_global_seed(seed: int = settings.SEED) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs from a single value.

    Sets the seed for the ``random`` module, NumPy, and TensorFlow in one
    call (via :func:`tf.keras.utils.set_random_seed`), making weight
    initialisation and dataset shuffling reproducible across runs.

    Per-sample augmentation reproducibility is handled separately by the
    stateless RNG in :func:`scripts.augmentation.augment_pair`, which is
    keyed by a deterministic per-element counter in
    :func:`scripts.dataset.build_dataset`.

    Notes
    -----
    Full op-level determinism (``tf.config.experimental.enable_op_determinism``)
    is intentionally **not** enabled here: it is unsupported by the Metal
    backend used on Apple Silicon and can degrade throughput. Seeding the
    RNGs plus the stateless augmentation already yields run-to-run
    reproducibility for this pipeline.

    Parameters
    ----------
    seed : int
        Seed value. Defaults to ``settings.SEED``.
    """
    tf.keras.utils.set_random_seed(seed)
