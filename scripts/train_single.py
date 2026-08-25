"""Train exactly one architecture, in its own process, and exit.

Invoked as a subprocess (one call per architecture) by ``020_training.ipynb``,
``021_training_nll.ipynb``, and ``023_training_variants.ipynb`` instead of
training multiple models in a loop within a single notebook kernel.

Why a subprocess: training several Keras models back-to-back in one Python
process on this project's hardware (Apple Silicon, ``tensorflow-metal``)
leaves GPU-side state behind that ``tf.keras.backend.clear_session()`` does
not fully release — confirmed directly (``fixing.md`` §7): the *first*
architecture trained in a process always trains cleanly, every subsequent
one increasingly risks going ``NaN`` from epoch 1, regardless of which
architecture goes first. Every architecture trained alone in a fresh process
has trained cleanly. A subprocess per architecture gives each one a fresh
GPU context — the same condition, reproduced for real.

Writes the checkpoint to ``<model_dir>/<arch>/best_model.keras`` (via
:func:`scripts.trainer.get_callbacks`, unchanged) and the Keras
``History.history`` dict to ``<model_dir>/<arch>/history.json``, since a
dict can cross the process boundary only via a file, not in memory.
"""

import argparse
import json
from pathlib import Path

from scripts.config import settings
from scripts.dataset import (
    build_dataset,
    load_image_pairs,
    mockup_aware_train_val_test_split,
)
from scripts.reproducibility import set_global_seed
from scripts.trainer import compile_model, get_callbacks, get_model
from scripts.trainer_nll import compile_model_nll, get_model_nll


def _build_datasets():
    """Rebuild the artwork-and-mockups train/val split and datasets.

    Identical to the dataset cell shared by ``020``/``021``/``023`` —
    fully determined by ``settings`` (fixed seed), so no dataset state
    needs to cross the process boundary from the notebook.
    """
    pairs = load_image_pairs(settings.IR_DIR, settings.RGB_DIR)
    train_pairs, val_pairs, _ = mockup_aware_train_val_test_split(
        pairs,
        train_ratio=settings.TRAIN_RATIO,
        val_ratio=settings.VAL_RATIO,
        mockup_ids=settings.MOCKUP_ARTWORK_IDS,
        mockup_test_ratio=settings.MOCKUP_TEST_RATIO,
        seed=settings.SEED,
    )
    train_ds = build_dataset(
        train_pairs,
        batch_size=settings.BATCH_SIZE,
        augment=True,
        shuffle=True,
        seed=settings.SEED,
        crop_size=settings.CROP_SIZE,
    )
    val_ds = build_dataset(
        val_pairs,
        batch_size=settings.BATCH_SIZE,
        augment=False,
        shuffle=False,
    )
    return train_ds, val_ds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, help="Architecture name.")
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument(
        "--kwargs",
        default="{}",
        help="JSON dict of extra builder kwargs, e.g. unet_v2's variant flags.",
    )
    parser.add_argument(
        "--nll",
        action="store_true",
        help="Train the NLL variant (scripts.trainer_nll) instead of the "
        "deterministic one (scripts.trainer).",
    )
    parser.add_argument(
        "--loss-name",
        default="laplace_nll",
        help="NLL loss name (only used with --nll). One of "
        "scripts.trainer_nll.NLL_LOSSES.",
    )
    args = parser.parse_args()

    set_global_seed()
    train_ds, val_ds = _build_datasets()
    builder_kwargs = json.loads(args.kwargs)

    print(f"\n{'=' * 60}\n  Architecture: {args.arch}\n{'=' * 60}")

    if args.nll:
        model = get_model_nll(args.arch, **builder_kwargs)
        model = compile_model_nll(
            model,
            lr=settings.LEARNING_RATE,
            loss_name=args.loss_name,
            beta=settings.NLL_BETA,
            weight_decay=settings.WEIGHT_DECAY,
        )
    else:
        model = get_model(args.arch, **builder_kwargs)
        model = compile_model(
            model,
            args.arch,
            lr=settings.LEARNING_RATE,
            loss_alpha=settings.LOSS_ALPHA,
            weight_decay=settings.WEIGHT_DECAY,
        )
    model.summary(line_length=80)

    callbacks = get_callbacks(args.arch, log_dir=args.log_dir, model_dir=args.model_dir)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    history_path = args.model_dir / args.arch / "history.json"
    history_path.write_text(json.dumps(history.history))

    best_val_loss = min(history.history["val_loss"])
    print(f"\nBest val_loss ({args.arch}): {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
