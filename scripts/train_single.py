"""Train exactly one architecture, in its own process, and exit.

Invoked as a subprocess (one call per architecture) by ``020_training.ipynb``,
``021_training_nll.ipynb``, ``023_training_variants.ipynb``, and
``024_training_beta_sweep.ipynb`` (one call per *beta*, same architecture)
instead of training multiple models in a loop within a single notebook kernel.

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

Also supports ``fixing.md`` #6's two-phase EfficientNet fine-tuning: pass
``--init-from <phase-1-checkpoint>`` to warm-start this process's model from
an existing checkpoint's weights before compiling (e.g. phase 1 trains
``efficientnet_unet`` with the encoder frozen; phase 2 trains
``efficientnet_unet_ft`` with ``--kwargs '{"freeze_encoder": false}'
--init-from models/deterministic/efficientnet_unet/best_model.keras``, and
usually a lower ``--lr``, ``settings.FINETUNE_LEARNING_RATE``).

``--nll-beta`` overrides ``settings.NLL_BETA`` for one run, so a beta sweep
(``024_training_beta_sweep.ipynb``) records each run's beta in the command
that launched it instead of in ambient config, where it would not survive
into the checkpoint. Point ``--model-dir`` at a per-beta directory so the
runs do not overwrite each other.

``--fold N --kfold-k K`` switches the train/val split to
``scripts.kfold.fold_split`` for ``fixing.md`` #4's Round 4 cross-validation
(``060_kfold_training.ipynb`` — one subprocess per fold, each resumable).
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
from scripts.kfold import fold_split
from scripts.reproducibility import set_global_seed
from scripts.trainer import compile_model, get_callbacks, get_model
from scripts.trainer_nll import compile_model_nll, get_model_nll


def _build_datasets(fold: int | None = None, kfold_k: int = settings.KFOLD_K):
    """Rebuild the train/val split and datasets.

    Fully determined by ``settings`` (fixed seed), so no dataset state needs to
    cross the process boundary from the notebook.

    With ``fold is None`` (default): the artwork-and-mockups split shared by
    ``020``/``021``/``023``. With ``fold`` set: the grouped k-fold split
    (``scripts.kfold``, ``fixing.md`` #4's Round 4) — fold ``fold`` of
    ``kfold_k``, held-out real artworks as val, the rest plus all mockups as
    train. There is no test split in k-fold mode; the held-out artworks are the
    fold's evaluation set.
    """
    pairs = load_image_pairs(settings.IR_DIR, settings.RGB_DIR)
    if fold is None:
        train_pairs, val_pairs, _ = mockup_aware_train_val_test_split(
            pairs,
            train_ratio=settings.TRAIN_RATIO,
            val_ratio=settings.VAL_RATIO,
            mockup_ids=settings.MOCKUP_ARTWORK_IDS,
            mockup_test_ratio=settings.MOCKUP_TEST_RATIO,
            seed=settings.SEED,
        )
    else:
        train_pairs, val_pairs = fold_split(
            pairs,
            k=kfold_k,
            fold=fold,
            mockup_ids=settings.MOCKUP_ARTWORK_IDS,
            seed=settings.KFOLD_SEED,
        )
        print(
            f"k-fold: fold {fold}/{kfold_k}  "
            f"train={len(train_pairs)}  val(held-out)={len(val_pairs)}"
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
    parser.add_argument(
        "--nll-beta",
        type=float,
        default=None,
        help="Beta exponent of the beta-weighted NLL losses (only used with "
        "--nll, meaningful for 'beta_nll'/'laplace_nll' and ignored by "
        "'gaussian_nll'). Defaults to settings.NLL_BETA. Pass it explicitly "
        "to sweep beta without touching ambient config, so the value a run "
        "used is recorded in the command that launched it.",
    )
    parser.add_argument(
        "--loss-alpha",
        type=float,
        default=None,
        help="Charbonnier weight of scripts.losses.combined_loss (ignored "
        "with --nll); (1 - alpha) weights the (1 - MS-SSIM) term. Defaults "
        "to settings.LOSS_ALPHA. Pass it explicitly to sweep the "
        "fidelity/structure trade-off without touching ambient config, so "
        "the value a run used is recorded in the command that launched it.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Adam learning rate. Defaults to settings.LEARNING_RATE; pass "
        "settings.FINETUNE_LEARNING_RATE explicitly for a phase-2 "
        "fine-tuning run (fixing.md #6).",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Path to a .keras checkpoint to warm-start this model's weights "
        "from, before compiling (fixing.md #6's two-phase EfficientNet "
        "fine-tuning: phase 2 builds with --kwargs "
        "'{\"freeze_encoder\": false}' and initializes from phase 1's "
        "checkpoint here). The architecture must be unchanged from the "
        "checkpoint's — only `trainable` differs between the two phases — "
        "so a plain tf.keras.Model.load_weights (matched by layer name/"
        "order, not the full model config) is sufficient; unlike "
        "scripts.trainer.load_model, this does not require the checkpoint "
        "to be recompiled first.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Grouped k-fold index (fixing.md #4's Round 4). When set, the "
        "train/val split is scripts.kfold.fold_split(k=--kfold-k, fold=--fold) "
        "instead of the standard mockup-aware split: held-out real artworks are "
        "val, the rest plus all mockups are train, no test split. Point "
        "--model-dir at a per-fold directory (e.g. models/kfold/fold_0).",
    )
    parser.add_argument(
        "--kfold-k",
        type=int,
        default=settings.KFOLD_K,
        help="Number of folds (only used with --fold). Defaults to settings.KFOLD_K.",
    )
    args = parser.parse_args()

    set_global_seed()
    train_ds, val_ds = _build_datasets(fold=args.fold, kfold_k=args.kfold_k)
    builder_kwargs = json.loads(args.kwargs)
    lr = args.lr if args.lr is not None else settings.LEARNING_RATE
    nll_beta = args.nll_beta if args.nll_beta is not None else settings.NLL_BETA
    loss_alpha = args.loss_alpha if args.loss_alpha is not None else settings.LOSS_ALPHA

    banner = (
        f"{args.arch} ({args.loss_name}, beta={nll_beta})"
        if args.nll
        else f"{args.arch} (combined_loss, alpha={loss_alpha})"
    )
    print(f"\n{'=' * 60}\n  Architecture: {banner}\n{'=' * 60}")

    if args.nll:
        model = get_model_nll(args.arch, **builder_kwargs)
        if args.init_from is not None:
            print(f"Initializing weights from {args.init_from}")
            model.load_weights(str(args.init_from))
        model = compile_model_nll(
            model,
            lr=lr,
            loss_name=args.loss_name,
            beta=nll_beta,
            weight_decay=settings.WEIGHT_DECAY,
        )
    else:
        model = get_model(args.arch, **builder_kwargs)
        if args.init_from is not None:
            print(f"Initializing weights from {args.init_from}")
            model.load_weights(str(args.init_from))
        model = compile_model(
            model,
            args.arch,
            lr=lr,
            loss_alpha=loss_alpha,
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
