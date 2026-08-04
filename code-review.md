# Code Review — Deep Layers

Scope: `scripts/`, `tests/unit/`, `pyproject.toml`, `requirements.txt`, notebook usage of the reviewed modules. Reviewed on 2026-08-04, branch `design/regional-delta-analysis`.

---

## 1. Code structure

```
scripts/
├── config.py              Pydantic Settings singleton (paths, hyperparameters)
├── reproducibility.py      Global RNG seeding
├── dataset.py              Pair discovery, grouped split, tf.data pipeline, padding
├── augmentation.py         Stateless paired RGB/IR augmentation
├── unet.py                 Baseline UNet
├── resunet.py               UNet with residual blocks
├── attention_unet.py       UNet with additive attention gates
├── efficientnet_unet.py    UNet with pretrained EfficientNetB0 encoder
├── losses.py                combined_loss, combined_loss_advanced (+ helpers)
├── metrics.py                PSNR / SSIM Keras metric wrappers
├── trainer.py                Model factory, compile, callbacks, checkpoint load
├── inference_utils.py       Overlapping-patch inference with Gaussian blending
└── visualization.py          Matplotlib plotting helpers
tests/unit/                  One test module per script, pure-logic focused
```

The module boundaries are clean and map 1:1 to pipeline stages (data → augmentation → model → loss/metric → train → infer → visualize). `trainer.py` acts as the single integration point that wires architectures to their loss via `_ADVANCED_LOSS_ARCHS`, which is a good "single source of truth" pattern — both training and checkpoint reloading derive the loss the same way, so they cannot silently diverge.

Docstrings are consistently NumPy-style with parameters, returns, and raises documented; type hints are used throughout. No circular imports were found, except one deliberately deferred import (`dataset.py:202`, see §3).

No CI configuration (`.github/workflows/`, etc.) was found despite a real `pytest` suite and `ruff` config existing — tests currently only run when a developer remembers to invoke them locally.

---

## 2. What each section does

**`config.py`** — A single `Settings` object (pydantic-settings, `.env`-overridable) is instantiated once at import time and holds every path and hyperparameter (patch multiple, batch size, LR, epochs, seed, split ratios, loss weights, crop size). All other modules import `settings` from here rather than hardcoding values.

**`reproducibility.py`** — `set_global_seed()` seeds Python/NumPy/TensorFlow RNGs via `tf.keras.utils.set_random_seed`. Deliberately does *not* enable `tf.config.experimental.enable_op_determinism()`, documented as unsupported/slow on the Metal backend; per-sample augmentation determinism is instead handled by a separate stateless-RNG mechanism.

**`dataset.py`** — Discovers RGB/IR pairs by matching filename stems (`load_image_pairs`), extracts an artwork ID from each filename (`extract_artwork_id`), and performs a two-stage `GroupShuffleSplit` (`grouped_train_val_test_split`) so all sections of one artwork stay in a single fold. `pad_to_multiple` pads a single image to the nearest multiple of `N` (used at whole-image inference time). `build_dataset` assembles the `tf.data` pipeline: load → shuffle → decode/normalize → (optional) augment with a deterministic per-element seed → batch → prefetch.

**`augmentation.py`** — `augment_pair` applies a shared random crop (concatenating RGB+IR on the channel axis so both share one crop box), then identical horizontal/vertical flips to both images, then brightness/contrast jitter to RGB only (IR reflectance is treated as illumination-independent). All randomness is stateless and seed-derived for reproducibility.

**`unet.py` / `resunet.py` / `attention_unet.py` / `efficientnet_unet.py`** — Four fully-convolutional encoder–decoder architectures, all accepting `(None, None, 3)` input and producing `(None, None, 1)` sigmoid output. `efficientnet_unet.py` additionally defines a custom `_ResizeToMatch` layer to reconcile off-by-one spatial mismatches between the frozen EfficientNetB0 backbone's floor-rounded downsampling and the decoder's exact-doubling `Conv2DTranspose`.

**`losses.py`** — `combined_loss` (MAE + 1−SSIM) for the three from-scratch architectures; `combined_loss_advanced` (MAE + Laplacian pyramid + FFT magnitude) for EfficientNet UNet. Laplacian levels are extracted via repeated avg-pool/upsample/subtract; FFT loss uses `rfft2d` magnitude difference normalized by image area.

**`metrics.py`** — `PSNRMetric` / `SSIMMetric`, stateful Keras `Metric` subclasses accumulating a running mean across batches.

**`trainer.py`** — `get_model` (name → builder factory), `uses_advanced_loss`/`compile_model` (architecture → loss selection, the SSOT mentioned above), `get_callbacks` (ModelCheckpoint/EarlyStopping/ReduceLROnPlateau/TensorBoard), `load_model` (checkpoint reload + recompile with matching loss).

**`inference_utils.py`** — `predict_with_overlap` tiles an arbitrarily large image into overlapping `patch_size` squares (reflect-padded if smaller than one patch), runs the model per patch, and blends outputs with a 2D Gaussian window (`_gaussian_window`) to avoid seam artifacts.

**`visualization.py`** — Plot helpers: sample RGB/IR grids, RGB|real|predicted triptychs, the absolute delta heatmap, and training-curve plots from a Keras `History.history` dict.

---

## 3. Issues found and possible fixes

### 3.1 No dimension validation on RGB/IR pairs (correctness risk)
`load_image_pairs` matches files by stem only; nothing checks that a matched RGB/IR pair has the same `(H, W)`. If they differ, `augment_pair` (`augmentation.py:53`) will fail deep inside `tf.concat([rgb, ir], axis=-1)` with a generic shape-mismatch error that gives no indication which artwork/section is at fault.
**Fix**: validate `rgb.shape[:2] == ir.shape[:2]` in `_load_pair` (or `load_image_pairs`) and raise a clear `ValueError` naming the offending file stem.

### 3.2 Deferred import in `dataset.py` (readability)
`build_dataset` imports `augment_pair` inside the function body (`dataset.py:202`) instead of at module level. No circular-import dependency exists between `dataset.py` and `augmentation.py` (augmentation only imports `tensorflow`), so this appears to be an unnecessary local import.
**Fix**: move `from scripts.augmentation import augment_pair` to the top of the file.

### 3.3 Duplicated `_conv_block` across four architecture modules (maintainability)
An identical two-line `Conv2D → BN → ReLU` × 2 block is redefined verbatim in `unet.py`, `attention_unet.py`, and `efficientnet_unet.py` (and a near-variant, `_residual_block`, in `resunet.py`). A change to the block (e.g. adding dropout, switching activation) requires editing three files in lockstep, with no compiler/test signal if one is missed.
**Fix**: extract `_conv_block` into a shared `scripts/blocks.py` and import it from the four architecture modules.

### 3.4 No CI pipeline
A real `pytest` suite and `ruff` configuration exist, but nothing runs them automatically on push/PR. Regressions can land on `main` without the suite being exercised.
**Fix**: add a minimal GitHub Actions workflow running `ruff check`, `ruff format --check`, and `pytest` on push/PR.

### 3.5 Early-bound `settings` defaults in function signatures (subtle footgun)
`trainer.py` (`compile_model`, `load_model`) and `config.py` bind `settings.LEARNING_RATE`, `settings.MODELS_DIR`, etc. as *default parameter values*, evaluated once when the module is imported. If `settings` is mutated at runtime (e.g. a notebook does `settings.LEARNING_RATE = 5e-4` to experiment), functions called without an explicit argument silently keep using the value captured at import time.
**Fix**: default to `None` and resolve `settings.X` inside the function body, or document explicitly that `settings` must be treated as immutable after import (the latter is cheaper and may already be the intended contract — worth stating in `config.py`'s docstring).

### 3.6 No `.env.example`
`config.py` supports `.env` overrides, but no template file documents the available keys for a new contributor.
**Fix**: add a `.env.example` listing every `Settings` field with its default and a one-line comment.

---

## 4. Possible optimizations

- **Batch patches in `predict_with_overlap`** (`inference_utils.py:120-126`): patches are run through the model one at a time in a nested Python loop (`model(patch_t, training=False)` per patch). For a large image this means hundreds of individual forward passes instead of a handful of batched ones. Collecting patches into batches of e.g. 8–16 before calling the model would substantially cut inference wall-clock time on GPU/Metal.
- **`tf.data.Dataset.cache()`**: `build_dataset` re-decodes JPEGs from disk every epoch. For a dataset of paintings that fits in memory, adding `.cache()` after `_load_pair` (before augmentation) would remove repeated JPEG-decode cost from every epoch after the first.
- **Mixed precision training**: none of the four architectures opt into `tf.keras.mixed_precision.set_global_policy("mixed_float16")`. On supported hardware this typically gives a meaningful throughput improvement for a negligible accuracy cost, especially useful given `EPOCHS=100` and four architectures to benchmark.
- **`_gaussian_window` recomputation**: recomputed on every `predict_with_overlap` call; harmless at current image counts but could be memoized (`functools.lru_cache`) if inference is run over many images in a batch job.

---

## 5. Possible additions for robustness

- **Regional delta analysis** (extends the design discussion already captured in `note.md`): the current `plot_delta` (`visualization.py:82`) is a raw `|real − predicted|` map, which conflates genuine hidden-detail signal with substrate/acquisition-driven gray-level shifts. Implementing the luminance/contrast/structure decomposition and per-window normalization discussed in `note.md` as a `scripts/delta_analysis.py` module would turn this from a qualitative visualization into a more defensible analytical output — directly relevant to the project's stated goal of surfacing underdrawings/pentimenti rather than acquisition artifacts.
- **Fail-fast data validation**: beyond the RGB/IR dimension check (§3.1), `grouped_train_val_test_split` has no guard against a fold ending up empty (e.g. too few distinct artworks for the requested `train_ratio`/`val_ratio`). A dataset with very few artworks would silently produce a degenerate split rather than a clear error.
- **Structural test for `efficientnet_unet`**: the EfficientNet UNet builder is excluded from unit tests because it downloads ImageNet weights. Instantiating it with `tf.keras.applications.EfficientNetB0(weights=None, ...)` would let a test verify the decoder wiring, skip-connection shapes, and `_ResizeToMatch` behaviour without any network dependency or slow download — currently that whole architecture has zero automated coverage of its own assembly logic.
- **Checkpoint/version metadata**: `get_callbacks`/`load_model` save/load `best_model.keras` with no accompanying metadata (git commit, config snapshot, training data split hash). For a research pipeline comparing four architectures over time, recording this alongside the checkpoint would make results reproducible and comparable months later.
- **Graceful handling of corrupt/unreadable images**: `_load_pair` assumes every file decodes cleanly; a single corrupt JPEG in `data/rgb` or `data/ir` will crash the whole `tf.data` pipeline mid-epoch rather than being skipped with a warning.
