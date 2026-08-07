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

### 3.1 No dimension validation on RGB/IR pairs (correctness risk) — FIXED
`load_image_pairs` matched files by stem only; nothing checked that a matched RGB/IR pair had the same `(H, W)`. If they differed, `augment_pair` (`augmentation.py:53`) would fail deep inside `tf.concat([rgb, ir], axis=-1)` with a generic shape-mismatch error that gave no indication which artwork/section was at fault.
**Fix applied**: `load_image_pairs` (`dataset.py`) now opens every matched pair with PIL and raises a `ValueError` naming the offending file stem and both sizes if `rgb_size != ir_size`, before the pair ever reaches the `tf.data` pipeline. Covered by a new regression test, `test_load_image_pairs_raises_on_size_mismatch` (`tests/unit/test_dataset.py`). This supersedes `010_eda.ipynb`'s old first-20-pairs manual check (§5 of that notebook), which is now purely descriptive (reports the size distribution across all pairs) since correctness is guaranteed upstream.

### 3.2 Deferred import in `dataset.py` (readability) — FIXED
`build_dataset` imported `augment_pair` inside the function body (`dataset.py:202`) instead of at module level. No circular-import dependency exists between `dataset.py` and `augmentation.py` (augmentation only imports `tensorflow`), so this was an unnecessary local import.
**Fix applied**: `from scripts.augmentation import augment_pair` moved to the top of `dataset.py`.

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
- **Two-phase fine-tuning for `efficientnet_unet`**: `build_efficientnet_unet` already exposes `freeze_encoder=False` (`efficientnet_unet.py:85-88`), but nothing in `trainer.py`/the training notebook actually uses it — the encoder trains fully frozen for all 100 epochs. The standard transfer-learning recipe (train the decoder to convergence with the encoder frozen, then unfreeze and fine-tune end-to-end at a lower learning rate, e.g. `LEARNING_RATE / 10`) is currently unimplemented and could improve accuracy without any architectural change.
- **No explicit regularization in the three from-scratch architectures**: `unet.py`, `resunet.py`, `attention_unet.py` rely on BatchNorm alone; the 1024-channel bottleneck (`unet.py:68`, `resunet.py:73`, `attention_unet.py:104`) is large relative to a painting-scale dataset split by artwork (likely tens, not thousands, of distinct works). Adding spatial dropout at the bottleneck/deeper decoder stages would be a low-risk way to reduce overfitting risk on a small dataset.
- **EfficientNetB0 backbone is dated (2019)**: `tf.keras.applications` also ships more recent ImageNet-pretrained backbones (e.g. `EfficientNetV2B0`, `ConvNeXtTiny`) as drop-in replacements for the `EfficientNetB0(...)` call in `efficientnet_unet.py:99-104` — same `include_top=False`/frozen-encoder pattern would apply, only the `_SKIP_LAYER_NAMES` (`efficientnet_unet.py:8-14`) would need to be re-mapped to the new backbone's layer names. See §7 "Alternative architectures" for literature justifying this and other backbone/architecture changes.

---

## 5. Architecture-specific comments

All four architectures (`unet.py`, `resunet.py`, `attention_unet.py`, `efficientnet_unet.py`) are purely convolutional, feed-forward encoder–decoders — no GAN/adversarial term, no diffusion process, no self-attention/transformer block anywhere in the codebase or notebooks (verified by a case-insensitive search for `gan|adversarial|discriminator|diffusion|transformer|vit\b`). Given the likely small size of a paired painting dataset (grouped train/val/test split by artwork ID, `config.py:67-68`), this is a defensible, low-data-friendly design rather than an oversight — see §7 for where a *more* data-hungry architecture would or would not be justified.

- **`attention_unet.py`'s "attention" is CNN spatial gating, not self-attention.** `_attention_gate` (`attention_unet.py:31-65`) implements Oktay et al.'s additive gate — it reweights the encoder skip connection using the decoder's gating signal, but has no long-range/global receptive field. A true self-attention block at the bottleneck (where the receptive field is largest and spatial resolution smallest, so the quadratic cost of self-attention is cheapest) would let the model relate distant regions of the same painting — potentially useful since underdrawing strokes can be a coherent pattern spanning a whole figure, not just local texture. See §7.3 (Restormer) for a concrete, resolution-scalable way to do this.
- **`_ResizeToMatch` in `efficientnet_unet.py`** (lines 18-35) is a well-targeted, minimal fix for the off-by-one skip/upsample mismatch caused by EfficientNet's floor-rounded stride-2 convolutions; no change recommended here, noting it only because it is the one architecture-specific engineering workaround in the codebase and would need re-verifying if the backbone is ever swapped (different backbones round dimensions differently or not at all).
- **Bottleneck channel counts are inconsistent across from-scratch architectures vs. EfficientNet UNet**: `unet.py`/`resunet.py`/`attention_unet.py` all default to a 1024-channel bottleneck (e.g. `unet.py:68`) fed by a `[64,128,256,512]` encoder, while `efficientnet_unet.py`'s decoder starts at 256 channels (`efficientnet_unet.py:97`) because the frozen EfficientNetB0 backbone already outputs 1280 channels at the bottleneck. This isn't a bug (the two are genuinely different feature scales) but makes head-to-head parameter-count/capacity comparisons between the four architectures less apples-to-apples than the model-comparison notebook (`030_evaluation.ipynb`) might assume.

## 6. Possible additions for robustness

- **Regional delta analysis** (extends the design discussion already captured in `note.md`): the current `plot_delta` (`visualization.py:82`) is a raw `|real − predicted|` map, which conflates genuine hidden-detail signal with substrate/acquisition-driven gray-level shifts. Implementing the luminance/contrast/structure decomposition and per-window normalization discussed in `note.md` as a `scripts/delta_analysis.py` module would turn this from a qualitative visualization into a more defensible analytical output — directly relevant to the project's stated goal of surfacing underdrawings/pentimenti rather than acquisition artifacts.
- **Fail-fast data validation**: beyond the RGB/IR dimension check (§3.1), `grouped_train_val_test_split` has no guard against a fold ending up empty (e.g. too few distinct artworks for the requested `train_ratio`/`val_ratio`). A dataset with very few artworks would silently produce a degenerate split rather than a clear error.
- **Structural test for `efficientnet_unet`**: the EfficientNet UNet builder is excluded from unit tests because it downloads ImageNet weights. Instantiating it with `tf.keras.applications.EfficientNetB0(weights=None, ...)` would let a test verify the decoder wiring, skip-connection shapes, and `_ResizeToMatch` behaviour without any network dependency or slow download — currently that whole architecture has zero automated coverage of its own assembly logic.
- **Checkpoint/version metadata**: `get_callbacks`/`load_model` save/load `best_model.keras` with no accompanying metadata (git commit, config snapshot, training data split hash). For a research pipeline comparing four architectures over time, recording this alongside the checkpoint would make results reproducible and comparable months later.
- **Graceful handling of corrupt/unreadable images**: `_load_pair` assumes every file decodes cleanly; a single corrupt JPEG in `data/rgb` or `data/ir` will crash the whole `tf.data` pipeline mid-epoch rather than being skipped with a warning.


## 7. Alternative architectures

The four current architectures (`unet.py`, `resunet.py`, `attention_unet.py`,
`efficientnet_unet.py`) are all supervised, deterministic, purely
convolutional encoder–decoders — no adversarial term, no diffusion process,
no self-attention/transformer block anywhere in the codebase. This is
well-suited to a small, artwork-grouped dataset, but the following
directions are worth considering if the goal is to push prediction quality
further, roughly ordered from "most compatible with a small dataset" to
"most data-hungry."

### 7.1. Adversarial term (PatchGAN / pix2pix-style)

Add a lightweight PatchGAN discriminator on top of one of the existing UNet
generators, keeping the current pixel/frequency losses (`combined_loss`,
`combined_loss_advanced`) as an auxiliary term alongside the adversarial
loss. This is the standard recipe for image-to-image translation and tends
to sharpen fine detail beyond what pixel-wise regression losses alone can
achieve.

- P. Isola, J.-Y. Zhu, T. Zhou, A.A. Efros, *"Image-to-Image Translation
  with Conditional Adversarial Networks,"* CVPR, 2017.
  https://doi.org/10.48550/arXiv.1611.07004 — the pix2pix framework this
  recipe is based on.
- **Directly on-domain precedent**: G.H. Cann, A. Bourached, R.-R. Griffiths,
  D.G. Stork, *"Resolution enhancement in the recovery of underdrawings via
  style transfer by generative adversarial deep neural networks,"* 2021.
  https://doi.org/10.48550/arXiv.2102.00209 — multi-scale GAN recovering
  underdrawings/ghost-paintings on works by Leonardo, explicitly addressing
  the scarcity of training data typical of the art domain.
- Companion paper, same authors: *"Recovery of underdrawings and
  ghost-paintings via style transfer by deep convolutional neural networks:
  A digital tool for art scholars,"* 2021.
  https://doi.org/10.48550/arXiv.2101.10807

### 7.2. Modern CNN backbone (ConvNeXt / ConvNeXt V2)

Drop-in replacement for the EfficientNetB0 encoder in `efficientnet_unet.py`
(same frozen-backbone pattern, only `_SKIP_LAYER_NAMES` would need
re-mapping). Keeps the convolutional inductive bias — important with little
data — while closing most of the accuracy gap to transformers.

- Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie, *"A
  ConvNet for the 2020s,"* CVPR, 2022.
  https://doi.org/10.48550/arXiv.2201.03545
- S. Woo et al., *"ConvNeXt V2: Co-designing and Scaling ConvNets with
  Masked Autoencoders,"* CVPR, 2023.
  https://doi.org/10.48550/arXiv.2301.00808

### 7.3. Efficient transformer block at the bottleneck (Restormer)

Rather than replacing a whole architecture, insert a Restormer-style
transposed-attention block only at the bottleneck (smallest spatial
resolution, largest receptive field — where the quadratic cost of
self-attention is cheapest). Restormer computes attention across channels
instead of pixels, so its complexity is linear, not quadratic, in image
resolution — relevant here since paintings are processed via
`predict_with_overlap` at potentially large sizes.

- S.W. Zamir et al., *"Restormer: Efficient Transformer for High-Resolution
  Image Restoration,"* CVPR, 2022 (oral).
  https://doi.org/10.48550/arXiv.2111.09881

### 7.4. Pure transformer U-Net (Swin-UNet) — use with caution

A fully transformer-based U-shaped architecture. Typically more data-hungry
than CNNs; the same authors who published the on-domain GAN papers above
(§1) explicitly flag the scarcity of paired training works in the art
domain, so this direction should only be pursued if the RGB/IR dataset
turns out to be large enough — worth checking dataset size before
considering this.

- H. Cao et al., *"Swin-Unet: Unet-like Pure Transformer for Medical Image
  Segmentation,"* 2021. https://doi.org/10.48550/arXiv.2105.05537

### 7.5. Multi-band / multispectral extension (MST++)

Relevant only if the project extends beyond a single IR band to multiband
reflectography — a spectral-wise transformer designed for exactly this kind
of narrow-band-from-RGB reconstruction problem, winner of the NTIRE 2022
Spectral Recovery Challenge.

- Y. Cai et al., *"MST++: Multi-stage Spectral-wise Transformer for
  Efficient Spectral Reconstruction,"* CVPRW, 2022.
  https://doi.org/10.48550/arXiv.2204.07908