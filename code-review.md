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
- **Mockup-aware train/val/test split** — IMPLEMENTED (`scripts/dataset.py:135-215`, `mockup_aware_train_val_test_split`): 6 of the 30 artwork groups (`tblu`, `tbianco`, `tbruno`, `tgiallo`, `trosso`, `tverde` — configurable via `config.settings.MOCKUP_ARTWORK_IDS`) are not real paintings but synthetic paint-on-support mockups created specifically to aid training, and together account for ~34% of all pairs (400/1164). The original `grouped_train_val_test_split` treats every artwork ID as an indivisible group to prevent leakage between sections of the same painting — correct for real artworks, but wasteful for mockup groups, which risked being held out of training entirely by an unlucky group-level split despite existing purely to be learned from. The new function keeps the group-level, leakage-free split for real artworks unchanged, but splits mockup groups at the individual pair level instead, sending only a small fraction (`mockup_test_ratio`, default 5%) to test and the rest to train/val. Added as a new function rather than a change to the existing one (both are wired as commented, side-by-side alternative blocks — "artworks split" vs "artwork and mockups split" — in `010_eda.ipynb`, `020_training.ipynb`, `030_evaluation.ipynb`) so existing checkpoints/results trained on the artworks split remain reproducible. Covered by `tests/unit/test_dataset.py::test_mockup_aware_split_*` (mockup groups end up mostly in train/val, real artworks still leakage-free, every pair covered exactly once).

---

## 5. Architecture-specific comments

The four original architectures (`unet.py`, `resunet.py`, `attention_unet.py`, `efficientnet_unet.py`) are purely convolutional, feed-forward encoder–decoders — no GAN/adversarial term, no diffusion process, no self-attention/transformer block. Given the likely small size of a paired painting dataset (grouped train/val/test split by artwork ID, `config.py:67-68`), this was a defensible, low-data-friendly starting point rather than an oversight — see §7 for where a *more* data-hungry architecture would or would not be justified. `scripts/unet_restormer.py` (§7.3) has since added one self-attention block, but scoped narrowly (bottleneck only, linear-cost channel attention) rather than adopting a data-hungry pure-transformer design (§7.4).

- **`attention_unet.py`'s "attention" is CNN spatial gating, not self-attention.** `_attention_gate` (`attention_unet.py:31-65`) implements Oktay et al.'s additive gate — it reweights the encoder skip connection using the decoder's gating signal, but has no long-range/global receptive field. A true self-attention block at the bottleneck (where the receptive field is largest and spatial resolution smallest, so the quadratic cost of self-attention is cheapest) would let the model relate distant regions of the same painting — potentially useful since underdrawing strokes can be a coherent pattern spanning a whole figure, not just local texture. `scripts/unet_restormer.py` (§7.3) implements this — on `unet`, not `attention_unet`, so the two mechanisms (spatial gating vs. channel self-attention) haven't yet been combined in one architecture.
- **`_ResizeToMatch` in `efficientnet_unet.py`** (lines 18-35) is a well-targeted, minimal fix for the off-by-one skip/upsample mismatch caused by EfficientNet's floor-rounded stride-2 convolutions; no change recommended here, noting it only because it is the one architecture-specific engineering workaround in the codebase and would need re-verifying if the backbone is ever swapped (different backbones round dimensions differently or not at all).
- **Bottleneck channel counts are inconsistent across from-scratch architectures vs. EfficientNet UNet**: `unet.py`/`resunet.py`/`attention_unet.py` all default to a 1024-channel bottleneck (e.g. `unet.py:68`) fed by a `[64,128,256,512]` encoder, while `efficientnet_unet.py`'s decoder starts at 256 channels (`efficientnet_unet.py:97`) because the frozen EfficientNetB0 backbone already outputs 1280 channels at the bottleneck. This isn't a bug (the two are genuinely different feature scales) but makes head-to-head parameter-count/capacity comparisons between the four architectures less apples-to-apples than the model-comparison notebook (`030_evaluation.ipynb`) might assume.

## 6. Possible additions for robustness

- **Regional delta analysis** (extends the design discussion already captured in `note.md`): the current `plot_delta` (`visualization.py:82`) is a raw `|real − predicted|` map, which conflates genuine hidden-detail signal with substrate/acquisition-driven gray-level shifts. Implementing the luminance/contrast/structure decomposition and per-window normalization discussed in `note.md` as a `scripts/delta_analysis.py` module would turn this from a qualitative visualization into a more defensible analytical output — directly relevant to the project's stated goal of surfacing underdrawings/pentimenti rather than acquisition artifacts.
- **Fail-fast data validation**: beyond the RGB/IR dimension check (§3.1), `grouped_train_val_test_split` has no guard against a fold ending up empty (e.g. too few distinct artworks for the requested `train_ratio`/`val_ratio`). A dataset with very few artworks would silently produce a degenerate split rather than a clear error.
- **Structural test for `efficientnet_unet`**: the EfficientNet UNet builder is excluded from unit tests because it downloads ImageNet weights. Instantiating it with `tf.keras.applications.EfficientNetB0(weights=None, ...)` would let a test verify the decoder wiring, skip-connection shapes, and `_ResizeToMatch` behaviour without any network dependency or slow download — currently that whole architecture has zero automated coverage of its own assembly logic.
- **Checkpoint/version metadata**: `get_callbacks`/`load_model` save/load `best_model.keras` with no accompanying metadata (git commit, config snapshot, training data split hash). For a research pipeline comparing four architectures over time, recording this alongside the checkpoint would make results reproducible and comparable months later.
- **Graceful handling of corrupt/unreadable images**: `_load_pair` assumes every file decodes cleanly; a single corrupt JPEG in `data/rgb` or `data/ir` will crash the whole `tf.data` pipeline mid-epoch rather than being skipped with a warning.


## 7. Alternative architectures

The four original architectures (`unet.py`, `resunet.py`, `attention_unet.py`,
`efficientnet_unet.py`) are all supervised, deterministic, purely
convolutional encoder–decoders — no adversarial term, no diffusion process,
no self-attention/transformer block. This was well-suited to a small,
artwork-grouped dataset, but the following directions are worth considering
if the goal is to push prediction quality further, roughly ordered from
"most compatible with a small dataset" to "most data-hungry." §7.6 and §7.7
(heteroscedastic head, architectural ablations) and §7.3 (bottleneck
self-attention) have since been implemented.

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

### 7.3. Efficient transformer block at the bottleneck (Restormer) — IMPLEMENTED

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

**Implementation**: `scripts/unet_restormer.py` (`build_unet_restormer`,
`RestormerBlock`) — a single Restormer transformer block inserted right
after the existing bottleneck `_conv_block`, otherwise identical to
`unet.py`. `RestormerBlock` implements the two Restormer sub-blocks with
pre-norm residual connections:

- **MDTA** (Multi-Dconv Head Transposed Attention): 1x1 conv + depthwise
  3x3 conv produce `(q, k, v)`; attention is a `(head_dim, head_dim)` map
  per head computed across channels rather than spatial positions, cost
  `O(H*W*head_dim^2)` — linear in image size, as opposed to the `O((HW)^2)`
  cost of ordinary spatial self-attention.
- **GDFN** (Gated-Dconv Feed-Forward Network): 1x1 conv expansion,
  depthwise 3x3 conv, then `GELU(x1) * x2` gating on the split expanded
  channels before projecting back down.

Registered as a new, independent architecture (`"unet_restormer"`) in
`scripts/trainer.py`'s `_BUILDERS`/`compile_model` — checkpoints save to
`models/unet_restormer/`, never colliding with `unet`/`unet_v2`. Directly
answers the gap noted in §5 (`attention_unet.py`'s gate has no long-range
receptive field): this block gives the network a genuinely global
receptive field, but only at the bottleneck, keeping the cost low.

Covered by `tests/unit/test_unet_restormer.py`: `RestormerBlock` presence
at the bottleneck, `dim`/`num_heads` divisibility validation, output
shape/range on both square and non-square inputs (exercises the dynamic
`tf.shape`-based reshape inside the attention computation), a save/load
round-trip regression test (`RestormerBlock` is
`register_keras_serializable`-decorated with a `get_config` override —
fourth instance of this bug class in the project, after `ClipLogVar`,
`_ResizeToMatch`, and `_Upsample2x`), and standalone `RestormerBlock`
shape/finiteness checks. `unet_restormer` also added to the generic
architecture builder tests (`tests/unit/test_models.py`). Build, compile,
and a real-data smoke fit/save/load/predict round trip verified in temp
`models`/`logs` dirs (including a non-square `64x96` prediction). Not yet
trained end-to-end — `022_training_v2.ipynb`/`032_evaluation_v2.ipynb`/
`062_model_comparison_v2.ipynb` train/evaluate/compare it alongside
`unet_v2` and `unet`.

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

### 7.6. Heteroscedastic aleatoric uncertainty head (learned per-pixel confidence) — IMPLEMENTED FOR ALL FOUR ARCHITECTURES

Implemented on `feature/heteroscedastic-nll`, kept entirely separate from
the four existing deterministic architectures (no existing file's
training behaviour changes). Started as a pilot on `attention_unet` only
(`scripts/attention_unet_nll.py`, `build_attention_unet_nll`), then
extended to the other three: `scripts/unet_nll.py`
(`build_unet_nll`), `scripts/resunet_nll.py` (`build_resunet_nll`),
`scripts/efficientnet_unet_nll.py` (`build_efficientnet_unet_nll`; reuses
`efficientnet_unet.py`'s pretrained encoder, skip connections, and
`_ResizeToMatch` layer directly rather than duplicating them, to avoid
registering a second Keras-serializable class under the same
package/name). The `ClipLogVar` layer used by all four is factored into
`scripts/nll_layers.py` for the same reason. Shared infrastructure:
`scripts/losses.py` (`gaussian_nll_loss`, additive), `scripts/metrics.py`
(`MuMAEMetric`, `MuSSIMMetric`, `MuPSNRMetric`, additive),
`scripts/trainer_nll.py` (`_BUILDERS_NLL` registry, all four
architectures), `scripts/inference_utils_nll.py`,
`scripts/visualization_nll.py` (`plot_predictions_nll`, `plot_zscore`,
`plot_delta_comparison`), and three notebooks mirroring the existing
pipeline — `021_training_nll.ipynb`, `031_evaluation_nll.ipynb`,
`051_delta_analysis_nll.ipynb` — all three now loop over (or, for `051`,
are selectable across) all four `*_nll` architectures. Checkpoints land
in `models/<arch>_nll/`, never colliding with the deterministic
architectures.

One real bug surfaced and was fixed during the pilot: the `log_var`
clip was originally a `Lambda` layer, which Keras refuses to
deserialize by default (`Lambda` wrapping a Python function is treated
as an arbitrary-code-execution risk) — this silently broke
`tf.keras.models.load_model` on any saved checkpoint, only surfacing
when `031_evaluation_nll.ipynb` tried to reload the trained model.
Fixed by replacing it with `ClipLogVar`, a small named/serializable
`keras.layers.Layer` subclass; a save/load round-trip regression test
was added (`tests/unit/test_nll.py`) to catch this class of bug in the
future.

`attention_unet_nll` has a real full training + evaluation run (22
epochs, best val_loss -1.4852; test `mae 0.1066, psnr 19.1175, ssim
0.5387` vs. deterministic `attention_unet`'s `mae 0.1124, ssim 0.5894,
psnr 18.64` — mixed result, `mae`/`psnr` improve but `ssim` regresses) and
a qualitative run of `051_delta_analysis_nll.ipynb` on a real painting
(`green`), whose `plot_delta_comparison` panel showed the learned z-score
tracking structural detail comparably to the structural delta, while the
fixed-window normalized delta looked comparatively flat on that image.
`unet_nll`, `resunet_nll`, and `efficientnet_unet_nll` are new (build,
compile, and a real-data smoke fit/save/load round-trip all verified in
this session) but not yet trained end-to-end — running
`021_training_nll.ipynb` with all four in `ARCHS` and then
`031_evaluation_nll.ipynb` is the next step. The overall keep/drop
decision for this direction is still open.

Motivation: the RGB→IR mapping is inherently one-to-many — pigments that
look alike in visible light can have markedly different IR reflectance, so
a single deterministic prediction per pixel forces the network to average
over genuinely ambiguous cases. This direction replaces the current
single-channel, deterministic output (`Conv2D(1, 1, activation="sigmoid")`,
identical across all four architectures) with a two-channel head predicting
both a mean `μ` (the expected IR value, same role as today's output) and a
log-variance `log σ²` (the network's own learned estimate of how ambiguous
that RGB context has historically been). Trained with a Gaussian
negative-log-likelihood loss instead of `combined_loss`/
`combined_loss_advanced`:

```
L = 0.5 * exp(-log_var) * (y_true - mu) ** 2 + 0.5 * log_var
```

Architecturally this is minimally invasive — only the final `1×1` conv of
each of the four encoder–decoders changes (an extra parallel `Conv2D(1, 1)`
branch for `log_var`, concatenated with the existing `mu` branch); no
change to the shared encoder/decoder trunks. `SSIMMetric`/`PSNRMetric`
(`scripts/metrics.py`) would need to slice out the `mu` channel before
comparing against `y_true`, since only `mu` is an image-shaped prediction.

Payoff for this project specifically: at inference the learned `σ` gives a
principled, color/context-conditioned normalization for the delta signal —
`z = (real_IR - mu) / sigma` — as an alternative or complement to the fixed
Gaussian-window normalization currently used in `delta_analysis.py`. A real
IR pixel within a small `z` is unremarkable for that pigment even if the
raw delta is large (that color is known to be variable); a pixel at large
`z` is a genuine anomaly candidate (underdrawing/pentimento) even where the
raw delta is modest. Crucially, `σ` is learned from RGB context alone and
never conditioned on the real IR value at inference time, so it cannot
"explain away" real anomalies the way a ground-truth-conditioned
hypothesis-selection scheme would.

Known training pitfall: naively minimizing the Gaussian NLL from scratch is
prone to instability — the network can trivially lower the loss early on by
inflating `σ` everywhere instead of learning an accurate `μ` (a form of
gradient starvation on the mean branch). The standard mitigations are a
warm-start (train `mu` alone with the current deterministic loss for the
first epochs, before enabling the variance branch/NLL) and/or a modified
loss that down-weights the variance term early in training.

- D.A. Nix, A.S. Weigend, *"Estimating the mean and variance of the target
  probability distribution,"* IEEE ICNN, 1994.
  https://doi.org/10.1109/ICNN.1994.374138 — original formulation of a
  neural network predicting both mean and variance of its target, trained
  with Gaussian NLL.
- A. Kendall, Y. Gal, *"What Uncertainties Do We Need in Bayesian Deep
  Learning for Computer Vision?,"* NeurIPS, 2017.
  https://doi.org/10.48550/arXiv.1703.04977 — the modern formulation used
  here (log-variance parameterization for numerical stability, aleatoric
  vs. epistemic uncertainty), standard reference for heteroscedastic heads
  in vision regression tasks.
- M. Seitzer, A. Tavakoli, D. Antic, G. Martius, *"On the Pitfalls of
  Heteroscedastic Uncertainty Estimation with Probabilistic Neural
  Networks,"* ICLR, 2022. https://doi.org/10.48550/arXiv.2203.09168 —
  documents the variance-inflation/gradient-starvation training instability
  described above and proposes a corrective (β-NLL) loss weighting;
  directly relevant if the plain NLL proves hard to train on this dataset.

**Update**: `scripts.losses.beta_gaussian_nll_loss` (the β-NLL correction
above) is implemented and selectable via `scripts.trainer_nll.NLL_LOSSES`
(`compile_model_nll`/`load_model_nll`'s `loss_name`/`beta` parameters).
`022_training_v2.ipynb` trains all four `*_nll` architectures with
`beta_nll` (`beta=0.5`), saved to a separate checkpoint tree
(`models/nll_beta/<arch>/`) so the existing `gaussian_nll` checkpoints from
`021_training_nll.ipynb` are preserved for comparison.

**Real run result (`032_evaluation_v2.ipynb`)**: `beta_nll` does **not**
close the `gaussian_nll` regression — it widens it, on `mae`/`psnr`/`ssim`,
on every one of the four architectures (e.g. `resunet_nll`: ssim
0.4669→0.3577, psnr 18.63→14.99; smallest regression on
`efficientnet_unet_nll`, ssim 0.5251→0.4150). See handoff for the full
per-architecture table.

**Evaluation methodology gap surfaced by this result**: `mae`/`ssim`/`psnr`
(`MuMAEMetric`/`MuSSIMMetric`/`MuPSNRMetric`, `scripts/metrics.py`) read
only the `mu` channel — by construction they cannot see whether `sigma` is
any good, which is the entire point of this head. Qualitative inspection
of `062_model_comparison_v2.ipynb` on `modern` (a real test pair built
ad hoc with specific known hidden details) suggests the opposite of the
quantitative result: the `beta_nll` z-score picks out most of the intended
details better than `gaussian_nll`'s, just under-contrasted. The two
signals (`mu` fidelity vs. `sigma`/z-score quality) are not the same thing
and can move in opposite directions — β-NLL is explicitly a mu/sigma
gradient trade-off (Seitzer et al. 2022), so this is plausible, not a bug.
Three follow-ups identified (1 and 3 now implemented — see §7.8):

1. **A calibration metric for `sigma`** — e.g. coverage probability or the
   correlation between `|real_IR - mu|` and predicted `sigma` — to get a
   numeric score for comparing NLL variants that doesn't route through
   `mu` alone. Priority: gives a real number to decide `beta` and to
   compare `gaussian_nll` vs. `beta_nll` (and future variants) on the
   uncertainty signal itself.
2. **Ground-truth-mask detection metric on `modern`** — `modern` is a
   single, purpose-built image with known hidden-detail regions; annotate
   a mask of those regions once and score every candidate signal (raw
   delta, structural delta, z-score, per variant) against it with
   AUROC/precision-recall. Most direct measure of "does this reveal what
   it's supposed to reveal," feasible precisely because it's one image.
3. **Parametric contrast control for the z-score plots** — `plot_zscore`/
   `plot_delta_comparison`/`plot_signal_comparison`'s fixed `Z_VMAX=4.0`
   clip may be flattening a real signal. Add a percentile- or
   gamma-based contrast parameter, computed after the z-score maps exist,
   so different "contrast levels" can be regenerated and compared without
   re-predicting.

### 7.7. `unet_v2`: encoder/decoder downsampling/upsampling and regularization variants — IMPLEMENTED

`scripts/unet_v2.py` (`build_unet_v2`) mirrors `unet.py` exactly with all
flags at their default (off), and exposes three modifications discussed
for this architecture as independent, ablatable parameters rather than a
single fused variant — so each can be attributed separately once trained:

- `use_strided_conv`: replaces `MaxPool2D` with a learned stride-2
  `Conv2D` for encoder downsampling. Motivation: `MaxPool2D` is a fixed,
  non-learned operator; letting the network learn what to keep when
  compressing may suit a domain-specific, subtle signal (underdrawing
  strokes) better than a generic max response.
- `use_upsample_conv`: replaces `Conv2DTranspose` with bilinear
  `UpSampling2D` + `Conv2D` for decoder upsampling, to avoid the
  checkerboard artifacts transposed convolution is prone to (Odena et al.,
  *"Deconvolution and Checkerboard Artifacts,"* Distill, 2016). Implemented
  as a custom `_Upsample2x` layer rather than `layers.UpSampling2D`
  directly — `UpSampling2D` inspects the *static* input shape and raises
  on this model's fully dynamic `(None, None, 3)` input; `_Upsample2x`
  resizes from the *runtime* shape via `tf.image.resize`, the same fix
  already used by `_ResizeToMatch` in `efficientnet_unet.py` for the same
  underlying reason. `register_keras_serializable`-decorated, so
  `tf.keras.models.load_model` can reconstruct it without
  `custom_objects` (`ClipLogVar`/`_ResizeToMatch` are the two prior
  instances of this bug class — see §7.6's "Known training pitfall" note
  and `nll_layers.py`); a save/load round-trip regression test
  (`tests/unit/test_unet_v2.py`) guards it going forward.
- `dropout_rate`: `SpatialDropout2D` applied at the bottleneck and the
  first (deepest) decoder block only, not every block. Applying it
  throughout would compound with the `BatchNormalization` already present
  in every conv block — dropout noise present in training but absent at
  inference shifts the statistics BN relies on (Li et al., *"Understanding
  the Disharmony between Dropout and Batch Normalization,"* 2019).
  Restricting it to the two deepest blocks concentrates regularization
  where overfitting risk is highest (large bottleneck relative to a
  painting-scale dataset — §4) while leaving the shallow encoder levels,
  where fine underdrawing-relevant detail lives, untouched.

Registered in `scripts/trainer.py`'s `_BUILDERS`/`compile_model` as a new,
independent architecture name (`"unet_v2"`) — checkpoints save to
`models/unet_v2/`, never colliding with `unet`. Covered by
`tests/unit/test_unet_v2.py` (each flag's effect on layer composition in
isolation, combined-flags output shape/range, save/load round trip) and
included in the generic architecture builder tests
(`tests/unit/test_models.py`). Build, compile, and a real-data smoke
fit/save/load round trip were verified in this session (temp
`models`/`logs` dirs); not yet trained end-to-end —
`022_training_v2.ipynb` trains it with all three modifications enabled
(`use_strided_conv=True, use_upsample_conv=True, dropout_rate=0.2`),
`032_evaluation_v2.ipynb`/`062_model_comparison_v2.ipynb` compare it
against `unet`.

### 7.8. Calibration metrics for `sigma` and parametric z-score contrast — IMPLEMENTED

Follow-ups 1 and 3 of §7.6, implemented as two new post-hoc modules. Both
operate on the `(mu, sigma)` arrays inference already produces, so every
existing checkpoint is scoreable and re-renderable without retraining and
without touching any architecture.

**`scripts/calibration.py`** — scores `sigma` on its own, closing the
"`mae`/`ssim`/`psnr` only ever see `mu`" gap:

| Metric | Reads | Calibrated value |
|---|---|---|
| `coverage_probability` | fraction of pixels inside `+/- k*sigma` | nominal `erf(k/sqrt2)`: `0.683`/`0.954`/`0.997` |
| `sigma_reliability` / `ence` | predicted vs. observed error per equal-population `sigma` bin (Levi et al.) | `0` |
| `z_std` | spread of `(real - mu) / sigma` | `1` (`>1` overconfident, `<1` under) |
| `error_sigma_correlation` | Spearman of `abs(real - mu)` vs. `sigma` | high = knows where it errs |
| `mean_gaussian_nll` | proper scoring rule, in nats | lower is better |
| `sharpness` / `dispersion` | mean `sigma`, and its coefficient of variation | see below |

The last row is not decoration. Calibration alone is trivially gamed by a
large constant `sigma`, which would score perfectly while collapsing the
learned z-score into a rescaled raw delta — `dispersion ~ 0` is therefore
a *disqualifier*, not a neutral statistic, and calibration and sharpness
have to be read as a pair (Gneiting et al. 2007). `mean_gaussian_nll`
keeps the `0.5*log(2*pi)` constant that `losses.gaussian_nll_loss` drops
(irrelevant to gradients, necessary for a comparable likelihood), and is
the natural tie-breaker when the `mu`-only metrics and the calibration
metrics disagree — precisely the `gaussian_nll` vs. `beta_nll` situation.
`evaluate_calibration(...).summary()` yields one flat row per model;
`032_evaluation_v2.ipynb` §4b tabulates all four architectures x both loss
variants, and `visualization_nll.plot_calibration` draws the reliability
diagram, coverage bars and z-histogram against `N(0, 1)`.

**`scripts/contrast.py`** — `ZScale` turns the hard-coded `Z_VMAX=4.0`
into a parameter: `FIXED` (the previous behaviour, still the default, and
the only mode comparable across images since it ignores the data) or
`PERCENTILE` of `|z|`, plus an optional `gamma` compression of the ramp
that expands faint detail at `|z| ~ 1`. Everything applies *after* the
z-score maps exist, so contrast variants cost a re-plot, not a
re-prediction. `ZScale.apply_many` computes one limit shared across
several models and returns it together with the scaled maps — the
cross-architecture figures in `060`/`062` were previously sharing a
hard-coded `vrange` by hand, which silently breaks the moment anyone
changes it in one cell and not another. `plot_zscore` and
`plot_delta_comparison` took a `z_scale` argument in place of
`vmax`/`z_vmax`; `062_model_comparison_v2.ipynb` §2c sweeps
fixed / p99.5 / p99.5+gamma 0.5 over the four `beta_nll` architectures.

`calibration.learned_zscore` is now the single definition of
`(real - mu) / sigma`, replacing the inline `(real - mu) / (sigma + 1e-8)`
that had been copied across `visualization_nll.py` and two notebooks, so
the signal being scored and the signal being displayed cannot drift apart.

Covered by `tests/unit/test_calibration.py` and
`tests/unit/test_contrast.py` (100% line coverage on both modules), which
verify the metrics against synthetically calibrated, overconfident and
underconfident predictions rather than against fixed expected numbers.

**What this does not settle**: these metrics say whether `sigma` is
*trustworthy*, not whether the z-score *reveals underdrawings*. Follow-up
2 of §7.6 (ground-truth mask on `modern`) remains the only thing that
answers the latter, and remains the gate on `gaussian_nll` vs. `beta_nll`.

- D. Levi, L. Gispan, N. Giladi, E. Fetaya, *"Evaluating and Calibrating
  Uncertainty Prediction in Regression Tasks,"* Sensors, 2022.
  https://doi.org/10.48550/arXiv.1905.11659 — ENCE and the binned
  reliability diagram for regression uncertainty.
- T. Gneiting, F. Balabdaoui, A.E. Raftery, *"Probabilistic forecasts,
  calibration and sharpness,"* JRSS-B, 2007.
  https://doi.org/10.1111/j.1467-9868.2007.00587.x — the principle that
  calibration must be maximised *subject to* sharpness, which is why
  `dispersion` is reported next to the calibration scores.

