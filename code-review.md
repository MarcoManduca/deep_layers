# Code review — structure and engineering choices

The final state of `scripts/` and `tests/unit/`: what each module does, the
non-obvious implementation decisions and why they were made, and the limitations
that remain. The theoretical justification for the modelling choices is in
[theory-links.md](theory-links.md); the quantitative results are in
[final-comments.md](final-comments.md).

---

## 1. Code structure

```
scripts/
├── config.py                 pydantic Settings singleton — every path and hyperparameter
├── reproducibility.py        global RNG seeding
├── dataset.py                pair discovery, artwork-and-mockups split, tf.data pipeline, padding
├── augmentation.py           stateless paired RGB/IR augmentation
├── kfold.py                  grouped k-fold split, composed on top of dataset.py's grouping
├── norm_utils.py             GroupNorm group-count helper; the _ReLUFix layer
│
├── unet.py                   baseline U-Net
├── resunet.py                U-Net with residual blocks
├── attention_unet.py         U-Net with additive attention gates on the skips
├── unet_residual.py          unet backbone, residual output head (mean(R,G,B) + signed delta)
├── unet_v2.py                unet with strided-conv / resize-conv / bottleneck-dropout toggles
├── unet_restormer.py         unet with a Restormer self-attention block at the bottleneck
├── unet_dilated.py           unet with a dilated (ASPP-style) bottleneck
├── unet_v2_dilated.py        unet_v2 with the same dilated bottleneck
├── efficientnet_unet.py      pretrained EfficientNetB0 encoder + U-Net decoder
├── aspp.py                   the parallel-dilation bottleneck block
├── residual_head.py          RGBToGray, ClipToUnitStraightThrough — the unet_residual head layers
│
├── unet_nll.py resunet_nll.py attention_unet_nll.py efficientnet_unet_nll.py
│                             heteroscedastic (mu, log_b) counterparts of the four base architectures
├── nll_layers.py             ClipLogVar — shared serializable clip layer for the *_nll builders
│
├── losses.py                 combined_loss, laplace_nll_loss (+ unused advanced/gaussian variants)
├── metrics.py                PSNR/SSIM Keras metrics; mu-only variants for the NLL models
├── trainer.py                model factory, compile, callbacks, checkpoint load — deterministic
├── trainer_nll.py            same, for the heteroscedastic models
├── train_single.py           CLI: train exactly one architecture in its own process, then exit
│
├── delta_analysis.py         local SSIM luminance/contrast/structure decomposition; raw/structural delta
├── calibration.py            sigma calibration metrics; learned and structural z-score
├── contrast.py               display-contrast control (ZScale) for z-score maps
├── detection.py              AUROC / average precision of a signal against a mask
├── stroke_stats.py           reference-free structure-tensor coherence
│
└── visualization.py visualization_nll.py   plotting helpers

tests/unit/                   one module per script, pure-logic focused
```

Module boundaries map one-to-one onto pipeline stages, duplicated once for the
deterministic / heteroscedastic split where the two genuinely differ (builders,
trainer, metrics, visualisation) and shared where they do not (`dataset.py`,
`delta_analysis.py`, `detection.py`, `stroke_stats.py`, `contrast.py`).
`trainer.py` / `trainer_nll.py` are the single integration points wiring an
architecture to its loss; both training and checkpoint reload derive the loss the
same way, so they cannot silently diverge. Docstrings are NumPy-style throughout;
no circular imports.

---

## 2. What each module does

**`config.py`** — one `Settings` object (pydantic-settings, `.env`-overridable):
patch multiple, batch size, learning rate, epochs, seed, split ratios, mockup
artwork IDs, loss weights (`LOSS_ALPHA`, `NLL_BETA`, `CHARBONNIER_EPS`),
`WEIGHT_DECAY`, `GRADIENT_CLIP_VALUE`, all callback parameters, dilation rates,
crop size, k-fold parameters, fine-tuning learning rate/epochs. Every other module
imports `settings` rather than hardcoding.

**`reproducibility.py`** — `set_global_seed()` seeds Python / NumPy / TensorFlow
via `tf.keras.utils.set_random_seed`. Deliberately does *not* call
`tf.config.experimental.enable_op_determinism()` (unsupported and slow on the
Metal backend); per-sample augmentation determinism is handled by a stateless-RNG
mechanism instead.

**`dataset.py`** — discovers RGB/IR pairs by matching filename stems, extracts an
artwork ID from each name. Two split functions: `grouped_train_val_test_split`
(every artwork ID is an indivisible group — the leakage-free primitive) and
`mockup_aware_train_val_test_split` (real artworks still grouped; the synthetic
mockup groups in `settings.MOCKUP_ARTWORK_IDS` are split at the pair level, most to
train, `MOCKUP_TEST_RATIO` to test — since those groups exist to be learnt from,
not generalised to). The mockup-aware split is the one every notebook uses.
`pad_to_multiple` pads a single image to the next multiple of N for whole-image
inference. `build_dataset` assembles the `tf.data` pipeline: load → shuffle →
decode/normalise → optional seeded augmentation → batch → prefetch.

**`augmentation.py`** — `augment_pair`: shared random crop (RGB+IR concatenated on
the channel axis so they share one box), then identical flips on both, then
brightness/contrast jitter on RGB only. All randomness is stateless and
seed-derived.

**`kfold.py`** — `grouped_kfold_splits` / `fold_split` / `fold_artwork_groups`:
partitions the real artworks into k groups by ID (no section leakage), holds one
group out per fold, keeps all mockups in every training set. `data/test/` is
external to every fold. Deterministic in `(KFOLD_K, KFOLD_SEED)`.

**`norm_utils.py`** — `num_groups(filters)` returns the largest divisor of the
channel count ≤ 32, so `GroupNormalization` works at any width. `_ReLUFix` / `relu`
— see §3.3.

**The deterministic builders** — `unet` / `resunet` / `attention_unet` /
`unet_residual` / `unet_v2` / `unet_restormer` / `unet_dilated` / `unet_v2_dilated`
/ `efficientnet_unet`: fully-convolutional encoder–decoders, `(None, None, 3)` in,
`(None, None, 1)` sigmoid out. Notable pieces:

- `efficientnet_unet._ResizeToMatch` reconciles the off-by-one spatial mismatch
  between EfficientNetB0's floor-rounded stride-2 downsampling and the decoder's
  exact-doubling upsampling.
- `unet_v2._Upsample2x` is a custom bilinear-resize layer used because
  `layers.UpSampling2D` inspects the *static* shape and fails on a fully dynamic
  input.
- `unet_residual` (`residual_head.py`): `RGBToGray` is a channel mean;
  `ClipToUnitStraightThrough` is a clip to `[0,1]` implemented as
  `x + stop_gradient(clip(x) − x)` — exact forward, transparent backward, so a
  pixel pushed out of range still receives a gradient instead of dying.
- `aspp.dilated_bottleneck` replaces the plain bottleneck with parallel
  `Conv2D(dilation_rate=r)` branches at `settings.DILATION_RATES`, concatenated and
  projected back — a wider receptive field without extra downsampling.

**The heteroscedastic builders** — `unet_nll` etc.: a 2-channel `(mu, log_b)`
output instead of one. `efficientnet_unet_nll` reuses `efficientnet_unet`'s
encoder/skips/`_ResizeToMatch` directly. `nll_layers.ClipLogVar` clips the
log-scale channel into `[NLL_LOG_VAR_MIN, NLL_LOG_VAR_MAX]` and is shared by all
four so only one class registers under that name for `load_model`.

**`losses.py`** — `combined_loss(alpha)` = `alpha·Charbonnier + (1−alpha)·(1−MS-SSIM)`
for every deterministic architecture (`alpha = 0.16` default). `laplace_nll_loss(beta)`
= `stop_gradient(b)^beta · (|y−mu|/b + log b)` for every heteroscedastic
architecture (`beta = 0.5` default). `combined_loss_advanced` (MAE + Laplacian
pyramid + FFT) and the Gaussian NLL variants remain in the file but are on no
architecture's default path — kept only so old checkpoints can still be recompiled.

**`metrics.py`** — `PSNRMetric` / `SSIMMetric` for the deterministic models;
`MuMAEMetric` / `MuSSIMMetric` / `MuPSNRMetric` read only the `mu` channel so the
heteroscedastic models stay directly comparable on the same numbers.

**`trainer.py` / `trainer_nll.py`** — `get_model` / `get_model_nll` (name → builder
factory), `compile_model` / `compile_model_nll` (Adam with `weight_decay` and
`clipvalue`, the matching loss), `get_callbacks` (Checkpoint / EarlyStopping /
ReduceLROnPlateau / TensorBoard, all monitoring `val_loss`), `load_model` /
`load_model_nll` (reload with `compile=False`, then recompile with the same loss).
`get_callbacks` is generic and shared unchanged between the two.

**`train_single.py`** — a CLI that builds datasets from `settings`, trains one
architecture, and writes both the `.keras` checkpoint and a `history.json` (a
subprocess cannot return the `History.history` dict in memory). Flags:
`--loss-alpha` / `--nll-beta` (sweep a weight without touching ambient config),
`--init-from` (warm-start weights, for two-phase EfficientNet fine-tuning),
`--fold` / `--kfold-k` (switch to the k-fold split). See §3.4 for why training goes
through this rather than a loop.

**`delta_analysis.py`** — `compute_local_stats` / `compute_ssim_components`
decompose local SSIM into luminance / contrast / structure maps over an 11×11
Gaussian window (the `tf.image.ssim` window; `gaussian_local_filter` is public so
`calibration.py` can smooth σ with exactly the same kernel). `analyze_delta`
returns `raw_delta`, `structural_delta` (`1 − structure`), and a `confidence_map`
(agreement between the two after min-max normalisation).

**`calibration.py`** — `learned_zscore` = `(real − mu)/σ`, `structural_zscore` =
`structural_delta / smoothed σ`, and `laplace_sigma_from_scale` = `b·√2` (the
Laplace standard deviation — *not* the Gaussian `exp(0.5·log_var)`).
`evaluate_calibration` scores whether σ itself can be trusted: coverage
probability, ENCE reliability, error/σ Spearman correlation, z-score moments, mean
Laplace NLL, and — deliberately alongside — sharpness and dispersion, since a large
*constant* σ would score as calibrated while being useless.

**`contrast.py`** — `ZScale` turns a z-score map's display limit into a parameter
(fixed or percentile, optional gamma), so a contrast variant costs a re-plot, not
a re-prediction. `apply_many` derives one shared limit across several maps for an
honest cross-model figure. Note that display contrast cannot change a rank-based
metric like AUROC — it is a rendering choice only.

**`detection.py`** — `evaluate_detection` / `rank_signals`: AUROC (primary,
prevalence-independent) and average precision (prevalence reported alongside,
since that is AP's chance level) of any magnitude signal against any mask.

**`stroke_stats.py`** — `stroke_coherence`: structure-tensor coherence, no mask
needed. A straight stroke scores ≈ 0.94, isotropic noise ≈ 0.08; scale-invariant,
so a raw delta and a z-score are directly comparable.

**`visualization*.py`** — sample grids, triptychs, delta/z-score galleries,
training-curve plots, calibration diagrams. `DEFAULT_Z_SCALE` (percentile p99.5,
no gamma) is the project's chosen contrast setting.

---

## 3. Engineering choices

### 3.1 GroupNormalization instead of BatchNormalization

Training runs at `BATCH_SIZE = 8` with ~18 normalisation layers per network —
well below the regime where BatchNorm's batch statistics are stable.
`GroupNormalization` is batch-size-independent by construction and is used
everywhere in the from-scratch architectures. The one exception is the pretrained
EfficientNetB0 encoder inside `efficientnet_unet`: its internal BatchNorm layers
carry the ImageNet running statistics and are left exactly as shipped — only the
decoder is converted to GroupNorm. `num_groups()` adapts the group count down from
32 for the small channel widths the unit tests exercise; every real width (64–1024)
uses 32 unmodified.

### 3.2 He initialisation

Xavier / Glorot initialisation is derived under a `tanh` assumption and
under-scales the variance for ReLU. Every ReLU-preceding convolution sets
`kernel_initializer="he_normal"`.

### 3.3 The `_ReLUFix` layer — a tensorflow-metal work-around

On Apple Silicon (confirmed on M4 Max, `tensorflow-macos 2.16.2`,
`tensorflow-metal 1.2.0`) the GPU-compiled, graph-mode ReLU kernel **fails to clip
negative values** — negative activations leak through. Over the depth of these
networks (8–9 blocks, each ending in ReLU) the leaked negatives accumulate,
amplified faster by additive paths (`resunet`'s residual add, `attention_unet`'s
gate) than by plain concatenation, until the weights go `NaN` — within the first
epoch for the residual/attention models, after a few epochs for plain `unet`.
Eager execution never shows this; only the default graph-compiled `model.fit()` on
GPU does. It is a known, open, unresolved Apple bug
(<https://developer.apple.com/forums/thread/818015>) with no upstream fix.

`norm_utils.relu()` replaces every `layers.ReLU()` / `activation="relu"` with
`tf.maximum(x, 0.0)` wrapped in `_ReLUFix`, a registered `Layer` subclass. It
must be a real layer, not a `Lambda`, because Keras 3 refuses to deserialize a
`Lambda`'s Python function from a checkpoint by default. The fix is a pure
activation swap — identical parameter counts, and it makes the graph-mode GPU
numbers match the eager ones exactly.

### 3.4 One subprocess per architecture

`tensorflow-metal` does not fully release GPU memory back to the OS between
sequential `model.fit()` calls in one process, even with
`tf.keras.backend.clear_session()`: the *first* architecture trained in a process
is always clean, every subsequent one increasingly risks a `NaN` from epoch 1,
regardless of order. `train_single.py` is therefore invoked once per architecture
as a subprocess (`subprocess.run(..., check=True)`); process exit is the only
reliable way to get a fresh GPU context. Runs are still sequential, not parallel.
Each subprocess rebuilds its own datasets deterministically from `settings`, so
nothing has to cross the process boundary except the checkpoint and a
`history.json` on disk.

### 3.5 Per-element gradient clipping

`tf.image.ssim_multiscale`'s backward pass computes `d/dx[x^p] = p·x^(p-1)` at the
standard MS-SSIM power factors; where TF clamps a per-scale structural term to
exactly zero internally, this evaluates at `x = 0` and the gradient hits `+Inf` —
a real singularity in TF's own implementation, more likely to fire for some
activation statistics than others. `compile_model` / `compile_model_nll` pass
`Adam(clipvalue=settings.GRADIENT_CLIP_VALUE)` (default `1.0`). It must be
`clipvalue`, not `clipnorm`: `clipnorm` turns an already-`Inf` entry into `NaN`
via `Inf/Inf` when computing the global norm, whereas `clipvalue` clips each entry
directly so `Inf` becomes a large finite number. Reverting to single-scale SSIM
would remove the singularity structurally but throws away the reason MS-SSIM was
adopted.

### 3.6 The custom-serializable-layer pattern

`_ReLUFix` (`norm_utils.py`), `_ResizeToMatch` (`efficientnet_unet.py`),
`_Upsample2x` (`unet_v2.py`), `ClipLogVar` (`nll_layers.py`), and the two
`residual_head.py` layers are all the same shape: a tiny `Layer` subclass
decorated with `@register_keras_serializable(package="deep_layers")`, needed
because Keras cannot otherwise reconstruct that operation from a saved checkpoint.
Shared classes (`ClipLogVar`) are imported, not copied, so exactly one class
registers under each name.

### 3.7 Checkpoint / Keras version coupling

Checkpoints are saved by Keras 3.15 and do not load under a different minor
version (the failure is an opaque initializer-argument error deep in
`Functional.from_config`, not a missing-file error). `requirements.txt` and
`env/environment.yml` both pin `keras==3.15.1`; a Keras upgrade requires re-saving
or retraining every checkpoint.

---

## 4. Architecture-specific notes

- **`attention_unet`'s "attention" is CNN spatial gating, not self-attention.**
  Oktay et al.'s additive gate reweights an encoder skip using the decoder's
  gating signal; it has no global receptive field. `unet_restormer` adds the only
  genuine self-attention in the project — on `unet`, not `attention_unet`, so the
  two mechanisms have not been combined.
- **`_ResizeToMatch` is specific to EfficientNetB0's rounding.** A different
  backbone would round dimensions differently and the layer would need
  re-verifying.
- **Bottleneck widths are not comparable across families.**
  `unet` / `resunet` / `attention_unet` use a 1024-channel bottleneck fed by a
  `[64,128,256,512]` encoder; `efficientnet_unet`'s decoder starts at 256 because
  the frozen backbone already outputs 1280. This is a genuine feature-scale
  difference, not a bug, but it means a raw parameter-count comparison across
  families is not apples-to-apples.
- **`unet_residual`'s grey prior is weak by design.** `mean(R,G,B)` is not IR
  luminance, and pigments with equal grey values can be IR-opaque or
  IR-transparent. The residual head only helps if the required residual is small
  and unstructured; whether that holds is an empirical question and the answer is
  in [final-comments.md](final-comments.md).

---

## 5. Known limitations

### 5.1 Duplicated conv block across architecture modules
An almost-identical two-`Conv2D` → GroupNorm → `relu` block is redefined in
`unet.py`, `unet_v2.py`, `unet_restormer.py`, `unet_nll.py`, `attention_unet.py`,
`attention_unet_nll.py`, and `efficientnet_unet.py` (`resunet.py` / `resunet_nll.py`
have an analogous residual block). Only the group-count helper has been extracted
to `norm_utils.py`; the block itself still requires editing several files in
lockstep. **Fix**: extract a shared `scripts/blocks.py`.

### 5.2 No CI pipeline
A real `pytest` suite and a `ruff` configuration exist, but nothing runs them on
push or PR. **Fix**: a minimal GitHub Actions workflow running `ruff check`,
`ruff format --check`, `pytest`.

### 5.3 No `.env.example`
`config.py` supports `.env` overrides but no template documents the keys. **Fix**:
add `.env.example` listing every `Settings` field with its default.

### 5.4 Early-bound `settings` defaults in signatures
`trainer.py` / `trainer_nll.py` / `config.py` bind `settings.X` as *default
parameter values*, evaluated once at import. Mutating `settings` at runtime and
then calling one of these functions without the explicit argument silently uses
the import-time value. **Fix**: default to `None` and resolve inside the body, or
document `settings` as immutable after import.

### 5.5 Whole-image inference is not compared against tiled inference
Every evaluation runs `pad_to_multiple` + a single `model.predict()` on the
full-resolution `data/test/` images, which are far larger than the training
patches. It runs without memory issues on this hardware, but whether a
Gaussian-blended overlapping-patch path would change the results on very large
scans was never measured directly.

### 5.6 Data-loading optimisations left on the table
`build_dataset` re-decodes JPEGs every epoch (a `.cache()` after pair loading
would remove that), and no architecture opts into mixed precision. Both are
low-risk throughput wins given the 100-epoch cap and the size of the model set.

---

## 6. Alternative architectures not pursued

Roughly ordered from most compatible with a small dataset to most data-hungry.

**Adversarial term (PatchGAN / pix2pix).** A lightweight PatchGAN discriminator on
top of a U-Net generator, with the current pixel/frequency losses as an auxiliary
term. Standard for image-to-image translation and tends to sharpen fine detail.
Deprioritised on purpose: small dataset plus hallucination risk for a forensic
tool, where a plausible-looking fabricated mark is worse than a missed one.
- P. Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks," *CVPR*, 2017. https://doi.org/10.48550/arXiv.1611.07004
- G.H. Cann et al., "Resolution enhancement in the recovery of underdrawings via style transfer by GANs," 2021. https://doi.org/10.48550/arXiv.2102.00209 — direct on-domain precedent.

**Modern CNN backbone (ConvNeXt / ConvNeXt V2).** Drop-in replacement for the
EfficientNetB0 encoder (same frozen-backbone pattern, only the skip-layer names
change). Keeps the convolutional inductive bias while closing most of the gap to
transformers.
- Z. Liu et al., "A ConvNet for the 2020s," *CVPR*, 2022. https://doi.org/10.48550/arXiv.2201.03545
- S. Woo et al., "ConvNeXt V2," *CVPR*, 2023. https://doi.org/10.48550/arXiv.2301.00808

**Pure transformer U-Net (Swin-UNet).** More data-hungry than CNNs; only worth
pursuing if the paired dataset grows substantially.
- H. Cao et al., "Swin-Unet," 2021. https://doi.org/10.48550/arXiv.2105.05537

**Multi-band / multispectral extension (MST++).** Relevant only if acquisition
moves beyond a single IR band — a spectral-wise transformer built for exactly
narrow-band-from-RGB reconstruction.
- Y. Cai et al., "MST++," *CVPRW*, 2022. https://doi.org/10.48550/arXiv.2204.07908
