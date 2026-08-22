# Code Review — Deep Layers

Scope: `scripts/`, `tests/unit/`, `pyproject.toml`, `requirements.txt`, notebook usage of the reviewed modules. Rewritten during the project-finalization pass (2026-08-22) to describe the current codebase only — prior findings that were fixed, superseded, or abandoned (duplicate `_conv_block` fix, deferred-import fix, `predict_with_overlap`/`inference_utils*.py`, `pseudo_mask.py`, the unimplemented literature directions' historical narrative) have been dropped rather than kept as a changelog. Project history lives in git log and (for evaluation methodology specifically) `evaluation.md`.

---

## 1. Code structure

```
scripts/
├── config.py                Pydantic Settings singleton (paths, hyperparameters)
├── reproducibility.py       Global RNG seeding
├── dataset.py                Pair discovery, artwork-and-mockups split, tf.data pipeline, padding
├── augmentation.py           Stateless paired RGB/IR augmentation
│
├── unet.py                   Baseline UNet
├── unet_v2.py                 UNet with ablatable strided-conv / upsample-conv / dropout variants
├── unet_restormer.py         UNet with a Restormer self-attention block at the bottleneck
├── resunet.py                 UNet with residual blocks
├── attention_unet.py         UNet with additive attention gates
├── efficientnet_unet.py      UNet with a pretrained EfficientNetB0 encoder
│
├── unet_nll.py                Heteroscedastic (mu, log_var) counterpart of unet.py
├── resunet_nll.py             Heteroscedastic counterpart of resunet.py
├── attention_unet_nll.py     Heteroscedastic counterpart of attention_unet.py
├── efficientnet_unet_nll.py  Heteroscedastic counterpart of efficientnet_unet.py
├── nll_layers.py              ClipLogVar — shared serializable layer for the four *_nll builders
│
├── losses.py                  combined_loss, combined_loss_advanced, gaussian_nll_loss, beta_gaussian_nll_loss
├── metrics.py                  PSNR/SSIM Keras metrics (deterministic and Mu*-prefixed NLL variants)
├── trainer.py                  Model factory, compile, callbacks, checkpoint load (deterministic architectures)
├── trainer_nll.py              Same, for the four NLL architectures and both NLL losses
│
├── delta_analysis.py          Local SSIM luminance/contrast/structure decomposition, confidence map
├── calibration.py              Sigma calibration metrics, learned z-score, structural z-score
├── contrast.py                  Display-contrast settings for z-score maps (ZScale)
├── detection.py                 AUROC/average-precision of a signal against a ground-truth mask
├── stroke_stats.py              Unsupervised structure-tensor coherence (reference-free)
│
├── visualization.py             Plot helpers for the deterministic pipeline
└── visualization_nll.py         Plot helpers for the NLL pipeline (signal comparison, calibration diagrams)

tests/unit/                      One test module per script, pure-logic focused
```

Module boundaries map 1:1 to pipeline stages, duplicated once for the deterministic/NLL split where the two genuinely differ (model builders, trainer, metrics, visualization) and shared where they don't (`dataset.py`, `delta_analysis.py`, `detection.py`, `stroke_stats.py`, `contrast.py`). `trainer.py`/`trainer_nll.py` each act as the single integration point wiring an architecture to its loss (`uses_advanced_loss` for the deterministic side, the `loss_name` parameter for NLL) — both training and checkpoint reloading derive the loss the same way, so they cannot silently diverge.

Docstrings are consistently NumPy-style with parameters, returns, and raises documented; type hints are used throughout. No circular imports found.

No CI configuration (`.github/workflows/`) exists despite a real `pytest` suite and `ruff` config — tests currently only run when a developer remembers to invoke them locally (see §3.2).

Checkpoints are organized by family under `models/`: `models/deterministic/<arch>/`, `models/nll_gaussian/<arch>/`, `models/nll_beta/<arch>/`. `get_callbacks`/`load_model`/`load_model_nll` all take `model_dir` as a caller-supplied parameter, so this tree is a convention enforced by the calling notebooks, not by the trainer code itself.

---

## 2. What each module does

**`config.py`** — A single `Settings` object (pydantic-settings, `.env`-overridable) holds every path and hyperparameter: patch multiple, batch size, LR, epochs, seed, split ratios (`TRAIN_RATIO`/`VAL_RATIO`/`MOCKUP_TEST_RATIO`), mockup artwork IDs, loss weights, crop size, NLL log-variance clip bounds. All other modules import `settings` from here rather than hardcoding values.

**`reproducibility.py`** — `set_global_seed()` seeds Python/NumPy/TensorFlow RNGs via `tf.keras.utils.set_random_seed`. Deliberately does *not* enable `tf.config.experimental.enable_op_determinism()` (unsupported/slow on the Metal backend); per-sample augmentation determinism is handled separately by a stateless-RNG mechanism.

**`dataset.py`** — Discovers RGB/IR pairs by matching filename stems (`load_image_pairs`), extracts an artwork ID from each filename (`extract_artwork_id`). Two split functions: `grouped_train_val_test_split` (every artwork ID is an indivisible group, leakage-free) and `mockup_aware_train_val_test_split` (real artworks still grouped and leakage-free; the synthetic paint-on-support mockup groups in `settings.MOCKUP_ARTWORK_IDS` are instead split at the individual-pair level, `MOCKUP_TEST_RATIO` to test, the rest to train/val — since those groups exist to be learned from, not generalized to). The latter is the only split every training/evaluation notebook actually uses; `grouped_train_val_test_split` remains as the general-purpose leakage-free primitive the mockup-aware split is built on. `pad_to_multiple` pads a single image to the nearest multiple of `N` (used at whole-image inference time on the full-resolution `data/test/` images, which run far larger than the 400×400 training patches — see §4 on why no tiled-inference path is used for that). `build_dataset` assembles the `tf.data` pipeline: load → shuffle → decode/normalize → (optional) augment with a deterministic per-element seed → batch → prefetch.

**`augmentation.py`** — `augment_pair` applies a shared random crop (concatenating RGB+IR on the channel axis so both share one crop box), then identical horizontal/vertical flips to both images, then brightness/contrast jitter to RGB only (IR reflectance is treated as illumination-independent). All randomness is stateless and seed-derived for reproducibility.

**`unet.py` / `resunet.py` / `attention_unet.py` / `efficientnet_unet.py`** — Four fully-convolutional deterministic encoder–decoders, all accepting `(None, None, 3)` input and producing `(None, None, 1)` sigmoid output. `efficientnet_unet.py` additionally defines `_ResizeToMatch` to reconcile off-by-one spatial mismatches between the frozen EfficientNetB0 backbone's floor-rounded downsampling and the decoder's exact-doubling `Conv2DTranspose`.

**`unet_v2.py`** — Same topology as `unet.py`, with three independently-toggleable modifications (all off by default, matching `unet.py` exactly when off): `use_strided_conv` (learned stride-2 downsampling instead of `MaxPool2D`), `use_upsample_conv` (bilinear upsample + `Conv2D` instead of `Conv2DTranspose`, avoiding checkerboard artifacts — via a custom `_Upsample2x` layer, since `layers.UpSampling2D` inspects the *static* shape and fails on this model's fully dynamic input), `dropout_rate` (`SpatialDropout2D` at the bottleneck and first decoder block only, to avoid compounding with `BatchNormalization` present in every conv block).

**`unet_restormer.py`** — Same topology as `unet.py`, with a single `RestormerBlock` (Zamir et al., CVPR 2022) inserted right after the bottleneck's conv block: MDTA (channel-wise attention, linear cost in image size) and GDFN (gated depthwise feed-forward). The only self-attention in the deterministic family, scoped narrowly (bottleneck only) rather than a full transformer architecture.

**`unet_nll.py` / `resunet_nll.py` / `attention_unet_nll.py` / `efficientnet_unet_nll.py`** — Heteroscedastic counterparts of the four base architectures: a 2-channel `(mu, log_var)` output instead of one, trained with a Gaussian negative-log-likelihood loss instead of `combined_loss`. `efficientnet_unet_nll.py` reuses `efficientnet_unet.py`'s pretrained encoder/skip-connections/`_ResizeToMatch` directly rather than duplicating them. `nll_layers.py::ClipLogVar` (clips `log_var` into `[NLL_LOG_VAR_MIN, NLL_LOG_VAR_MAX]`) is shared by all four so only one class is registered under that name for `tf.keras.models.load_model`.

**`losses.py`** — `combined_loss` (MAE + 1−SSIM) for `unet`/`unet_v2`/`unet_restormer`/`resunet`/`attention_unet`; `combined_loss_advanced` (MAE + Laplacian pyramid + FFT magnitude) for `efficientnet_unet` only (`trainer.uses_advanced_loss` is the single source of truth for which). `gaussian_nll_loss` and `beta_gaussian_nll_loss` (Seitzer et al. 2022's gradient-starvation correction, weighted by `stop_gradient(sigma) ** (2*beta)`) for the NLL architectures, selectable via `trainer_nll.NLL_LOSSES`.

**`metrics.py`** — `PSNRMetric`/`SSIMMetric` for the deterministic architectures; `MuMAEMetric`/`MuSSIMMetric`/`MuPSNRMetric` for the NLL ones, reading only the `mu` channel so they stay directly comparable to the deterministic metrics regardless of loss variant.

**`trainer.py`** — `get_model` (name → builder factory over `unet`/`unet_v2`/`unet_restormer`/`resunet`/`attention_unet`/`efficientnet_unet`), `uses_advanced_loss`/`compile_model`, `get_callbacks` (ModelCheckpoint/EarlyStopping/ReduceLROnPlateau/TensorBoard), `load_model` (checkpoint reload + recompile with matching loss).

**`trainer_nll.py`** — Same shape, for the four NLL architectures: `get_model_nll`, `compile_model_nll`/`load_model_nll` (both take `loss_name`/`beta`), reusing `trainer.get_callbacks` unchanged (generic, not architecture-specific).

**`delta_analysis.py`** — `compute_local_stats`/`compute_ssim_components` decompose local SSIM into luminance/contrast/structure maps (`gaussian_local_filter`, public so `calibration.py` can smooth `sigma` with the exact same window). `analyze_delta` combines these into a `DeltaAnalysisResult`: `raw_delta`, `structural_delta` (`1 - structure`, the "hidden detail" indicator, insensitive to substrate/acquisition gray-level shifts that inflate raw delta), and a `confidence_map` (agreement between the two after min-max normalization).

**`calibration.py`** — `learned_zscore` (`(real - mu) / sigma`) and `structural_zscore` (`structural_delta / smoothed sigma` — the best-performing detection signal found in this project's evaluation work) for the NLL family. `evaluate_calibration` scores whether `sigma` itself can be trusted (coverage probability, ENCE reliability, error/sigma correlation, sharpness/dispersion, mean Gaussian NLL) — independent of mask-based detection scoring, since it needs no ground truth.

**`contrast.py`** — `ZScale`/`ZScaleMode` turn a z-score map's display limit into a parameter (`FIXED` or `PERCENTILE`, optional `gamma` compression), so contrast variants cost a re-plot, not a re-prediction. `apply_many` computes one shared limit across several maps for honest cross-model comparison.

**`detection.py`** — `evaluate_detection`/`rank_signals`: AUROC (prevalence-independent, primary ranking) and average precision (with prevalence reported alongside, since that's AP's chance level) of any magnitude signal against any ground-truth mask.

**`stroke_stats.py`** — `stroke_coherence`: structure-tensor coherence, needing no reference mask. An underdrawing is made of oriented, elongated strokes; prediction noise is isotropic — those are statistically separable without knowing where the strokes are. The reference-free corroborating axis alongside `detection.py`'s AUROC.

**`visualization.py`** — Plot helpers for the deterministic pipeline: sample RGB/IR grids, RGB\|real\|predicted triptychs, the raw delta heatmap, training-curve plots from a Keras `History.history` dict.

**`visualization_nll.py`** — `plot_predictions_nll`/`plot_zscore` (single-model sanity views); `plot_signal_comparison` (one signal type across several models, one shared scale); `plot_calibration` (reliability diagram, coverage bars, z-distribution). `DEFAULT_Z_SCALE` (percentile `p99.5`, no gamma) is the project's chosen contrast setting, picked by visual comparison across the ground-truth images.

---

## 3. Known open issues

### 3.1 Duplicated `_conv_block` across seven architecture modules (maintainability)
An identical (or near-identical) two-Conv2D→BN→activation block is redefined verbatim in `unet.py`, `unet_v2.py`, `unet_restormer.py`, `unet_nll.py`, `attention_unet.py`, `attention_unet_nll.py`, and `efficientnet_unet.py` (`resunet.py`/`resunet_nll.py` have an analogous `_residual_block`). A change to the block requires editing up to seven files in lockstep, with no compiler/test signal if one is missed. Grew from 4 to 7 duplicates as the NLL/`unet_v2`/`unet_restormer` variants were added.
**Fix**: extract `_conv_block` into a shared `scripts/blocks.py` and import it from each architecture module.

### 3.2 No CI pipeline
A real `pytest` suite (18 test modules) and `ruff` configuration exist, but nothing runs them automatically on push/PR. Regressions can land on `main` without the suite being exercised.
**Fix**: add a minimal GitHub Actions workflow running `ruff check`, `ruff format --check`, and `pytest` on push/PR.

### 3.3 Early-bound `settings` defaults in function signatures (subtle footgun)
`trainer.py`/`trainer_nll.py` (`compile_model`, `load_model`, `compile_model_nll`, `load_model_nll`) and `config.py` bind `settings.LEARNING_RATE`, `settings.MODELS_DIR`, etc. as *default parameter values*, evaluated once when the module is imported. If `settings` is mutated at runtime (e.g. a notebook does `settings.LEARNING_RATE = 5e-4` to experiment), functions called without an explicit argument silently keep using the value captured at import time.
**Fix**: default to `None` and resolve `settings.X` inside the function body, or document explicitly that `settings` must be treated as immutable after import.

### 3.4 No `.env.example`
`config.py` supports `.env` overrides, but no template file documents the available keys for a new contributor.
**Fix**: add a `.env.example` listing every `Settings` field with its default and a one-line comment.

---

## 4. Possible optimizations

- **`tf.data.Dataset.cache()`**: `build_dataset` re-decodes JPEGs from disk every epoch. For a dataset of paintings that fits in memory, adding `.cache()` after pair loading (before augmentation) would remove repeated JPEG-decode cost from every epoch after the first.
- **Mixed precision training**: none of the ten architectures opt into `tf.keras.mixed_precision.set_global_policy("mixed_float16")`. On supported hardware this typically gives a meaningful throughput improvement for a negligible accuracy cost, especially useful given `EPOCHS=100` and ten architecture/loss combinations to train.
- **Two-phase fine-tuning for `efficientnet_unet`/`efficientnet_unet_nll`**: both builders expose `freeze_encoder=False`, but nothing in `trainer.py`/`trainer_nll.py`/the training notebooks uses it — the encoder trains fully frozen for the whole run. The standard transfer-learning recipe (train the decoder to convergence with the encoder frozen, then unfreeze and fine-tune end-to-end at a lower learning rate) is unimplemented.
- **No explicit regularization in `unet.py`/`resunet.py`/`attention_unet.py`**: these rely on BatchNorm alone at a 1024-channel bottleneck, large relative to a painting-scale dataset split by artwork. `unet_v2.py`'s `dropout_rate` option addresses this for its own architecture but the idea hasn't been ported back to the three original builders.
- **EfficientNetB0 backbone is dated (2019)**: `tf.keras.applications` ships more recent ImageNet-pretrained backbones (e.g. `EfficientNetV2B0`, `ConvNeXtTiny`) as drop-in replacements — same `include_top=False`/frozen-encoder pattern would apply, only `_SKIP_LAYER_NAMES` would need re-mapping. See §6 for literature justifying this and other backbone/architecture changes.
- **Whole-image inference on `data/test/`, not tiled**: every evaluation/perceptual notebook (04x/05x) runs `pad_to_multiple` + a single `model.predict()` call on the real `data/test/` images, which range up to ~3700×2800 px — far larger than the 400×400 training patches. A Gaussian-blended overlapping-patch inference path existed earlier in the project's history and was removed as unused once every notebook had already converged on the whole-image approach; whether tiled inference would change results on these large images (memory aside — it already runs without issue on this hardware) was never directly compared. Worth a controlled comparison before relying on whole-image inference as a production assumption for arbitrarily large scans.

---

## 5. Architecture-specific comments

The ten architectures (six deterministic-family builders including `unet_v2`/`unet_restormer`, four NLL-family builders) are purely convolutional, feed-forward encoder–decoders — no GAN/adversarial term, no diffusion process. `unet_restormer.py` is the only one with a self-attention block, scoped narrowly (bottleneck only, linear-cost channel attention). Given the likely small size of a paired painting dataset (artwork-grouped split, `config.py`), this remains a defensible, low-data-friendly design.

- **`attention_unet.py`'s "attention" is CNN spatial gating, not self-attention.** `_attention_gate` implements Oktay et al.'s additive gate — it reweights the encoder skip connection using the decoder's gating signal, but has no long-range/global receptive field. `unet_restormer.py` addresses this — on `unet`, not `attention_unet`, so the two mechanisms (spatial gating vs. channel self-attention) haven't been combined in one architecture.
- **`_ResizeToMatch` in `efficientnet_unet.py`/`efficientnet_unet_nll.py`** is a minimal, targeted fix for the off-by-one skip/upsample mismatch caused by EfficientNet's floor-rounded stride-2 convolutions. It would need re-verifying if the backbone is ever swapped (different backbones round dimensions differently or not at all). `unet_v2.py`'s `_Upsample2x` and `nll_layers.py`'s `ClipLogVar` are the same underlying pattern (a small custom layer, `register_keras_serializable`-decorated, needed because Keras cannot otherwise reconstruct it from a saved checkpoint) recurring across the codebase — worth keeping in mind if a fourth case comes up, as a signal that this pattern might deserve a shared base utility.
- **Bottleneck channel counts are inconsistent across from-scratch architectures vs. EfficientNet UNet**: `unet.py`/`resunet.py`/`attention_unet.py` default to a 1024-channel bottleneck fed by a `[64,128,256,512]` encoder, while `efficientnet_unet.py`'s decoder starts at 256 channels because the frozen EfficientNetB0 backbone already outputs 1280 channels. This isn't a bug (genuinely different feature scales) but makes head-to-head parameter-count/capacity comparisons across architectures less apples-to-apples than a metrics table alone might suggest.

## 6. Alternative architectures not yet tried

Roughly ordered from "most compatible with a small dataset" to "most data-hungry."

### 6.1. Adversarial term (PatchGAN / pix2pix-style)

Add a lightweight PatchGAN discriminator on top of one of the existing UNet generators, keeping the current pixel/frequency losses as an auxiliary term alongside the adversarial loss. Standard recipe for image-to-image translation; tends to sharpen fine detail beyond what pixel-wise regression losses alone can achieve. Explicitly deprioritized in an earlier session's discussion — small dataset + hallucination risk for a forensic/scholarly tool where a plausible-looking but fabricated mark would be worse than a missed one.

- P. Isola, J.-Y. Zhu, T. Zhou, A.A. Efros, *"Image-to-Image Translation with Conditional Adversarial Networks,"* CVPR, 2017. https://doi.org/10.48550/arXiv.1611.07004
- G.H. Cann, A. Bourached, R.-R. Griffiths, D.G. Stork, *"Resolution enhancement in the recovery of underdrawings via style transfer by generative adversarial deep neural networks,"* 2021. https://doi.org/10.48550/arXiv.2102.00209 — direct on-domain precedent, multi-scale GAN recovering underdrawings on works by Leonardo, explicitly addressing training-data scarcity typical of the art domain.

### 6.2. Modern CNN backbone (ConvNeXt / ConvNeXt V2)

Drop-in replacement for the EfficientNetB0 encoder in `efficientnet_unet.py` (same frozen-backbone pattern, only `_SKIP_LAYER_NAMES` would need re-mapping). Keeps the convolutional inductive bias — important with little data — while closing most of the accuracy gap to transformers.

- Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie, *"A ConvNet for the 2020s,"* CVPR, 2022. https://doi.org/10.48550/arXiv.2201.03545
- S. Woo et al., *"ConvNeXt V2,"* CVPR, 2023. https://doi.org/10.48550/arXiv.2301.00808

### 6.3. Pure transformer U-Net (Swin-UNet) — use with caution

A fully transformer-based U-shaped architecture, typically more data-hungry than CNNs; the same authors as the on-domain GAN paper above explicitly flag the scarcity of paired training works in the art domain. Only worth pursuing if the RGB/IR dataset turns out to be large enough.

- H. Cao et al., *"Swin-Unet,"* 2021. https://doi.org/10.48550/arXiv.2105.05537

### 6.4. Multi-band / multispectral extension (MST++)

Relevant only if the project extends beyond a single IR band to multiband reflectography — a spectral-wise transformer designed for exactly this kind of narrow-band-from-RGB reconstruction, winner of the NTIRE 2022 Spectral Recovery Challenge.

- Y. Cai et al., *"MST++,"* CVPRW, 2022. https://doi.org/10.48550/arXiv.2204.07908

## 7. Implemented directions — pointers, not a changelog

These were proposed and built earlier in the project. Rationale and code live here; real-run results and current standing (which signal/architecture wins, current default recommendations) live in the 04x/05x notebooks and `evaluation.md`, not duplicated here to avoid drifting out of sync with the notebooks again.

- **Restormer bottleneck block** (§5, `unet_restormer.py`) — Zamir et al., CVPR 2022. https://doi.org/10.48550/arXiv.2111.09881
- **`unet_v2`'s three ablatable modifications** (§2, `unet_v2.py`) — strided-conv downsampling, upsample-conv (Odena et al., *"Deconvolution and Checkerboard Artifacts,"* Distill, 2016), bottleneck-only dropout (Li et al., *"Understanding the Disharmony between Dropout and Batch Normalization,"* 2019).
- **Heteroscedastic aleatoric uncertainty head** (§2, `*_nll.py`, `losses.py`, `calibration.py`) — the RGB→IR mapping is inherently one-to-many (pigments alike in visible light can have markedly different IR reflectance), so a single deterministic prediction forces the network to average over genuinely ambiguous cases. Kept on theoretical grounds independent of its mixed empirical record on `mae`/`ssim`/`psnr` alone.
  - D.A. Nix, A.S. Weigend, *"Estimating the mean and variance of the target probability distribution,"* IEEE ICNN, 1994. https://doi.org/10.1109/ICNN.1994.374138
  - A. Kendall, Y. Gal, *"What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?,"* NeurIPS, 2017. https://doi.org/10.48550/arXiv.1703.04977
  - M. Seitzer, A. Tavakoli, D. Antic, G. Martius, *"On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks,"* ICLR, 2022. https://doi.org/10.48550/arXiv.2203.09168 — the β-NLL correction implemented as `losses.beta_gaussian_nll_loss`.
- **Sigma calibration and z-score contrast** (§2, `calibration.py`, `contrast.py`) — closes the gap that `mae`/`ssim`/`psnr` read only `mu` and cannot say whether `sigma` itself is trustworthy.
  - D. Levi, L. Gispan, N. Giladi, E. Fetaya, *"Evaluating and Calibrating Uncertainty Prediction in Regression Tasks,"* Sensors, 2022. https://doi.org/10.48550/arXiv.1905.11659
  - T. Gneiting, F. Balabdaoui, A.E. Raftery, *"Probabilistic forecasts, calibration and sharpness,"* JRSS-B, 2007. https://doi.org/10.1111/j.1467-9868.2007.00587.x
- **Ground-truth-mask signal evaluation** (§2, `detection.py`, `stroke_stats.py`) — real hand-drawn masks (`data/test/annotations/*_Map.png`) replaced an earlier data-derived cross-modal pseudo-mask approach once enough real annotations existed; `stroke_stats.py`'s reference-free structure-tensor coherence remains as the corroborating unsupervised axis regardless of which reference is used.
