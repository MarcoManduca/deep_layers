# Design notes — regional analysis of the IR delta

## Context

The current `delta` (`delta = |real_IR − predicted_IR|`) is a raw,
pixel-by-pixel absolute difference. It does not distinguish between:

- a **gray-level shift** (local bias — e.g. a more absorbent substrate,
  non-uniform illumination, IR capture exposure) that moves a zone's mean
  value while preserving its structure/texture;
- a **genuine structural discontinuity** (underdrawing, pentimento, reuse of
  the support) that alters the shape of the local distribution, not just its
  mean value.

A purely absolute delta conflates these two cases and can produce false
positives on zones that are simply darker/lighter for innocuous physical
reasons (substrate, acquisition conditions) rather than because of actual
hidden detail.

## Proposal

Complement the raw delta with a regional (per-zone/window) analysis based on
mean gray level, trend, uniformity and variability, in order to separate
"expected but different for physical reasons" from "genuinely unexpected".

### 1. Luminance / contrast / structure decomposition

Reuse the decomposition underlying SSIM (Wang et al.), already present in
`losses.py` / `metrics.py`: SSIM is the product of three terms — luminance,
contrast, structure. Compute the three maps separately per local window,
instead of only the aggregated scalar index, and use **only the structure
component** as the "hidden detail" indicator, treating luminance/contrast
differences as a plausible substrate/acquisition effect. This is a natural
extension, reusing infrastructure already present in the project.

### 2. Local normalization (per-window z-score)

For each zone, compute:

```
z_real = (real − μ_real_local) / σ_real_local
z_pred = (pred − μ_pred_local) / σ_pred_local
```

and compare the normalized values instead of the raw ones. A zone with the
same pattern but a different mean/scale would have a normalized delta ≈ 0.

### 3. Per-zone distribution comparison

In addition to the mean, compute variance/uniformity (e.g. local entropy,
standard deviation) and compare distributions with a metric such as
Wasserstein distance or the KS-statistic over sliding windows. If the
distributions are shape-compatible but shifted, an acquisition artifact is
likely; if the shape itself changes (e.g. a new bimodality, an unexpected
tail), it is more indicative of a genuine mark.

### 4. Output

Instead of a single delta map, produce 2-3 complementary maps:

- raw delta (current);
- structural/normalized delta (new);
- a "confidence" map (where the two techniques agree vs. diverge),

to help the conservator distinguish acquisition noise from real signal.

## Implementation status

- §1, §3 and §4 are implemented as post-hoc analysis in
  `scripts/delta_analysis.py` (`analyze_delta`), visualised by
  `plot_delta_analysis` and driven from `notebooks/050_delta_analysis.ipynb`.
- §2 is additionally implemented as a **training objective**:
  `scripts.losses.combined_loss_normalized` (MAE + per-window z-score).
  It is registered in `trainer._ARCH_LOSSES` as `LossName.NORMALIZED` but
  not yet assigned to any architecture. Its window parameters, weights and
  the standard-deviation floor live in `Settings`; the floor and the weight
  ratio were set from measurements on the project's own IR set rather than
  chosen by hand. See the loss docstring for the two properties that
  constrain how it may be used (affine invariance, term-scale mismatch).

## Placement in the project

A separate analysis module (e.g. a future `scripts/delta_analysis.py`),
downstream of the existing inference (`predict_with_overlap` in
`scripts/inference_utils.py`), without touching the model architectures.
What is already available in `metrics.py` / `visualization.py` to build on
should be checked at implementation time.

