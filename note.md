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

## Placement in the project

A separate analysis module (e.g. a future `scripts/delta_analysis.py`),
downstream of the existing inference (`predict_with_overlap` in
`scripts/inference_utils.py`), without touching the model architectures.
What is already available in `metrics.py` / `visualization.py` to build on
should be checked at implementation time.

## Update: fixed-window normalization vs. learned (heteroscedastic) normalization

`scripts/delta_analysis.py` implements §2 above as written: a **fixed**
per-window z-score, computed independently for `real` and `pred` from
their own local mean/std over an 11×11 Gaussian window (same window as
`tf.image.ssim`). It answers *"is this pixel unusual relative to its own
local spatial neighborhood?"* — a purely spatial, model-agnostic
normalization, blind to what color/pigment is actually present.

A second, complementary normalization was added later (`code-review.md`
§7.6, implemented as a pilot on `feature/heteroscedastic-nll`):
`attention_unet_nll` (`scripts/attention_unet_nll.py`) predicts a
per-pixel `(mu, log_var)` instead of a single deterministic IR value,
trained with a Gaussian negative-log-likelihood
(`scripts/losses.py::gaussian_nll_loss`). `sigma = exp(0.5 * log_var)` is
then a **learned** per-pixel scale, and the resulting z-score

```
z = (real_IR − mu) / sigma
```

(`scripts/visualization_nll.py::plot_zscore`) answers a different
question: *"is this pixel unusual relative to how much this RGB
color/context has historically varied in IR, across the whole training
set?"* — a color/context-conditioned normalization, learned end-to-end,
rather than a fixed spatial one.

The two are not redundant — they normalize against different baselines
and can disagree usefully:

| | §2 fixed-window z-score | learned z-score (§7.6) |
|---|---|---|
| Normalizes against | local spatial neighborhood (11×11 window, same image) | historical variability of this RGB color/context (learned from the whole training set) |
| Needs | nothing beyond `real`/`pred` | a trained `attention_unet_nll` checkpoint |
| Blind spot | a color that is *always* variable will still flag large local contrast as unusual | a region whose local neighborhood is atypical but whose color has always been stable elsewhere will still flag correctly |

Not yet evaluated on real (non-smoke-trained) checkpoints — see
`.claude/handoff/HANDOFF.md` "Next steps" for the pending full training
run and qualitative comparison on real paintings.

