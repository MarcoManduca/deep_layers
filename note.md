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

### Update: evaluating the learned z-score itself, not just `mu`

Now that all four `*_nll` architectures have real training runs with both
`gaussian_nll` and `beta_nll` (`code-review.md` §7.6 "Update"), a gap
became apparent: `mae`/`ssim`/`psnr` only ever compare `mu` against
`real_IR` — they never look at `sigma`, so they cannot tell whether the
learned z-score above is any good. On `modern` (a real test pair built ad
hoc with specific known hidden details), the `beta_nll` z-score
subjectively reveals more of the intended detail than `gaussian_nll`'s,
despite `beta_nll` scoring worse on `mae`/`ssim`/`psnr` — the two are
measuring different things and can disagree. Three follow-ups (not yet
implemented):

1. A calibration metric for `sigma` on its own (e.g. coverage probability,
   or correlation between `|real_IR - mu|` and `sigma`). — **IMPLEMENTED**,
   see below.
2. A ground-truth-mask detection metric (AUROC/precision-recall) on
   `modern` specifically, since it's one purpose-built image with known
   hidden-detail regions — annotate once, score every candidate signal.
   **Still open** — the one follow-up that needs manual annotation.
3. Parametric contrast control for the z-score plots (currently a fixed
   `Z_VMAX=4.0` clip) — percentile- or gamma-based, applied after the
   z-score maps are already computed. — **IMPLEMENTED**, see below.

### Update: `scripts/calibration.py` and `scripts/contrast.py`

**Follow-up 1** is `scripts/calibration.py`: post-hoc, NumPy-only, working
on the `(mu, sigma)` arrays that inference already produces, so every
existing checkpoint can be scored without retraining. It reports coverage
probability at `+/- k*sigma` against the nominal Gaussian value, a
reliability curve over equal-population `sigma` bins with its ENCE summary
(Levi et al.), the Spearman correlation between `|real_IR - mu|` and
`sigma`, the moments of the learned z-score (`z_std = 1` when calibrated),
the mean Gaussian NLL as a proper scoring rule, and — deliberately
alongside them — sharpness and dispersion. The last two are the guard
against the degenerate case: a large *constant* `sigma` scores as
perfectly calibrated while reducing the learned z-score to a rescaled raw
delta, so `dispersion ~ 0` invalidates a good calibration score rather
than confirming it. `evaluate_calibration(...).summary()` returns one flat
row per model, tabulated per architecture in `032_evaluation_v2.ipynb`
§4b; `visualization_nll.plot_calibration` draws the reliability diagram,
the coverage bars and the z-histogram against `N(0, 1)`.

**Follow-up 3** is `scripts/contrast.py`: `ZScale` makes the display limit
a parameter — either fixed (`|z| <= 4`, the original behaviour and still
the default) or a percentile of `|z|` in the maps being shown — with an
optional gamma compression of the ramp, so faint detail at `|z| ~ 1` can
be expanded without touching what was computed. Everything is applied
*after* the z-score maps exist, so contrast variants are compared by
re-plotting, never by re-predicting. `ZScale.apply_many` derives a single
shared limit across several models, which is what keeps a cross-
architecture figure honest; `062_model_comparison_v2.ipynb` §2c sweeps
fixed / p99.5 / p99.5+gamma over the four `beta_nll` architectures.

Both are wired into `plot_zscore` / `plot_delta_comparison` (the `vmax` /
`z_vmax` arguments became a single `z_scale`), and `learned_zscore` in
`calibration.py` is now the single definition of `(real - mu) / sigma`
shared by the plots and the metrics, so the signal being scored and the
signal being displayed cannot drift apart.

Note what this does **not** settle: these metrics say whether `sigma` is
*trustworthy*, not whether the z-score *reveals underdrawings*. Only
follow-up 2 answers that, and it remains the gate on the `gaussian_nll`
vs. `beta_nll` question.

