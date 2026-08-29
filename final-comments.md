# Final evaluation — model comparison and recommendations

What the models achieve, how they compare, and which ones to use. The reasoning
proceeds from the reconstruction task, to the detection signal built on top of it,
to cross-validation, to the design choices that were tested and did *not* help,
and finally to the recommended models and the ceiling of the approach.

The theory behind the design is in [theory-links.md](theory-links.md); the code is
in [code-review.md](code-review.md).

---

## 1. The deliverable and what limits it

The product is the **residual/signal map** that reveals underdrawings,
*pentimenti*, and reused supports. The pipeline predicts IR from RGB (`mu`),
subtracts it from the real IR, and analyses what is left. But the signal of
interest is, by construction, *the part of the IR that RGB cannot predict*, so
detector quality depends on exactly two things:

1. how well `mu` reproduces the *ordinary surface* IR — a cleaner residual makes a
   real anomaly stand out;
2. how well "`mu` is wrong because of a hidden mark" is separated from "`mu` is
   wrong because RGB→IR is genuinely ambiguous here, or because of craquelure, or
   model imprecision".

Nothing an architecture change can do addresses (2) directly, and (1) is already
close to saturated — see §10.

Detection is scored against **three hand-drawn masks** (`GT01`, `GT02`, `GT03`).
Three masks support no strong statistical claim; every number below should be read
with that in mind, and the cross-validation in §5 exists to bound the uncertainty
that follows from it.

---

## 2. Reconstruction fidelity

On the held-out test set, MAE / SSIM / PSNR across the whole model set land in a
narrow band: **PSNR 16–19 dB, SSIM 0.43–0.58, MAE 0.09–0.17**. Within that band:

- `resunet` and its heteroscedastic counterpart lead pixel fidelity — MAE ≈ 0.11
  deterministic, ≈ 0.09 for `resunet_nll`, PSNR ≈ 18.6 dB.
- `unet`, `attention_unet`, `efficientnet_unet` cluster together a little lower
  (PSNR ≈ 16.2–16.5, SSIM ≈ 0.545–0.551).
- `unet_residual` at the default loss weight barely beats a plain `mean(R,G,B)`
  image — the residual head learns almost nothing (see §7).

The spread across architectures is small — a few tenths of a dB. This is the first
sign that model capacity is not the constraint.

---

## 3. The signal that carries detection

Starting from the raw residual, the candidate signals split cleanly into two
families with very different behaviour:

| family | signals | detection AUROC |
|---|---|---|
| **magnitude** | `raw_delta`, learned `\|z\|` = `\|real − mu\|/σ` | 0.37–0.60 — weak, at or below chance on some images |
| **structural** | `structural_delta` = `1 − local SSIM structure`; `structural_z` = `structural_delta / σ` | 0.62–0.72 — the usable signal |

The gap between the two families dwarfs any difference *within* either. The reason
is physical: substrate absorption, non-uniform illumination and exposure move a
region's mean IR level without changing its structure, and a magnitude signal
cannot tell that apart from a real mark. The SSIM structure term is invariant to
exactly those grey-level shifts, so `structural_delta` isolates the part of the
disagreement that looks like *content*.

Dividing the structural signal by the learned σ (`structural_z`, heteroscedastic
models only) adds a small, consistent improvement — the σ head's one measurable
contribution to detection.

**The detection signal is therefore fixed as `structural_delta` (deterministic) /
`structural_z` (heteroscedastic).**

---

## 4. Detection results

Mean AUROC over `GT01`–`GT03`, best signal per model:

| model | signal | AUROC (single split) |
|---|---|---|
| `unet` | `structural_delta` | 0.71 |
| `attention_unet` | `structural_delta` | 0.70 |
| `resunet` | `structural_delta` | 0.68 |
| `attention_unet_nll` | `structural_z` | 0.72 |
| `resunet_nll` | `structural_z` | 0.73 |
| `unet_residual` | `structural_delta` | 0.67 — equal to the `mean(R,G,B)` floor |
| `mean(R,G,B)` (no model) | `structural_delta` | 0.67 |

Two patterns are stable across every cut of the data:

- **`structural_z` ≥ `structural_delta`** on the heteroscedastic models, and the
  best heteroscedastic models (`attention_unet_nll`, `resunet_nll`) edge the best
  deterministic ones.
- **Per-image difficulty is a property of the painting, not the model.** GT01 is
  strong for everything (AUROC 0.73–0.82), GT03 is weak for everything
  (0.57–0.69). GT03's underdrawing is visible only in the real IR — no RGB→IR
  model can predict it — so its residual is inherently harder to rank.

Stroke coherence (reference-free) ranks the same structural signals on top,
independently of the masks — a real corroboration. It is *not* a tie-breaker
between signals of similar AUROC, though: it rewards oriented structure in
general, so a noisier signal can score a spuriously high coherence.

---

## 5. Cross-validation

Because there are only three masks, a **grouped 3-fold cross-validation** was run
on the two leading heteroscedastic models — real artworks partitioned by ID (no
section leakage), all mockups in every training set, the three GT paintings
external to every fold.

| model | `structural_z` AUROC, per fold | 3-fold ensemble | held-out MAE |
|---|---|---|---|
| `attention_unet_nll` | 0.699 ± 0.008 | 0.719 | 0.103 ± 0.013 |
| `resunet_nll` | 0.716 ± 0.011 | 0.728 | 0.092 ± 0.010 |

- **Fold-to-fold standard deviation is ≈ 0.01 AUROC** — far below the gap between
  the signal families, and below the spread between paintings. The single fixed
  split used everywhere else in the project was adequate; cross-validation
  *confirms* the point estimates rather than deflating them.
- The **3-fold ensemble** (pixel-wise mean of the fold predictions) beats every
  single fold on both signals — a small, free gain.
- `resunet_nll` is at least `attention_unet_nll`'s equal under cross-validation:
  ahead on pixel fidelity, ahead on `structural_z` (0.728 vs 0.719 ensemble,
  ~1 std so not decisive alone), and clearly better on the hard painting GT03
  (`structural_z` ensemble 0.690 vs 0.653).

**Headline result: `structural_z` detection AUROC ≈ 0.70 ± 0.01 per fold, ≈ 0.72
for a 3-fold ensemble**, on GT01–03.

---

## 6. The heteroscedastic head

The RGB→IR mapping is genuinely one-to-many, and the heteroscedastic head is the
theoretically correct response (see [theory-links.md](theory-links.md)). Measured
against the deliverable, its record is mixed:

- **Detection payoff is marginal.** `structural_z` beats `structural_delta` by
  ~0.02 AUROC — real and repeatable, but small, and only for `resunet`/`attention`.
- **The learned σ is optimistic.** On clean held-out data the z-score standard
  deviation is 0.40–0.48 where a calibrated model gives 1.0 — roughly 2.5×
  overconfident — and the error/σ rank correlation is near zero: σ does not
  localise where the model is actually wrong.
- **A trivial training-free alternative is better calibrated.** Estimating a local
  noise floor directly from each test image's own residual (a robust, smoothed
  local filter) gives z-score std 0.76–0.83 and roughly a third of the calibration
  error of the learned σ — but it does not improve detection either.

The structural decomposition already handles the "grey-level shift vs. real
structure" separation that σ was meant to help with, and a single per-pixel scale
parameter is too weak an instrument to exploit the one-to-many structure further.

**Keep the heteroscedastic head in the reported results as a tried, honest
negative** — theoretically motivated, empirically marginal — and do not build on
it. The training-free local estimate is the better *uncertainty map* for a
conservator, though it too does not help detection.

---

## 7. The fidelity / structure loss trade-off

The deterministic loss is `alpha·Charbonnier + (1−alpha)·(1−MS-SSIM)`. Sweeping
`alpha ∈ {0.16, 0.50, 0.84}` across the base architectures:

- **`alpha = 0.16` (the default, MS-SSIM-dominated) is correct for detection** on
  `unet` and `attention_unet`. Raising `alpha` improves their pixel fidelity but
  costs detection AUROC — `unet` goes from 0.71 to 0.67 (the floor) at
  `alpha = 0.50`. This is the fidelity/detection tension made concrete: an
  L1-dominated objective optimises absolute level, not the local structure the
  underdrawing signal is made of.
- **`resunet` at `alpha = 0.50` is the one genuine improvement in the sweep** — it
  raises *both* pixel fidelity (MAE 0.114 → 0.110, PSNR +0.4 dB) *and* detection
  (AUROC 0.681 → 0.706, matching `unet`). If a single deterministic model has to
  be picked, this is the configuration to use for `resunet`.
- **`attention_unet` degrades sharply at higher `alpha`** — AUROC drops below the
  `mean(R,G,B)` floor. Its detection strength depends on the MS-SSIM-heavy loss.
- **`alpha = 0.84` is a training-stability hazard** — `resunet` collapsed at that
  weight (SSIM 0.09); the others degraded.

`unet_residual` is a separate case: its `structural_delta` equals the
`mean(R,G,B)` floor at *every* `alpha`. At `alpha = 0.16` the `tanh` residual head
collapses to the identity (the free grey path already satisfies most of an
MS-SSIM-dominated loss, leaving no gradient pressure to build a residual); at
higher `alpha` the head is used and gives passable pixel fidelity, but the signal
it produces is still indistinguishable from a channel mean. The residual head does
not help detection.

---

## 8. Design choices that were tested and did not help

- **Dilated (ASPP-style) bottleneck.** No fidelity or detection benefit over the
  plain bottleneck, and genuine optimisation instability — the dilated variants
  peaked early and then degraded, unlike their non-dilated counterparts. The wide
  concat-then-project block appears to make the loss landscape harder.
- **Two-phase EfficientNet fine-tuning.** Unfreezing the pretrained encoder and
  fine-tuning end-to-end at a low learning rate consistently *underperformed* the
  frozen-encoder baseline on every metric — the paired dataset is too small to
  fine-tune an ImageNet backbone without eroding the features that made it useful.
- **Multi-scale band-pass sharpening of the residual.** Tuning a band-pass to
  stroke width *lowers* AUROC (best band −0.04, most −0.06 to −0.12) while
  inflating stroke coherence non-specifically.
- **Classical decompositions (PCA / ICA on the RGB+IR stack).** No component is
  operationally usable — the informative index is a different one on each
  painting.
- **Loss-weight sweep for the heteroscedastic β.** β = 0.5 is already the
  best-calibrated point; no other value improves it.
- **Dropping the mockups and the grouped split.** An ablation
  (`020_bis_training_generic.ipynb` / `C6_mockups_vs_generic.ipynb`) retrained
  `unet` / `resunet` on a split that excludes every synthetic paint-on-support
  mockup and cuts the remaining real artworks train/val at the pair level, with
  no artwork grouping. On `data/test/` this raised `unet` reconstruction fidelity
  (+0.5 dB PSNR) but left `structural_delta` detection AUROC flat — `unet`
  0.707 → 0.704, `resunet` 0.681 → 0.691, both inside the per-fold noise — with
  no coherent per-image direction. The fidelity gain is expected and
  uninteresting: pair-level splitting leaks each painting's style across
  train/val, so early stopping selects a checkpoint fit more closely to a corpus
  the `data/test/` paintings resemble more than the mockups do — an easier task,
  not a better model. The mockups and the 3-way grouped split are not what limits
  detection. (The `attention_unet` generic run diverged — `val_loss` 0.46 vs.
  ~0.22, stopped at 21 epochs — and its `C6` row is a failed checkpoint, not a
  result.)

---

## 9. The deep model is not the bottleneck

A parameter-free per-image ordinary-least-squares fit `IR ~ [R, G, B, 1]`, with no
deep model at all, scores **AUROC 0.824 on GT01** — the single best per-image
number anywhere in the project — then collapses to 0.55 / 0.51 on GT02 / GT03.
GT01's underdrawing is largely *linearly* separable from RGB; GT02's and GT03's
are subtle enough that only the real IR reveals them. A trivial linear baseline
being in the same ballpark as a 31M-parameter network is direct evidence that the
limiting factor is the data and the problem, not the model.

---

## 10. Recommended models

There is **no single best model** — the two metric families disagree on the
winner, and that disagreement is itself a finding:

| goal | model(s) | signal | number |
|---|---|---|---|
| **hidden-detail detection** (the deliverable) | `resunet_nll` and `attention_unet_nll` — on par | `structural_z`, 3-fold ensemble | AUROC ≈ 0.72 |
| **pixel fidelity** | `resunet` (at loss weight `alpha = 0.50`) | — | MAE ≈ 0.11, PSNR ≈ 19 dB |
| **best deterministic detector** | `unet` or `attention_unet` (default loss weight) | `structural_delta` | AUROC ≈ 0.70 |

For a best-effort detection map, use the **3-fold ensemble** of `resunet_nll`
(or a combined `resunet_nll` + `attention_unet_nll` ensemble), report
`structural_z` *and* `structural_delta`, and state the per-fold spread (±0.01) and
the three-mask sample size honestly.

The heteroscedastic models are recommended for detection not because the σ head is
a large win — it is not (§6) — but because `structural_z` is consistently the top
signal by a small margin and the head is worth reporting as a tried direction.

---

## 11. The ceiling, and what would move it

The architecture axis is saturated; the levers with real upside are all outside
the scope of closing this project:

1. **More ground-truth annotations** (10–15 masks instead of 3). Three masks
   support no statistical claim and no real model selection. This blocks
   everything else.
2. **An IR inpainting model** — reconstruct masked IR regions from RGB plus
   surrounding IR; inpainted-vs-real disagreement flags anomalies. This uses the
   IR's own spatial structure, which a pure RGB→IR model ignores. The single
   model change with the highest plausible upside.
3. **Multispectral acquisition.** Carbon-black underdrawing has a spectral
   signature that one IR band discards. Re-acquiring at a few bands is a larger
   lever than any model change — an instrument change, not a code change.

Not worth pursuing: another architecture; more patches of the same works.

---

## 12. Framing

Stated correctly, the result is a complete and honest story for a course project:

> *RGB→IR translation plus structural residual analysis recovers the underdrawing
> signal at AUROC ≈ 0.72 (3-fold ensemble, three hand-drawn masks). The
> heteroscedastic uncertainty head, though theoretically motivated, adds little
> and its learned scale is poorly calibrated. A trivial per-image linear baseline
> is competitive on the one painting whose underdrawing is linearly separable from
> RGB, which — together with the tiny spread across ten architectures — places the
> ceiling on dataset diversity and single-band IR, not on model capacity.*
