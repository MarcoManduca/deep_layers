# Final comments — how to improve the deliverable, and where the ceiling is

_Written 2026-08-27, after `C4_adaptive_sigma.ipynb`. Strategic assessment of what would
actually move the project's deliverable (the residual/signal map that reveals
underdrawings, pentimenti, and reused supports), given the physical priors and the
dataset constraints._

## The underlying diagnosis

We predict IR from RGB (`μ`), subtract it from the real IR, and analyse the residual —
but **the signal we care about is, by construction, the part of the IR that RGB cannot
predict.** So detector quality depends on only two things:

1. how well `μ` predicts the *surface* IR — a cleaner residual makes the anomaly stand out;
2. how well we separate "`μ` is wrong because of an underdrawing" from "`μ` is wrong
   because RGB→IR is genuinely ambiguous here / craquelure / model imprecision".

Six-plus architectures (`unet`, `resunet`, `attention_unet`, `unet_v2`,
`unet_restormer`, `efficientnet_unet`, the dilated variants) all land at psnr 16–18.5,
ssim 0.43–0.56, detection AUROC 0.65–0.71. The spread across architectures is tiny.
**This says the "architecture" axis is saturated — the bottleneck is data and problem
formulation, not model capacity.**

Current best detection signal: `structural delta` (deterministic) / `structural z`
(NLL), AUROC ~0.70, measured on **3 hand-drawn masks** (`GT01`/`GT02`/`GT03`).

## Is a different architecture the answer?

**No.** It would move psnr by ~0.3 dB and AUROC by ~0.02. The only architectural
direction with real potential — modelling the *pigment/material* rather than RGB pixels
— needs pigment labels we do not have, so it collapses back to "better features" = data.

## Is a better training dataset the answer?

Yes, but not "more patches of the same works" (the case diversity is genuinely small).
In order of real leverage:

1. **More ground-truth annotations** (10–15 masks instead of 3). Three masks do not
   support *any* statistical claim. Even 10–15 would enable a defensible number and real
   model selection. This is the constraint that blocks everything else — worth more than
   another training round.
2. **The mockups may be hurting.** They inject "similar RGB / different IR / no hidden
   detail" pairs, which teach `μ` that RGB is unreliable → `μ` becomes conservative and
   blurry → residuals inflate everywhere → the anomaly is harder to isolate. Worth an
   ablation: **train `μ` without mockups** and check whether detection contrast on the
   real test paintings improves. (C4 §7 — the learned `σ` is ~2.5× overconfident even on
   clean val — suggests the model has *not* over-learned ambiguity, but this should be
   verified directly.)
3. **The real ceiling is single-band IR.** Carbon-black underdrawing has a specific
   spectral signature; one IR image discards the discriminative spectral information. If
   acquisition can be redone at a few bands (~1000 / 1500 / 2000 nm), that is a larger
   lever than any model change. Instrument change, not a code change.

## A completely different approach?

The interesting options, in cost/benefit order:

1. **Multi-scale band-pass of the residual**, tuned to stroke width. Underdrawing
   (strokes, contours), craquelure (high frequency), and substrate/lighting shifts (low
   frequency) live in distinct spatial-frequency bands. The structural-SSIM decomposition
   is already a version of this; an explicit band-pass is **cheap, needs no retraining**,
   and targets the signal directly. Try it as a new signal in the `C`-series.
2. **Classical per-image baseline (ICA / PCA on the RGB+IR stack).** This is what
   conservators actually use — no training, per-image, no train→test transfer problem.
   If an ICA of `[R, G, B, IR]` separates the underdrawing as a component as well as the
   0.70-AUROC deep model, it is both a humbling baseline and arguably a *better*
   deliverable (interpretable). **Should be run as a comparison** — it may already beat
   the deep approach on these specific test images.
3. **IR inpainting model.** Train a model to reconstruct masked IR regions from RGB +
   surrounding IR. At test, inpainted-vs-real disagreement flags anomalies. This uses the
   IR's own spatial structure, which the pure RGB→IR model ignores; underdrawings are
   spatially coherent strokes an inpainter trained on hidden-detail-free data would not
   reproduce. Medium cost, plausibly the single biggest improvement from a model change.
4. **Conditional density model (normalising flow / diffusion).** Model the full
   distribution `p(IR | RGB)` and use the log-likelihood of the observed IR patch as the
   anomaly score — the principled tool for "RGB→IR is one-to-many and I want to flag
   improbable IR". The heteroscedastic head is the 1-parameter special case of this. But
   it is data-hungry; with ~860 patches a full flow/diffusion probably will not hold.

## Is the heteroscedastic head useless for the project?

**For the detection deliverable: essentially yes.** C4 showed it adds ~0 to AUROC and
its learned `σ` is poorly calibrated (z_std 0.40–0.48 on clean held-out data, ~2.5×
overconfident; a trivial training-free adaptive local filter reaches 0.76–0.83). The
theoretical justification (RGB→IR is one-to-many) is *correct*, but a single per-pixel
scale parameter is too weak an instrument to exploit it, and the structural decomposition
already handles the "substrate/lighting shift vs. real structure" separation that `σ` was
meant to help with.

Where it is **not** useless:

- As a tried-and-reported negative result — genuine scientific value in the write-up
  ("theoretically motivated, empirically marginal, here is why").
- As the trivial special case of option 4 (conditional density) — the *idea* survives,
  the *implementation* does not.
- The training-free adaptive `b` (C4 §7) beats it as an uncertainty map for the
  conservator.

Keep it in the write-up with the negative result; do not build on it.

## Prioritised recommendation

**If the goal is to close with an honest deliverable:**

1. **More GT annotations** (10–15). Unlocks the ability to *claim* anything; everything
   else is unmeasurable without it.
2. **Multi-scale band-pass of the residual** — new signal, no retraining, ~half a day.
3. **Classical ICA / PCA per-image baseline** — run it and compare; may be competitive.

**If there is appetite for one model experiment:**

4. **`μ` without mockups** (ablation) — test whether the mockups hurt contrast.
5. **IR inpainting model** — the model change with the highest upside.

**Out of scope for closing, but the honest "with more time" answer:** multispectral IR
acquisition (the true ceiling); conditional normalising flow.

**Not worth it:** another architecture; more patches of the same works.

## Update — C5 (2026-08-27): the signal-processing avenue is closed

`C5_signal_sharpening.ipynb` tried the two training-free levers above (points 1–2 of
"a completely different approach"). Result: **nothing beats `structural delta`.**

- Multi-scale band-pass of the residual *lowers* AUROC (best band −0.04, most −0.06 to
  −0.12) while inflating stroke coherence — it amplifies oriented structure in general,
  not the underdrawing. Coherence is only a valid tie-breaker between signals of
  comparable AUROC.
- PCA / ICA on `[R,G,B,IR]`: no component is operationally usable — the informative one
  is a different index on each painting (`pca c3`: 0.82 on GT01, 0.43 on GT03).
- **The one carry-forward finding**: `linreg residual` — a parameter-free per-image OLS
  of `IR ~ [R,G,B,1]`, no deep model — scores **0.824 on GT01** (the single best
  per-image AUROC in the experiment), then collapses to 0.55 / 0.51 on GT02 / GT03.
  GT01's underdrawing is largely *linearly* separable from RGB (the deep `μ` was not
  needed there); GT02/GT03's are subtle enough that only the real IR reveals them.
  **This is direct evidence that the deep model is not the limiting factor** — a trivial
  linear baseline is in the same ballpark on this data. The bottleneck is the
  data/problem, per this document's thesis.

Detection signal is therefore fixed as `structural delta` (deterministic) /
`structural z` (NLL). The remaining levers with real upside are the ones this document
already lists as out-of-scope-for-closing: more GT annotations, an IR inpainting model,
and multispectral acquisition.

## Framing note

AUROC ~0.70 on 3 masks, framed correctly, may already be a defensible course-project
result: _"RGB→IR translation plus structural residual analysis recovers the underdrawing
signal at AUROC ~0.70; the heteroscedastic uncertainty head, though theoretically
motivated, adds nothing; the ceiling is set by single-band IR and dataset diversity, not
model capacity."_ That is a complete, honest story. The task may be to frame what exists
rather than to improve the number.
