# Evaluation strategy — what each pipeline measures, and what it doesn't

This note collects, in one place, every evaluation pipeline built across
`feature/heteroscedastic-nll` and `feature/uncertainty-evaluation`, and
records a discussion (2026-08-19) that changed how two of them should be
interpreted. It complements `note.md` (design rationale for the delta
analysis and the pseudo-mask) rather than replacing it.

## 1. The pipelines, in the order they were built

| # | Where | What it measures | Against what | Data used |
|---|---|---|---|---|
| 1 | `030_evaluation.ipynb` / `031_evaluation_nll.ipynb` / `032_evaluation_v2.ipynb` (mae/ssim/psnr tables) | How well `mu` (or the deterministic output) reproduces `real_IR` | `real_IR` | test fold of `data/ir`/`data/rgb`, grouped or mockup-aware split matching training |
| 2 | `scripts/calibration.py`, wired into `032_evaluation_v2.ipynb` §4b | Whether the learned `sigma` is *trustworthy* — coverage probability, ENCE reliability curve, Spearman(`\|real-mu\|`, `sigma`), z-score moments, Gaussian NLL, sharpness/dispersion | the Gaussian assumption itself, not any hidden-detail reference | same test fold as above |
| 3 | `scripts/contrast.py` (`ZScale`), wired into `plot_zscore`/`plot_delta_comparison`, swept in `062_model_comparison_v2.ipynb` §2c | Display contrast of the z-score maps (fixed / percentile / percentile+gamma) | — (a rendering choice, not a metric) | whatever image is being plotted |
| 4 | `scripts/pseudo_mask.py` + `scripts/detection.py` + `scripts/stroke_stats.py`, run in `033_signal_evaluation.ipynb` | AUROC/AP of every candidate signal (raw delta, structural delta, `\|z\|`) against a cross-modal pseudo ground truth (§2 below), plus an unsupervised stroke-coherence axis, plus their rank correlation | *"structure present in IR and not explained by RGB"* (pseudo_mask) and *"oriented, elongated structure vs. isotropic noise"* (stroke_stats) | test fold of `data/ir`/`data/rgb` (see §3 below — **not** `data/test/`) |
| 5 | `050_delta_analysis.ipynb`, `051_delta_analysis_nll.ipynb`, `060_model_comparison_nll.ipynb`, `062_model_comparison_v2.ipynb` (qualitative panels) | Visual inspection of delta/z-score maps, side by side across signal types and/or architectures | a human's read of the image, informally against the known construction of `modern` when that image is used | `data/test/{rgb,ir}/{case,green,modern,total}.jpg`, IMAGE_STEM-selectable |
| 6 | `scripts/pseudo_mask.py` + `scripts/detection.py` + `scripts/stroke_stats.py`, run in `033_bis_signal_evaluation.ipynb` | Same as row 4 — AUROC/AP against the cross-modal pseudo-mask, plus stroke coherence, plus their rank correlation. **No hand-drawn mask** — same data-derived reference as `033`, not the annotated one from §2 | same as row 4 | `data/test/{rgb,ir}/{case,green,modern,total}.jpg` (all available pairs) |

Rows 1–3 all read `mu`/`sigma` fidelity to the model's own prediction
target; they say nothing about whether the residual reveals hidden
content. Row 4 was built specifically to close that gap without hand
annotation. Row 5 is where `modern` — the one image with a fully known,
deliberately placed ground truth — is actually looked at.

## 2. Discussion: is `modern` usable as a quantitative ground truth?

`note.md`'s original objection to scoring `modern` numerically (AUROC
against a hand-drawn mask) rested on three points: no statistical power
(n=1), construction bias, and material mismatch.

**Correction from this discussion (2026-08-19):** the "no statistical
power" framing was imprecise — the actual value of `modern` is that its
ground truth is *known with certainty*, which no real artwork can offer.
The other two objections were re-examined and substantially weakened:

- **Construction bias**: `modern` was built to simulate a real
  contemporary artwork, created before the pipeline existed, with only
  the project's general goal in mind (not tuned against the pipeline's
  specific inductive biases). The hidden details simulate how a real
  contemporary artist would actually work, not a synthetic pattern
  designed to be easy to detect.
- **Material mismatch**: the pigments used to build `modern` are
  represented in the training data — the training corpus already
  includes both historical and modern/contemporary pigments, so `modern`
  is not physically out of distribution for the model.

**Conclusion**: `modern` is worth scoring quantitatively (AUROC /
average precision / the `033`-style ranking, against its own hand-drawn
mask) and should be treated as a **positive control with an upper-bound
reading** — if a signal fails here, the pipeline is broken; if it
succeeds, that is evidence (not proof) that it also works on real
paintings, and a useful check against learning pathologies (e.g. a
signal that degrades on `modern` after a training change is a red flag
even before any real-painting result is available). It does not, on its
own, establish that a signal generalizes to genuine historical
pentimenti — no image can establish that alone — but the two reasons
that previously limited its usefulness are addressed.

**Action implied**: the follow-up marked "SUPERSEDED" in `note.md` (hand
annotation + AUROC on `modern`) should be revived, not dropped — see
`ground-truth-annotation.md` for the existing procedure. It complements
`033`/`033_bis`, it does not compete with them (see §3/§4a). It is
explicitly **not** what `033_bis` does (§4a below) — it is left for a
separate, dedicated notebook, not yet built.

## 3. Discussion: is the main-corpus test fold (`033`) indicative of hidden-detail detection?

`033_signal_evaluation.ipynb` runs on the test fold of `data/ir`/`data/rgb`
— the corpus used for train/val/test, not `data/test/`. The working
assumption when `033` was built (and initially repeated in this
discussion, based on `README.md`'s framing of the project) was that this
corpus consists of real reflectography pairs acquired specifically to
examine hidden underdrawings, so residual structure there is a legitimate
(if noisy) hidden-detail signal.

**Correction from this discussion (2026-08-19), from the domain side**:

- Canvas/support texture is explicitly *not* a target of interest (this
  was already known — `note.md`'s "superset" caveat).
- The artworks in the main corpus (`data/ir`/`data/rgb`) do **not**
  contain documented underdrawings, restorations or pentimenti.
- The only systematic RGB/IR difference in that corpus is paint stroke
  **thickness**, which changes local IR transparency — an optical/material
  effect, not hidden content.

**Conclusion**: the pseudo-mask's "structure in IR not explained by RGB"
does not identify hidden detail on this fold, because there is no hidden
detail there to find, by construction of the dataset. This is a stronger
statement than `note.md`'s existing "superset" caveat (which says the
pseudo-mask over-counts real content plus artifacts) — on this
particular fold, the mask counts *only* an artifact (stroke-thickness/
transparency structure), not real content at all. `033`'s numeric
ranking is therefore **not evidence about hidden-detail detection
ability** as currently framed in the notebook's own markdown ("ranks
every hidden-detail signal... without a hand-drawn mask").

This does not make `033` worthless — see §4a/§5 below — but it does mean
its result needs to be re-labeled and its role in the overall evaluation
strategy reconsidered.

## 4a. `033_bis`: the same pseudo-mask logic, on `data/test/`

`033_bis_signal_evaluation.ipynb` is **not** the hand-annotated
evaluation implied by §2/§3 — that stays a separate, not-yet-built
notebook (`ground-truth-annotation.md`). `033_bis` runs the exact same
machinery as `033` (`scripts.pseudo_mask.cross_modal_pseudo_mask`,
`scripts.detection.rank_signals`, `scripts.stroke_stats`) unchanged, on a
different image set: every pair in `data/test/{rgb,ir}/` instead of the
main-corpus test fold.

The reason this is still worth running, given §3's finding that the
main-corpus fold has no documented hidden content: `data/test/` images
*do* — each one is known, from outside the codebase, to carry hidden
detail from a specific, identified cause (underdrawing, pentimento,
reused support). The pseudo-mask's "structure in IR not explained by
RGB" is measuring an artifact on the main corpus (§3) but a real,
if imprecise (still a superset — canvas/support texture scores too),
phenomenon here. Same construction caveat as always: the pseudo-mask
shares its form with the structural delta, so cross-signal-type
comparisons still need §6 (rank correlation with stroke coherence), not
§4 alone.

Only 4 images currently (`case`, `green`, `modern`, `total`) — the
`std` bars in its ranking will be wide until more are added, which
requires no notebook change (it globs `data/test/rgb/*.jpg` at run
time).

## 4b. Known issue: `033`'s test fold does not match training

`033_signal_evaluation.ipynb` builds its evaluation set with:

```python
pairs = load_image_pairs(settings.IR_DIR, settings.RGB_DIR)
_, _, test_pairs = grouped_train_val_test_split(pairs, ...)
```

i.e. only the plain, per-artwork split. `030_evaluation.ipynb`,
`031_evaluation_nll.ipynb` and `032_evaluation_v2.ipynb` all carry a
second block, absent from `033`, that overwrites `test_pairs` with
`mockup_aware_train_val_test_split(...)` — the split actually used to
train the checkpoints (`020`/`021`/`022_training_*.ipynb`), with an
inline comment warning that mixing the two splits leaks train data into
the reported test metrics.

Checked locally (2026-08-19): with the notebook's current
`N_EVAL = 6`, the 6 evaluated pairs are all sections of `natmorta1` (a
real, non-mockup artwork) and are confirmed absent from the
mockup-aware train/val set — **no leakage in the currently committed
results**. But the risk is latent, not resolved: the full plain-split
test fold is

```
trosso     60  ← mockup group
tbruno     47  ← mockup group
natmorta1  35
santo      30
q2         20
```

`tbruno`/`trosso` are 2 of the 6 mockup groups (`settings.MOCKUP_ARTWORK_IDS`)
and, under the mockup-aware split the checkpoints were actually trained
with, are ~95% in train/val rather than held out. Raising `N_EVAL` past
35, reordering, or picking images by hand would silently start scoring
models on data they were trained on. A secondary, independent issue: at
`N_EVAL = 6` every evaluated pair comes from a single artwork, so even
setting leakage aside the sample has no cross-artwork diversity.

**Fix**: add the missing mockup-aware block to `033`, mirroring
`032_evaluation_v2.ipynb`, before trusting any number it produces.

## 5. `033_bis` results (2026-08-19, real run, 4 images: `case`/`green`/`modern`/`total`)

28 candidate signals (6 deterministic architectures × {delta, structural},
4 NLL architectures × {gaussian, beta} × {delta, `|z|`}). Full tables in
`notebooks/033_bis_signal_evaluation.ipynb` §4/§5/§6.

**A sharp two-tier split, not a fine-grained ranking.** All 6
`* structural` signals cluster at **AUROC 0.76–0.81**; every other
signal — every raw delta and every NLL `|z|`, gaussian or beta — clusters
at **0.53–0.59**, barely above chance (0.5), with overlapping std bars
(0.04–0.08). The gap between the two tiers (~0.2 AUROC) dwarfs the
spread within either tier.

- Read the top tier with the construction caveat already documented
  (§4a, `note.md`): the pseudo-mask and the structural delta share the
  same local-SSIM-structure form, so this split is partly the expected
  artifact, not proof structural delta is "better" in some absolute
  sense.
- What is **not** an artifact: stroke coherence (§5), built on a
  structure-tensor, shares no construction with the pseudo-mask, and
  ranks the same 6 structural signals on top (0.29–0.33) with everything
  else lower (0.17–0.24). Spearman(auroc, coherence) = **0.523** over 28
  signals — a moderate, independent agreement between two references
  that fail in different ways. That corroboration is real, even though
  the absolute AUROC numbers for the top tier should still be read with
  the construction caveat in mind.
- **Within the structural tier** (the only comparison the construction
  caveat doesn't undermine): `unet_restormer` and `unet_v2` tie for best
  (0.812 each), ahead of `resunet`/`efficientnet_unet`/`attention_unet`/
  `unet` (0.758–0.786). This corroborates `032`'s `mu`-fidelity finding
  (`unet_v2`/`unet_restormer` beat `unet` on mae/psnr/ssim) with an
  independent, hidden-detail-oriented reference.
- **The NLL head shows no detection edge here.** `|z|` does not
  separate from raw delta, and `beta_nll` does not separate from
  `gaussian_nll` — all 16 NLL signals sit in the lower tier, indifferent
  from one another within noise. `resunet_nll (beta)` is last on both
  axes. See §6 below for why this looks like it contradicts the
  qualitative read in `060`/`062`, and why it likely doesn't.

## 6. Reconciling `033_bis` with the qualitative read in `060`/`062`

`062_model_comparison_v2.ipynb`'s qualitative read (recorded earlier: the
`beta_nll` z-score "picks out most of the intended hidden details ...
just under-contrasted") reads as a clear win for the z-score. `033_bis`
shows no such win on the same signal, quantitatively. Both can be
correct at once — they are not measuring the same thing, for three
independent reasons.

**Contrast is not the missing piece — AUROC cannot see contrast at
all.** AUROC is computed from the *rank order* of pixel values (it's the
probability a random positive pixel outranks a random negative one), and
rank order is invariant under any strictly increasing transform of the
signal. `scripts/contrast.py`'s gamma compression is exactly such a
transform; its percentile clipping can only *tie* pixels together, which
cannot improve AUROC (sklearn breaks ties conservatively) and never
improves rank separation. **Stretching the display contrast of the
z-score, at any percentile or gamma, cannot move its AUROC** — if the
raw `|z|` values don't separate hidden-detail pixels from background
pixels in rank, no contrast setting fixes that, because contrast is a
purely visual remapping downstream of the same ranking. If the z-score
is worth keeping, the fix has to change the *signal itself* (e.g.
combining it with the structural component, spatial smoothing, a
different normalization) — not how it's displayed.

**Three real (non-contrast) reasons the two evaluations can diverge:**

1. **Global vs. local judgment.** `033_bis`'s AUROC scores every pixel
   in the image against the pseudo-mask — a large area, most of it
   background. A human looking at `062`'s figure focuses on the one
   region they already know to check, and doesn't "count" noise
   elsewhere in the frame against the signal the way a global metric
   does. A z-score that is genuinely striking on one known region while
   noisy everywhere else can look convincing and still score near chance
   overall.
2. ~~Averaged across 4 images, not read per-image~~ — **checked and
   ruled out** (2026-08-19, see below). This reasoning predicted a
   strong `modern` result diluted by the other three images. The
   per-image breakdown shows the opposite.
3. **Different questions.** The pseudo-mask marks a superset — every
   local IR structure the RGB doesn't explain, everywhere in the image
   (§4a's caveat, restated). `062`'s read asks "does the z-score reveal
   the specific features I placed in `modern`?" — a narrower, targeted
   question. `033_bis` asks "does the z-score rank the *entire*
   pseudo-mask above background?" — a broader, blunter one. A signal can
   answer the narrow question well and the broad one poorly at the same
   time.

### Per-image breakdown (2026-08-19) — point 2 above was wrong

A per-image AUROC cell was added to `033_bis` (§4, after the ranking
table) and run. Result: `modern` is not diluting a strong score — it is
the **weakest image for nearly every signal**, structural and
delta/`|z|` alike (structural: 0.68–0.77 on `modern` vs. 0.77–0.86 on the
other three; several delta/`|z|` signals — e.g. `attention_unet_nll
(beta) |z|` at 0.429, `resunet_nll (beta)` at ~0.446,
`unet_restormer delta` at 0.456 — score **below 0.5 on `modern`
specifically**, i.e. worse than chance, while scoring normally on the
other three images).

Several unrelated signals (different architectures, different losses,
both delta and `|z|`) all degrading on the *same single image* is the
signature of a shared-reference problem, not sixteen independent
model-signal problems. The shared reference for all of them is the
cross-modal pseudo-mask. `modern` is a purpose-built contemporary
mock-up; if its RGB/IR relationship departs from the pattern the
pseudo-mask's heuristic (IR contrast × un-explained-by-RGB correlation)
was implicitly shaped around — real, aged paintings, mostly — the top-5%
pseudo-mask on `modern` may simply be marking different pixels than
where the real, deliberately placed hidden details actually are. That
would produce exactly this: several models scoring worse than chance
against a reference that is itself pointed at the wrong pixels, on one
specific image, while behaving normally elsewhere.

This is a testable claim, and `modern` is the one image in the whole
project where it actually can be tested: it is the only one with hidden
detail placed by a known process rather than merely known-to-exist. A
hand-drawn mask on `modern` (`ground-truth-annotation.md`, not yet
started) would settle it directly — if the hand-mask ranking is
substantially better than the pseudo-mask ranking shown here, the
pseudo-mask is the problem on this image, not the signals; if it stays
weak even against the real mask, the signals themselves are what's
failing on `modern`. **Net effect: annotating `modern` first, rather than
starting the annotation effort elsewhere, is now the higher-priority
next step** — it is both the original follow-up (§2) and the direct
diagnostic for this finding.

**Net**: this isn't necessarily a contradiction to resolve by tuning
display contrast (contrast cannot change AUROC at all, see above) — and
it isn't simply an averaging artifact either (point 2, ruled out). The
live hypothesis is that the pseudo-mask reference itself is unreliable
specifically on `modern`, and annotating `modern`'s real mask is the way
to find out.

## 7. Net effect on the evaluation strategy

- `modern` (row 5, quantified — future dedicated notebook, not yet
  built) becomes the primary source of evidence for "does this signal
  reveal hidden detail", once it is scored quantitatively — not a
  fallback kept only for sanity-checking.
- `033`'s pseudo-mask/stroke-coherence pipeline, run on the main corpus,
  should be reframed as measuring **residual structure sensitivity /
  stroke-thickness sensitivity**, not hidden-detail detection. It may
  still be useful for that narrower, real purpose (e.g. comparing how
  "clean" vs. "noisy" different architectures' residuals are on real
  paint), but the notebook's framing and any conclusions drawn from it
  need updating to reflect what it actually measures.
- `033_bis` (row 6, §4a), the same pseudo-mask logic run on `data/test/`
  instead, sits in between: still a data-derived (not hand-drawn)
  reference, but on images where the "structure in IR the RGB can't
  explain" premise actually holds. Its ranking is closer in spirit to
  hidden-detail detection than `033`'s, though still a superset (canvas/
  support texture scores too) and still small-sample until more `data/test/`
  images are added.
- Calibration (`scripts/calibration.py`) and contrast control
  (`scripts/contrast.py`) are unaffected by this discussion — they don't
  make any hidden-detail claim, only a `sigma`-quality claim.
