# Ground-truth mask for detection-metric evaluation (`modern`)

Procedure for annotating `data/test/rgb|ir/modern.jpg` with a binary
ground-truth mask, to be used as the basis for a detection metric
(AUROC/precision-recall) that scores hidden-detail signals (raw delta,
structural delta, learned z-score, ...) against what `modern` was
actually built to contain — see `code-review.md` §7.6 "Update:
evaluating the learned z-score itself" and `note.md`'s matching section
for the motivation (why `mae`/`ssim`/`psnr`, which only compare `mu`,
can't answer this).

Not yet implemented — decision on whether/when to build the actual
detection-metric notebook is still open.

## 1. Create the mask image

- Trace, by hand, the region(s) of `modern` known to contain the
  intentionally hidden detail (it's a real test image built ad hoc for
  this purpose, so the ground truth is known by construction, not
  inferred).
- **Resolution**: ideally the same pixel dimensions as
  `data/test/ir/modern.jpg` (or `rgb/modern.jpg`, same size). If the
  annotation tool forces a different canvas size, resize the mask back
  to that exact resolution before use — avoid resizing the source image
  instead, to not lose ground-truth precision at the edges.
- **Format**: PNG, not JPEG. JPEG compression blurs edges and turns a
  binary mask into a gradient of near-white/near-black pixels, which
  breaks a clean 0/255 threshold.
- **Color convention**: white (255) = hidden-detail area, black (0) =
  everything else. If it's useful later to distinguish detail *types*
  (e.g. pentimento vs. reused support), use distinct gray levels instead
  of a strict binary — not needed for a first pass.
- **Tooling**: any local image editor with a pen/lasso tool works
  (Preview, GIMP, Photoshop, ...). No new dependency needed.
  - If a browser-based annotation tool is preferred instead:
    **makesense.ai** (free, no login, runs client-side, brush/polygon
    tools, exports both mask images and JSON/COCO) or **VGG Image
    Annotator (VIA)** (single self-contained HTML file, polygon/box
    oriented, exports JSON) are common no-install options.
- **Save to**: `data/test/annotations/modern_mask.png` (new directory;
  `data/test/` is already gitignored, so this won't need a separate
  ignore rule).

## 2. Load the mask

```python
mask = np.array(Image.open("data/test/annotations/modern_mask.png").convert("L"))
mask = mask > 127  # binary bool array, shape (H, W)
```

## 3. Score candidate signals against it

For each signal already produced by the existing pipeline — raw delta,
structural delta (`scripts.delta_analysis.analyze_delta`), learned
z-score (`gaussian_nll` and `beta_nll`, via `_predict_mu_sigma` as used
in `062_model_comparison_v2.ipynb`) — flatten signal and mask to 1D and
compute:

```python
from sklearn.metrics import roc_auc_score, average_precision_score

auroc = roc_auc_score(mask.ravel(), signal.ravel())
ap = average_precision_score(mask.ravel(), signal.ravel())
```

- **AUROC**: how well the signal separates "hidden detail" pixels from
  "normal" pixels, threshold-independent. Primary metric.
- **Average precision (AUPRC)**: more informative than AUROC if the
  annotated area is a small minority of the image (likely, if the mask
  covers a few localized regions) — class imbalance inflates AUROC.
- Repeat for every (signal × model/loss-variant) combination to get one
  comparable table instead of only a visual/qualitative read.

`scikit-learn` would need to be added as a dependency if not already
present (check `requirements.txt`/`env/environment.yml` first).

## 4. Where this would live

A new lightweight notebook (or a new section in `062`) that: loads the
mask, computes the signals already produced by existing code
(`analyze_delta`, `_predict_mu_sigma`), scores each against the mask,
prints a comparison table. No new modeling code — only the mask load +
scoring step is new.
