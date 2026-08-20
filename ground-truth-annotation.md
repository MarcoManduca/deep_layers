# Hand-drawn ground-truth masks for `data/test/`

Procedure for annotating hidden-detail regions on the images in
`data/test/{rgb,ir}/`, so `033_bis_signal_evaluation_annotated.ipynb` can
score every candidate signal (raw delta, structural delta, learned
z-score, per architecture and per loss variant) against a real reference
with `scripts/detection.py`, instead of the cross-modal pseudo ground
truth `033_signal_evaluation.ipynb` uses on the unannotated main corpus.

## Why these images specifically

Every image in `data/test/` (`case`, `green`, `modern`, `total`, and any
added later) is known to carry hidden detail attributable to a specific,
identified cause (underdrawing, pentimento, reused support, etc.) — this
is domain knowledge from outside the codebase, not something inferred
from the images themselves. That is what makes hand-annotating them
worthwhile: unlike the main training/validation/test corpus
(`data/ir`/`data/rgb`, evaluated in `033`), where the only systematic
RGB/IR difference is paint-stroke thickness rather than documented hidden
content (see `evaluation.md` §3), a mask drawn on these images is a real
positive reference, not an artifact.

## Mask format

- **PNG**, not JPEG (no lossy compression at a hard 0/255 boundary).
- **Same pixel resolution** as the corresponding `data/test/rgb/<stem>.jpg`
  / `data/test/ir/<stem>.jpg` (they're already the same size as each
  other).
- **Grayscale or RGB**, white (`255`) marks a hidden-detail pixel, black
  (`0`) marks background. Anti-aliased edges are fine — the loader
  thresholds at the midpoint.
- Saved as `data/test/annotations/<stem>_mask.png`, e.g.
  `data/test/annotations/modern_mask.png`. The directory doesn't exist
  yet — create it when the first mask is added.
- One mask per image is enough even when several causes are present in
  the same painting (underdrawing *and* a pentimento, say) — the
  detection metrics only need "hidden detail here: yes/no", not which
  kind. If separating causes later becomes useful, per-cause masks can
  be added as `<stem>_mask_<cause>.png` without changing the loader
  (`033_bis` would need a small extension to combine or select them).

## How a mask gets used

`033_bis_signal_evaluation_annotated.ipynb` loads the PNG, thresholds it
to boolean, and passes it straight to `scripts.detection.rank_signals`
alongside the same per-model signal maps `033` computes (raw delta,
structural delta for deterministic architectures; raw delta and `|z|` for
NLL architectures under both `gaussian_nll` and `beta_nll`). Images
without a mask file are skipped for the detection ranking (with a
message, not a failure) but still contribute to the unsupervised stroke-
coherence axis, which needs no reference at all.

## Free tooling, if a JSON/polygon workflow is preferred instead

A hand-drawn PNG was chosen over polygon annotation because freeform
tracing is simpler for organic underdrawing shapes than placing
coordinates. If a polygon-based workflow becomes preferable later (e.g.
to keep per-region metadata), two free browser tools export to JSON
without installation: [makesense.ai](https://www.makesense.ai/) and the
[VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/).
Converting a polygon export to a mask PNG is a small, separate script —
not written, since the PNG workflow above is the current choice.

## Status

Not yet done for any image — `data/test/annotations/` does not exist.
`033_bis_signal_evaluation_annotated.ipynb` is written to run against
zero, one, or several annotated images, so it becomes progressively more
informative (and statistically meaningful — the rank correlation in its
§6 needs more than one image to mean anything, same caveat as `033`'s
"fail early" check for candidate signals) as masks are added, without
needing changes to the notebook itself.
