# Hand-drawn ground-truth masks for `data/test/`

Procedure for annotating hidden-detail regions on images in
`data/test/{rgb,ir}/`, so `033_bis_signal_evaluation.ipynb` can score
every candidate signal (raw delta, structural delta, learned z-score,
per architecture and per loss variant) against a real reference with
`scripts/detection.py`, instead of the cross-modal pseudo ground truth
`033_signal_evaluation.ipynb` uses on the unannotated main corpus.

## Why these images specifically

Every image in `data/test/` that has a matching mask is known to carry
hidden detail attributable to a specific, identified cause (underdrawing,
pentimento, reused support, etc.) — this is domain knowledge from outside
the codebase, not something inferred from the images themselves. That is
what makes hand-annotating them worthwhile: unlike the main
training/validation/test corpus (`data/ir`/`data/rgb`, evaluated in
`033`), where the only systematic RGB/IR difference is paint-stroke
thickness rather than documented hidden content (see `evaluation.md`
§3), a mask drawn on these images is a real positive reference, not an
artifact.

`data/test/` also holds other images with no mask (added purely for
visual/qualitative inspection in `051`/`060`/`062`) — those are not part
of `033_bis`'s detection ranking, only the ones with a mask are.

## Mask format

- **PNG**, not JPEG (no lossy compression at a hard 0/255 boundary).
- **Same pixel resolution** as the corresponding `data/test/rgb/<stem>.jpg`
  / `data/test/ir/<stem>.jpg` (they're already the same size as each
  other).
- **Grayscale or RGB**, white (`255`) marks a hidden-detail pixel, black
  (`0`) marks background. Anti-aliased edges are fine — the loader
  thresholds at the midpoint (127).
- Saved as `data/test/annotations/<stem>_Map.png`, e.g.
  `data/test/annotations/GT01_Map.png`. Note the `_Map` suffix (not
  `_mask`, as an earlier draft of this document named it).
- One mask per image is enough even when several causes are present in
  the same painting (underdrawing *and* a pentimento, say) — the
  detection metrics only need "hidden detail here: yes/no", not which
  kind. If separating causes later becomes useful, per-cause masks can
  be added as `<stem>_Map_<cause>.png` without changing the loader
  (`033_bis` would need a small extension to combine or select them).

## How a mask gets used

`033_bis_signal_evaluation.ipynb` scans `data/test/annotations/` for
`*_Map.png` files and scores only the matching `(rgb, ir)` pairs — no
longer the cross-modal pseudo-mask, and no longer every image under
`data/test/`. For each ground-truth image it loads the PNG, thresholds
it to boolean, and passes it straight to `scripts.detection.rank_signals`
alongside the same per-model signal maps `033` computes (raw delta,
structural delta for deterministic architectures; raw delta and `|z|`
for NLL architectures under both `gaussian_nll` and `beta_nll`).

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

Three masks exist: `GT01_Map.png`, `GT02_Map.png`, `GT03_Map.png` (their
`(rgb, ir)` pairs are `data/test/{rgb,ir}/GT01.jpg` etc.).
`033_bis_signal_evaluation.ipynb` was updated to consume them directly —
no separate "annotated" notebook was needed after all, the pseudo-mask
code in `033_bis` was replaced in place. More masks can be added the
same way at any time; the notebook picks up whatever is present in
`data/test/annotations/` without further changes.
