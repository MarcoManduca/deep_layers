# Fixing Plan — Deep Layers

Scope: action plan for the ranked findings surfaced by `theory-links.md` §8.4-8.10 (course-lecture-to-code review). Each finding is restated with its priority, the type of change it requires, which checkpoints it forces a retrain of, and the reasoning behind the sequencing chosen to minimize the number of full retrain passes. Written before any code change — this is a plan document, not a changelog.

---

## 1. Findings, priority, and change type

Ranked most to least severe, as in `theory-links.md` §8.4-8.10.

| # | Finding | Priority | Change type | Retrain scope |
|---|---|---|---|---|
| 1 | `BatchNorm` with `BATCH_SIZE=8` (18 BatchNorm layers in `unet`, batch size ~10-30x below the 50-256 range two lectures cite as typical) | Highest — best-evidenced (two independent lectures flag the same parameter) | Architectural — replace `BatchNormalization` with `GroupNormalization` | All BatchNorm-using architectures |
| 2 | No weight decay anywhere (`kernel_regularizer`, `regularizers`, `Adam(weight_decay=...)` all absent) | High | Hyperparameter — add `kernel_regularizer` or `Adam(weight_decay=...)` | All architectures |
| 3 | All 23 conv layers use `GlorotUniform` (Xavier) with ReLU activations — the literature's own stated bad pairing | Medium | Hyperparameter — `kernel_initializer="he_normal"` | All architectures (mitigated by BatchNorm/GroupNorm, so lower urgency than #1/#2 despite being "free") |
| 4 | No cross-validation — single fixed split, point-estimate metrics only | Medium | Evaluation methodology — grouped k-fold on top of the existing artwork-grouped split | K trainings per model included, not architecture-specific |
| 5 | `resunet.py`'s residual shortcut is an unconditional learned projection (`Conv2D 1×1` + BatchNorm on every block), not the pure identity path the "uninterrupted additive path" argument (L06 slides 77-78) requires | Medium | Architectural — identity shortcut where channels already match | `resunet`, `resunet_nll` only |
| 6 | Two-phase EfficientNet fine-tuning unused (`freeze_encoder=False` exists, never set) | Lower | Training procedure — unfreeze encoder in a second training phase | `efficientnet_unet`, `efficientnet_unet_nll` only |
| 7 | Dilated convolution never tried — the standard way to widen receptive field without downsampling, which is this project's core architectural tension | Lower | Architectural — new variant, `dilation_rate` on selected conv layers | New variants only, not a fix to an existing model |
| 8 | Documentation-only gaps (heteroscedastic-head universality/manifold argument, annotation-budget rationale, augmentation label-preservation principle — none written up) | Lowest | Documentation only | None |

---

## 2. Retrain-minimizing plan

The naive approach — one retrain pass per finding — would mean up to 6 full passes over the checkpoint tree. Findings #1-#3 and #5 are architecturally/hyperparameter-compatible (none undoes another's effect), so they are bundled into shared retrain rounds instead. #4, #6, #7 are kept as separate rounds because they either multiply cost (#4), are scoped to a disjoint set of models (#6), or are exploratory additions rather than fixes (#7).

### Round 0 — Pilot validation (small, before committing to a full pass)

- **Models**: `unet`, `unet_v2` only.
- **Change**: #1 (`BatchNormalization` → `GroupNormalization`) in isolation.
- **Why first and alone**: #1 is the highest-priority finding but also the riskiest — it is the only one of the bundle that changes how activations are normalized at every layer. Validating it cheaply on two representative deterministic architectures before committing #1 across the whole tree avoids discovering a regression only after 12 models have been retrained.

### Round 1 — Combined "recipe v2" retrain

- **Models**: `unet`, `unet_v2`, `resunet`, `attention_unet`, `unet_restormer`, and their `_nll` gaussian/beta counterparts — 12 of the 14 current checkpoints (all except the two EfficientNet-based models, handled in Round 2).
- **Changes bundled in one pass**:
  - #1 `BatchNormalization` → `GroupNormalization` (`num_groups=32`, uniform across these architectures — see §3 below for why)
  - #2 weight decay (`Adam(weight_decay=...)` or `kernel_regularizer`)
  - #3 `kernel_initializer="he_normal"`
  - #5 identity shortcut in `resunet`/`resunet_nll` (bundled here since those two models are already being retrained in this pass)
- **Why bundled**: none of these changes interacts adversarially with another — weight decay and initializer changes only affect the optimization trajectory, not the forward-pass structure that #1 changes. Splitting them into three separate full retrain passes would triple the compute cost for no gain in signal, since the plan does not require isolating each one's individual marginal contribution before shipping.
- **`unet_restormer` exception**: already uses `LayerNormalization` inside `RestormerBlock` (a Transformer-derived block, where LayerNorm is the established choice — see §3). Left unmodified by #1; only #2/#3 apply to it in this round.

### Round 2 — EfficientNet, handled separately

- **Models**: `efficientnet_unet`, `efficientnet_unet_nll`, plus a new `efficientnet_unet_ft` variant (see below).
- **Why separate from Round 1**: `efficientnet_unet(_nll)` uses a pretrained EfficientNetB0 encoder. Applying #1 to the encoder's BatchNorm layers would destroy the pretrained batch statistics, defeating the purpose of transfer learning — so #1 is scoped to the custom decoder only (`decoder_filters = [256, 128, 64, 32, 16]`, see §3 for the group-count implication).
- **Changes**: #1 (decoder only), #2, #3 applied to the decoder; #6 (two-phase fine-tuning) applied as a second training phase.
- **On saving the two-phase fine-tuning as a separate checkpoint** (user question, resolved here): yes — save the frozen-encoder result as `efficientnet_unet` (current behavior, updated with the recipe-v2 changes) and the unfrozen second-phase result as a distinct new checkpoint, e.g. `efficientnet_unet_ft`, rather than overwriting one with the other. This lets `052_model_comparison.ipynb` compare frozen vs. fine-tuned directly instead of losing the frozen baseline.

### Round 3 — Dilated convolution (exploratory)

- **Models**: new variants of `unet`, `unet_v2` (as requested), saved under distinct names (e.g. `unet_v2_dilated`) rather than overwriting the Round 1 checkpoints.
- **Why its own round**: #7 is a new architectural direction, not a fix to an existing design choice — it needs to be evaluated on its own against the Round 1 baseline, not conflated with the BatchNorm/weight-decay/init changes.

### Round 4 — Cross-validation (#4), last

- **Models**: limited to the 2-3 best-performing architectures resulting from Rounds 1-3 (not the full tree), to bound the combinatorial cost — k-fold multiplies training cost by k per model included.
- **Why last**: k-fold validation should run against the final recipe, not be repeated every time an upstream change (Round 1-3) is introduced. Running it first would mean re-running the whole k-fold sweep after every subsequent architecture change.

### Outside the rounds — no retrain

- **#8** (documentation gaps): can be written up at any time, in parallel with the rounds above; blocks nothing and is blocked by nothing.

---

## 3. GroupNorm vs. LayerNorm, and group-count implications

**Why replace BatchNorm rather than increase batch size**: images in this project run up to ~3700×2800px (whole-image inference via `pad_to_multiple` + `model.predict()`, no tiling); a batch size in the 50-256 range cited as typical would likely exceed available GPU memory at that resolution, and raising batch size also requires re-tuning the learning rate (conventionally scaled with batch size), turning a one-line fix into a new hyperparameter search — while still requiring a full retrain either way. Replacing the normalization layer fixes the root cause (BatchNorm's per-batch statistics are noisy at N=8) independently of batch size, at the same retrain cost.

**GroupNorm over LayerNorm for the plain conv architectures**: GroupNorm (Wu & He, 2018) was introduced specifically to address BatchNorm's small-batch instability in vision models, normalizing within channel groups rather than across a full batch. LayerNorm normalizes across all channels at once and is the established choice for Transformer/RNN-style blocks where a "token" has no strong per-channel spatial structure — which is why it is appropriate inside `RestormerBlock` (attention-based) but not the natural fit for the plain convolutional blocks in `unet`/`unet_v2`/`resunet`/`attention_unet`.

**Channel counts verified in `scripts/`**:

| Architecture | Encoder/decoder channels | Bottleneck |
|---|---|---|
| `unet`, `unet_v2`, `resunet` (+ `_nll`) | `[64, 128, 256, 512]` | `1024` |
| `attention_unet` (+ `_nll`) | `[64, 128, 256, 512]` (attention-gate intermediate channels `f // 2` = `[32, 64, 128, 256]`, but those 1×1 convs carry no BatchNorm in the current code, so #1 does not touch them) | `1024` |
| `efficientnet_unet` (+ `_nll`) | decoder: `[256, 128, 64, 32, 16]` | reduction from `1280` (pretrained encoder) → `256` |

All values are powers of two, but they are not uniform across every layer — the EfficientNet decoder goes down to 16 channels. Implication: `num_groups=32` is safe as a uniform choice for `unet`/`unet_v2`/`resunet`/`attention_unet` (divides every layer from 64 to 1024), but the EfficientNet decoder needs a smaller or adaptive `num_groups` (e.g. capped at the layer's channel count) for its final 16-channel block, since 32 does not divide 16. This is an implementation detail for Round 1/2, not a blocker for this plan.

---

## 4. References

- `theory-links.md` §8.4-8.10 — the ranked findings this plan resolves, and the lecture citations behind each (L03 slide 44/57, L05 slides 19/48/78/81-83, L06 slides 77-78, L07 slides 23/35/37).
- `theory-links.md` §3.5 ("BatchNorm: one slide, four hits, one real problem"), §5.2 ("Weight decay: absent, and unremarked"), §5.5 ("Weight initialization: the wrong default for our activation") — the fuller per-lecture discussion behind findings #1-#3.
- Wu, Y., & He, K. (2018). *Group Normalization*. ECCV 2018 — motivates GroupNorm as a batch-size-independent alternative to BatchNorm for vision models, cited in `theory-links.md` §8.4 as the untried remedy.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. ICCV 2015 — introduces the He/Kaiming initializer for ReLU networks, the fix for finding #3.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016 — the original ResNet paper; `theory-links.md` §8.9 notes the projection-shortcut variant it describes is meant only for layers where channel counts change, not unconditionally as in `resunet.py`.
- `code-review.md` §3-4 — existing open-issues list; per `theory-links.md` §8.4/§8.10, findings #1 and #6 are not yet folded in there (tracked as Next steps in `.claude/handoff/HANDOFF.md`).
- `evaluation.md` §4b — records the single-artwork test fold instance cited as evidence for finding #4 (no cross-validation).
- `.claude/handoff/HANDOFF.md` — session context this plan was derived from; lists the same 8 findings as "Top priority" before this document existed.
