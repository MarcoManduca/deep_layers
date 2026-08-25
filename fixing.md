# Fixing Plan — Deep Layers

Scope: action plan for the ranked findings surfaced by `theory-links.md` §8.4-8.10 (course-lecture-to-code review), plus two loss-function findings (#9, #10) surfaced afterward from a direct question about loss correctness, a paper in `biblio/`, and a colleague's review. Each finding is restated with its priority, the type of change it requires, which checkpoints it forces a retrain of, and the reasoning behind the sequencing chosen to minimize the number of full retrain passes.

---

## 0. Implementation status

- **Round 0 (pilot)** — executed on branch `project-fixing`, `notebooks/P0_groupnorm_pilot.ipynb`. Result: `unet` improved on every metric with GroupNorm (loss 0.2127→0.1964, ssim 0.549→0.585, psnr 18.64→19.35); `unet_v2` regressed marginally (loss 0.1944→0.1966, ssim 0.592→0.586) — no catastrophic instability in either case, so Round 1 proceeded as planned. Baseline (BatchNorm) checkpoints backed up in `models/pf1/` before any Round 1 change.
- **Round 1 — code and notebooks implemented, not yet retrained.** Every change below (#1, #2, #3, #5, #9, #10, §5 callbacks) is live in `scripts/` and `notebooks/020`–`023`/`030`–`032`/`041`/`042`/`051`–`053`; 235/235 unit tests pass. **No model has been retrained under the new recipe yet** — running `020_training.ipynb`, `021_training_nll.ipynb`, and `023_training_variants.ipynb` (in any order) is the next step before anything downstream (evaluation/signal notebooks) produces meaningful numbers again.
  - Finding #5 turned out not to change model behavior as originally conceived: verified that every `resunet`/`resunet_nll` block always changes channel count (one block per depth level, widening on the way down), so a conditional identity shortcut never fires — kept as unconditional projection, documented in the code. Fixing this for real would need a structural change (a second, same-width block per level) outside this finding's scope.
  - `efficientnet_unet`/`efficientnet_unet_nll` are untouched by Round 1 — they keep the pre-Round-1 recipe (`combined_loss_advanced`, both `gaussian_nll`/`beta_nll`) until Round 2, now consolidated into `022_training_efficientnet.ipynb`/`032_evaluation_efficientnet.ipynb`.
- **Round 2, 3, 4** — not started.

---

## 1. Findings, priority, and change type

Ranked most to least severe, as in `theory-links.md` §8.4-8.10.

| # | Finding | Priority | Change type | Retrain scope |
|---|---|---|---|---|
| 1 | `BatchNorm` with `BATCH_SIZE=8` (18 BatchNorm layers in `unet`, batch size ~10-30x below the 50-256 range two lectures cite as typical) | Highest — best-evidenced (two independent lectures flag the same parameter) | Architectural — replace `BatchNormalization` with `GroupNormalization` | All BatchNorm-using architectures |
| 2 | No weight decay anywhere (`kernel_regularizer`, `regularizers`, `Adam(weight_decay=...)` all absent) | High | Hyperparameter — add `kernel_regularizer` or `Adam(weight_decay=...)` | All architectures |
| 3 | All 23 conv layers use `GlorotUniform` (Xavier) with ReLU activations — the literature's own stated bad pairing | Medium | Hyperparameter — `kernel_initializer="he_normal"` | All architectures (mitigated by BatchNorm/GroupNorm, so lower urgency than #1/#2 despite being "free") |
| 4 | No cross-validation — single fixed split, point-estimate metrics only | Medium | Evaluation methodology — grouped k-fold on top of the existing artwork-grouped split | K trainings per model included, not architecture-specific |
| 5 | `resunet.py`'s residual shortcut is an unconditional learned projection (`Conv2D 1×1` + BatchNorm on every block), not the pure identity path the "uninterrupted additive path" argument (L06 slides 77-78) requires | Medium — **resolved as not applicable** (see §0): every block always changes channel count, so a conditional identity path would be dead code | Architectural — identity shortcut where channels already match | `resunet`, `resunet_nll` — kept as unconditional projection (GroupNorm/He init still applied) |
| 6 | Two-phase EfficientNet fine-tuning unused (`freeze_encoder=False` exists, never set) | Lower | Training procedure — unfreeze encoder in a second training phase | `efficientnet_unet`, `efficientnet_unet_nll` only |
| 7 | Dilated convolution never tried — the standard way to widen receptive field without downsampling, which is this project's core architectural tension | Lower | Architectural — new variant, `dilation_rate` on selected conv layers | New variants only, not a fix to an existing model |
| 8 | Documentation-only gaps (heteroscedastic-head universality/manifold argument, annotation-budget rationale, augmentation label-preservation principle — none written up) | Lowest | Documentation only | None |
| 9 | Deterministic-family loss is `ℓ1`/`ℓ2`-dominant, and `efficientnet_unet` has no perceptual term at all (Zhao et al. 2016 — see §5). `combined_loss`'s `LOSS_ALPHA=0.7` weights MAE over SSIM in the opposite proportion to the paper's best-performing config; `combined_loss_advanced` (MAE + Laplacian pyramid + FFT) has no SSIM/MS-SSIM term | High — direct, well-evidenced fix, not merely an omission | Hyperparameter/loss-formula — unify all 6 deterministic architectures under one loss, `0.16·Charbonnier + 0.84·(1−MS-SSIM)` (colleague proposal, matches the paper's `Mix`) | All 6 deterministic architectures (every one that trains against `combined_loss`/`combined_loss_advanced`) |
| 10 | NLL-family loss is Gaussian (`ℓ2`-weighted uncertainty); no Laplace (`ℓ1`-weighted) alternative has been tried, despite the same L1-over-L2 evidence applying to the likelihood term as to the deterministic fidelity term | High — same evidence base as #9, applied to the likelihood term | Loss-formula — replace `gaussian_nll_loss`/`beta_gaussian_nll_loss` with a single Laplace NLL, `stop_grad(b)^β · (\|y−μ\|/b + log b)`, β=0.5 (colleague proposal); collapses the `nll_gaussian`/`nll_beta` split into one loss per architecture | All 4 NLL architectures — collapses 8 NLL checkpoints (4 architectures × {gaussian, beta}) into 4 (one per architecture) |

---

## 2. Retrain-minimizing plan

The naive approach — one retrain pass per finding — would mean up to 8 full passes over the checkpoint tree. Findings #1-#3, #5, #9, and #10 are all compatible changes to the same training run (none undoes another's effect), so they are bundled into shared retrain rounds instead. #4, #6, #7 are kept as separate rounds because they either multiply cost (#4), are scoped to a disjoint set of models (#6), or are exploratory additions rather than fixes (#7).

**Note on #9/#10 (added after the colleague's loss-function review, following the Round 0 pilot)**: both are retrain-requiring changes — a different loss changes what the optimizer minimizes, so every checkpoint trained under the old loss must be retrained regardless of whether its architecture changes. Since #9/#10 touch the same models already being retrained for #1/#2/#3/#5 in Round 1, and the EfficientNet models already being retrained in Round 2, they are folded into whichever round already retrains each model — no new round, per the decision to minimize retrain cycles.

### Round 0 — Pilot validation (small, before committing to a full pass)

- **Models**: `unet`, `unet_v2` only.
- **Change**: #1 (`BatchNormalization` → `GroupNormalization`) in isolation.
- **Why first and alone**: #1 is the highest-priority finding but also the riskiest — it is the only one of the bundle that changes how activations are normalized at every layer. Validating it cheaply on two representative deterministic architectures before committing #1 across the whole tree avoids discovering a regression only after 12 models have been retrained.

### Round 1 — Combined "recipe v2" retrain

- **Models**: `unet`, `unet_v2`, `resunet`, `attention_unet`, `unet_restormer` (5 deterministic architectures) + `unet_nll`, `resunet_nll`, `attention_unet_nll` (3 NLL architectures — `unet_v2`/`unet_restormer` have no NLL counterpart in the current tree). 8 checkpoints total (down from the 11 that these architectures represent in the current 14-checkpoint tree, since #10 collapses each NLL architecture's `nll_gaussian`/`nll_beta` pair into a single checkpoint). EfficientNet-based models are handled separately in Round 2.
- **Changes bundled in one pass**:
  - #1 `BatchNormalization` → `GroupNormalization` (`num_groups=32`, uniform across these architectures — see §3 below for why)
  - #2 weight decay (`Adam(weight_decay=...)` or `kernel_regularizer`)
  - #3 `kernel_initializer="he_normal"`
  - #5 identity shortcut in `resunet`/`resunet_nll` (bundled here since those two models are already being retrained in this pass)
  - #9 unified deterministic loss (`0.16·Charbonnier + 0.84·(1−MS-SSIM)`) for `unet`, `unet_v2`, `resunet`, `attention_unet`, `unet_restormer`
  - #10 unified NLL loss (Laplace-beta, β=0.5) for `unet_nll`, `resunet_nll`, `attention_unet_nll` — replaces both `gaussian_nll_loss` and `beta_gaussian_nll_loss` for these three architectures
  - Retuned, parametric training callbacks (`val_loss` monitored uniformly across checkpoint/early-stop/LR-reduce, new patience/factor/cooldown/min_delta/min_lr values sourced from `settings.*` — see §5)
- **Why bundled**: none of these changes interacts adversarially with another — weight decay, initializer, and loss-formula changes only affect the optimization objective/trajectory, not the forward-pass structure that #1 changes (except for the NLL head's second output channel, see §5). Splitting them into separate full retrain passes would multiply the compute cost for no gain in signal, since the plan does not require isolating each one's individual marginal contribution before shipping.
- **`unet_restormer` exception**: already uses `LayerNormalization` inside `RestormerBlock` (a Transformer-derived block, where LayerNorm is the established choice — see §3). Left unmodified by #1; #2/#3/#9 still apply to it in this round (it has no NLL counterpart, so #10 does not apply).

### Round 2 — EfficientNet, handled separately

- **Models**: `efficientnet_unet`, `efficientnet_unet_nll` (now a single checkpoint, per #10's collapse), plus a new `efficientnet_unet_ft` variant (see below). 3 checkpoints total.
- **Why separate from Round 1**: `efficientnet_unet(_nll)` uses a pretrained EfficientNetB0 encoder. Applying #1 to the encoder's BatchNorm layers would destroy the pretrained batch statistics, defeating the purpose of transfer learning — so #1 is scoped to the custom decoder only (`decoder_filters = [256, 128, 64, 32, 16]`, see §3 for the group-count implication).
- **Changes**: #1 (decoder only), #2, #3 applied to the decoder; #6 (two-phase fine-tuning) applied as a second training phase; #9 (unified deterministic loss, replacing `combined_loss_advanced` — this also removes the Laplacian-pyramid/FFT terms, see §5) for `efficientnet_unet`; #10 (unified NLL loss) for `efficientnet_unet_nll`.
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

## 4. Loss-function findings (#9, #10)

Surfaced from a direct question about whether the current losses are correct for the RGB→IR task, cross-checked against Zhao, Gallo, Frosio & Kautz, *"Loss Functions for Image Restoration with Neural Networks"* (2016) — a paper specifically about the loss layer's impact on image-restoration quality — and then refined by a colleague's proposed unification.

**What the current losses already get right**: neither `combined_loss` nor `combined_loss_advanced` ever uses plain `ℓ2`, which the paper identifies as the single biggest source of avoidable quality loss (splotchy artifacts in flat regions, poor correlation with perceived quality) — both already lean on `ℓ1`-family + a structural/frequency term, the right general direction.

**#9 — Deterministic family**: the paper's best-performing loss (`Mix`) is `0.84·MS-SSIM + 0.16·ℓ1`, empirically outperforming `ℓ2`, `ℓ1`, `SSIM` alone, and `MS-SSIM` alone on every quality metric tested (Table I). Two concrete gaps against this project's current losses:

- `combined_loss`'s `LOSS_ALPHA=0.7` weights the fidelity term (MAE) at 70% and the perceptual term (SSIM) at only 30% — the inverse emphasis of the paper's finding.
- `combined_loss` uses single-scale `tf.image.ssim`, not multi-scale. The paper shows single-scale SSIM is a forced trade-off (small `σ_G` fixes edges but reintroduces flat-region artifacts; large `σ_G` does the opposite) — precisely the reason MS-SSIM exists. `tf.image.ssim_multiscale` is a direct drop-in.
- `combined_loss_advanced` (`efficientnet_unet` only) has no perceptual term at all — just MAE + Laplacian pyramid + FFT magnitude, none of which model HVS luminance/contrast/structure sensitivity the way SSIM-family metrics do.

The colleague's proposed unification, `loss = 0.16·Charbonnier(y, ŷ) + 0.84·(1 − MS-SSIM(y, ŷ))` for **all 6** deterministic architectures, resolves all three: matches the paper's weighting exactly, upgrades to multi-scale, and gives `efficientnet_unet` a perceptual term for the first time — at the cost of dropping its Laplacian-pyramid/FFT terms, which is a real design trade-off (those terms were specifically motivated for that architecture's richer pretrained features) worth confirming empirically rather than assuming as a pure improvement. Charbonnier (`sqrt((y−ŷ)² + ε²) − ε`) is a smooth `ℓ1` approximation standard in image-restoration literature, preferred over raw `ℓ1` for its defined gradient at zero error.

**#10 — NLL family**: the paper's finding — `ℓ1`-family losses beat `ℓ2`-family — applies just as much to the likelihood term of the heteroscedastic models as to the deterministic fidelity term. The current `gaussian_nll_loss` (`0.5·exp(-log_var)·(y−μ)² + 0.5·log_var`) is exactly the NLL of a Gaussian, i.e. `ℓ2`-weighted-by-uncertainty. The colleague's proposal, `stop_grad(b)^β · (|y−μ|/b + log b)` with β=0.5, is the NLL of a **Laplace** distribution (scale `b`) instead — the `ℓ1`-weighted-by-uncertainty analogue, with the same Seitzer et al. (2022) beta-reweighting trick applied to the Laplace scale rather than the Gaussian variance (a motivated generalization, not the literally published formula — worth documenting as such, the way `code-review.md` §7.6 already documents the existing beta-NLL choice).

Two implementation notes for whoever picks this up:
- The NLL heads' second output channel is currently named/interpreted as `log_var` (`ClipLogVar`, `NLL_LOG_VAR_MIN/MAX`). Under Laplace it becomes `log_b` (log-scale, not log-variance) — same layer, same clip mechanism, but the sensible numeric range for `b` is not necessarily the same as for `log_var`, so the clip bounds should be re-checked rather than assumed to transfer.
- Per the decision recorded when this finding was added: the Laplace-beta loss **replaces** both `gaussian_nll_loss` and `beta_gaussian_nll_loss` (rather than being added as a third variant), collapsing the `models/nll_gaussian/`+`models/nll_beta/` split into one NLL checkpoint per architecture. This deliberately closes the gaussian-vs-beta comparison axis that `.claude/handoff/HANDOFF.md` records as having a "mixed empirical record" — a considered trade-off (simpler tree, one fewer axis to maintain), not an oversight.

---

## 5. Callback tuning (Round 1)

Surfaced while discussing #9/#10: since the training objective changes shape in Round 1 (`combined_loss`'s balance shifts from MAE-dominant to 84%-MS-SSIM-dominant), the callbacks that decide when to checkpoint/stop/reduce LR should be reconsidered alongside it, not left as-is. A colleague proposal split the three callbacks across different monitored metrics (`val_mae` for checkpointing, `val_ssim` for early-stopping/LR); discussed and rejected in favor of monitoring **`val_loss` uniformly across all three** — since `val_loss` already *is* the actual training objective in whatever proportion #9 lands on, splitting onto different metrics risks saving a checkpoint that has no relationship to the plateau that determined how long training ran, and `val_ssim` alone (single-scale) doesn't even match what the loss now optimizes (MS-SSIM).

**Implemented** (see §0). `scripts/config.py`/`scripts/trainer.py`/`scripts/trainer_nll.py`:

- New fields added to `scripts/config.py` (`Settings`), with the values discussed here as defaults:
  - `EARLY_STOPPING_PATIENCE: int = 20` (was hardcoded `15` in `trainer.get_callbacks`)
  - `EARLY_STOPPING_MIN_DELTA: float = 0.0`
  - `EARLY_STOPPING_RESTORE_BEST_WEIGHTS: bool = False` (was hardcoded `True`; safe because every notebook reloads the best checkpoint from disk rather than reusing the in-memory `model` post-`fit()` — see caveat below)
  - `REDUCE_LR_FACTOR: float = 0.25` (was hardcoded `0.5`)
  - `REDUCE_LR_PATIENCE: int = 6` (was hardcoded `7`)
  - `REDUCE_LR_COOLDOWN: int = 2` (new — `get_callbacks` currently passes no `cooldown`, so it defaults to `0`; prevents rapid repeated LR drops right after a reduction)
  - `REDUCE_LR_MIN_DELTA: float = 1e-4` (new — currently unset/`0`)
  - `REDUCE_LR_MIN_LR: float = 1e-6` (was hardcoded `1e-7`)
- All three callbacks' `monitor` set to `"val_loss"` (`mode="min"`, the Keras default for a loss — no `mode=` override needed, unlike the rejected `val_ssim`/`val_mae` proposal which needed explicit `mode="max"`/`"min"`).
- `trainer.get_callbacks` now accepts these as parameters defaulting to the new `settings.*` fields (same pattern already used by `compile_model(lr: float = settings.LEARNING_RATE, ...)`), rather than hardcoding the literals inline as before. `trainer_nll` reuses the same `get_callbacks` (it never had its own).
- **Caveat to carry forward**: `restore_best_weights=False` only stays safe as long as no notebook evaluates the in-memory `model` object immediately after `fit()` instead of reloading `best_model.keras` from disk. True today (`020_training.ipynb` and this plan's future notebooks all reload via `trainer.load_model`); worth a one-line check if that convention ever changes.

---

## 6. References

- `theory-links.md` §8.4-8.10 — the ranked findings this plan resolves, and the lecture citations behind each (L03 slide 44/57, L05 slides 19/48/78/81-83, L06 slides 77-78, L07 slides 23/35/37).
- `theory-links.md` §3.5 ("BatchNorm: one slide, four hits, one real problem"), §5.2 ("Weight decay: absent, and unremarked"), §5.5 ("Weight initialization: the wrong default for our activation") — the fuller per-lecture discussion behind findings #1-#3.
- Wu, Y., & He, K. (2018). *Group Normalization*. ECCV 2018 — motivates GroupNorm as a batch-size-independent alternative to BatchNorm for vision models, cited in `theory-links.md` §8.4 as the untried remedy.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. ICCV 2015 — introduces the He/Kaiming initializer for ReLU networks, the fix for finding #3.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016 — the original ResNet paper; `theory-links.md` §8.9 notes the projection-shortcut variant it describes is meant only for layers where channel counts change, not unconditionally as in `resunet.py`.
- `code-review.md` §3-4 — existing open-issues list; per `theory-links.md` §8.4/§8.10, findings #1 and #6 are not yet folded in there (tracked as Next steps in `.claude/handoff/HANDOFF.md`).
- `evaluation.md` §4b — records the single-artwork test fold instance cited as evidence for finding #4 (no cross-validation).
- Zhao, H., Gallo, O., Frosio, I., & Kautz, J. (2016). *Loss Functions for Image Restoration with Neural Networks*. IEEE Transactions on Computational Imaging (arXiv:1511.08861) — motivates findings #9/#10: `ℓ2` correlates poorly with perceived quality and produces flat-region artifacts; `ℓ1` alone already beats it; the best combination tested is `0.84·MS-SSIM + 0.16·ℓ1` (`Mix`). Read in full from `biblio/1511.08861v3.pdf`.
- Seitzer, M., et al. (2022). *On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks*. ICLR 2022 (arXiv:2203.09168) — origin of the beta-reweighting trick `#10` generalizes from the Gaussian variance to the Laplace scale; already cited for the existing `beta_gaussian_nll_loss` in `code-review.md` §7.6.
- `.claude/handoff/HANDOFF.md` — session context this plan was derived from; lists the same 8 findings as "Top priority" before this document existed; also records the gaussian-vs-beta NLL comparison that #10 deliberately closes.
