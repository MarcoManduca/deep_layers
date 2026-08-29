# Course concepts → project decisions

A map between the *Foundations of Deep Learning* course material (Prof. Paolo
Napoletano, UNIMIB Data Science, a.a. 2025/2026) and the concrete decisions in
this repository. For every concept the course introduces, this file records
**where it lives in the code** and, more importantly, **why the project chose
what it chose** — including the places where we deliberately depart from the
textbook recipe, and the places where the course names a problem we actually have.

## How this file is organised

**Part 0** is a glossary of the elementary vocabulary: a definition, where it
appears in the code, and a pointer to wherever it is argued at length. Read it
first or use it as an index; it is deliberately terse.

**Parts 1–6 follow the lectures in order, L03 → L08**, which is also roughly
simplest to hardest: what learning is and when it generalizes (L03), how a
feed-forward network learns (L04), what a convolution buys (L05), what
recurrence would buy and why we decline it (L06), how to stop a large model
overfitting a small dataset (L07), and finally the two topics the project is
actually built on — autoencoder topology and dense prediction (L08).

**Part 7** is a set of per-lecture summary tables. **Part 8** records the design
questions this mapping raised and how each was resolved.

Every claim about the code was verified against this repository, and the
arithmetic is shown so it can be re-checked.

## Lectures mapped

| Lecture | Topics | Part |
|---|---|---|
| **L03 — Machine Learning Basics** | generalization, i.i.d. assumptions, capacity, over/underfitting, gradient descent, mini-batch SGD, cross-validation | **Part 1** |
| **L04 — Feed-Forward Networks** | ANNs, non-linearity, activation functions, backpropagation, vanishing gradient | **Part 2** |
| **L05 — Convolutional Neural Networks** | convolution, pooling, activations, BatchNorm, AlexNet→ResNet, transfer learning | **Part 3** (+ most of Part 0) |
| **L06 — Recurrent Neural Networks** | RNNs, unrolling, architecture taxonomy, vanilla RNN, BPTT, vanishing/exploding gradients, LSTM | **Part 4** |
| **L07 — Regularization in Deep Learning** | capacity, data augmentation, norm penalties, dropout, initialization, early stopping, universality, evaluation metrics | **Part 5** |
| **L08 — Autoencoders + Semantic Segmentation** | autoencoders, connection to PCA, deep autoencoders, manifold, SemSeg architectures | **Part 6** |

L00–L02 are out of scope by decision.

## Citation convention

Slide numbers are the ones **printed on the slide**, not the PDF page. The two
coincide for L03, L06 and L08. They diverge elsewhere, because animated slides
occupy several PDF pages each: L05 runs +1 from page 11 and +2 from page 71
(slide 48 = page 47); L04 runs +1 from page 16 and +2 from page 36; L07 runs +2
from page 12, +4 from page 24 and +6 from page 34 (slide 48 = page 42).

Every slide number and every quotation here was checked against the extracted PDF
text. Where a slide's content is an image rather than text — the gradient-flow
mathematics in L06 slides 53–63, the MNIST results in L08 slides 17–21 — the
claim is described rather than quoted.

## A note on Part 4

This project contains no recurrent network and never will in its present form.
Part 4 is therefore mostly a record of *why not*, plus the four places where
L06's material transfers to what we actually built anyway: parameter sharing,
truncated BPTT, additive gradient paths, and gating. A lecture that does not
apply is still worth mapping — knowing precisely why it does not apply is what
makes the choices elsewhere defensible rather than accidental.

## Companion documents

Read [README.md](README.md) first for what the project does; this file assumes it
and adds the theory layer. Also [code-review.md](code-review.md) (code structure
and engineering choices) and [final-comments.md](final-comments.md) (the
quantitative model comparison and the recommended models).

## A note on the final recipe

Several course concepts below are discussed first against the textbook default and
then against what the project actually settled on, because the two differ. The
final training recipe uses **`GroupNormalization`** (not BatchNorm — unstable at
`BATCH_SIZE = 8`), **He initialisation** (not the Glorot default — ReLU
throughout), **weight decay** `1e-5` and **per-element gradient clipping** in the
optimiser, and a **grouped k-fold** cross-validation on top of the fixed split.
Where a section discusses the choice as an open question, Part 8 records how it
was resolved and what was found.

---

# Part 0 — Glossary

Terse by design: definition, where it is in the code, and a pointer to the
argument. L05 supplies most of this vocabulary; the other lectures assume it.

## 0.1 Filter (kernel)

A small array of numbers — typically `3×3` — multiplied element-wise against a
patch of the image and summed to one output number. **L05 slides 5, 18.**

L05 slide 5 frames it as *local connectivity*: "we will connect each neuron to
only a local region of the input volume", with the asymmetry that connections are
local in width and height but "**always full along the entire depth**". So a
`3×3` filter on a 64-channel input is really `3×3×64`.

Every filter in this project is **learned**, with one exception: the fixed
Gaussian window `_gaussian_kernel_1d` in
[delta_analysis.py](scripts/delta_analysis.py), which is a measurement tool
applied after inference, not part of a model. L05 slide 14's hand-chosen
edge-detection kernel is the lecture's version of the same idea.

## 0.2 Sliding window

Take a window of fixed size, move it across the image position by position,
compute something at every position, assemble the results. L05 slide 18 gives it
in words: "placing 3 by 3 filter on the top left corner of image… **move filter
to right one pixel at a time**, and repeat this process".

**The term means four different things in this project, and conflating them is
the main way to misread the code:**

| # | Sliding window of… | What slides across the image | Computed at each position | Status |
|---|---|---|---|---|
| 1 | **a learned filter** | a `3×3` kernel | a weighted sum → *convolution* | every architecture — L05 slide 18 |
| 2 | **a whole classifier** | an image patch | one CNN forward pass classifying the centre pixel | **rejected** — L08 slides 28–32, §6.11 |
| 3 | **a statistics window** | an `11×11` Gaussian window | local mean, variance, covariance — nothing learned | [delta_analysis.py](scripts/delta_analysis.py), §6.5 |
| 4 | **a whole trained model** | a large tile of a painting | a full model prediction, then blend tiles | **retired** — a tiled/blended inference path existed earlier and was removed once every notebook used whole-image inference (§4.6, code-review §5.5) |

A convolution *is* a sliding window (case 1). But "sliding window" as L08 uses it
in the semantic-segmentation section (case 2) is a strategy for the whole task,
not an operation inside a layer — and it is the one the course rejects. Cases 3
and 4 are ours and correspond to nothing in any lecture in scope.

## 0.3 Convolution

A sliding window of a learned filter. L05 slide 6: CNNs "use convolution in place
of general matrix multiplication in at least one layer". Slide 9 names the output
the **feature map** and notes that what is computed is cross-correlation
("convolution without kernel-flipping"). **L05 slides 6, 9, 12–15, 29–30.**

Two properties, one slide each in L05 — **sparse connectivity** (12) and
**parameter sharing** (13):

- **Weight sharing.** L05 slide 29 gives the count as `(K × K × D₁) × C` weights
  plus `C` biases, worked on slide 30 as `3×3×3 + 1 = 28` per filter, `2800` for a
  100-filter layer. Applied to `unet`'s first block: `3×3×3×64 = 1728` weights and
  no bias (`use_bias=False`, since BatchNorm's learned shift makes it redundant).
  The count does not change with image size.
- **Translation equivariance.** L05 slide 53, on AlexNet's learned filters: "if
  detecting a horizontal edge is important at some location in the image, it
  should intuitively be useful at some other location as well". This is why one
  trained model runs on any image size — see §6.11.

## 0.4 Padding

Without padding a `3×3` filter cannot be centred on a border pixel, so the output
shrinks by 2 pixels per convolution; L05 slide 9 calls the un-padded version
"valid". L05 slide 19: "Padding: to not alter the spatial dimensions of the
input. We pad input in every direction with 0's". L05 slide 28 gives the rule:
zero padding with **`(K−1)/2`** preserves size.

For `K = 3` that is a pad of 1, which is exactly `padding="same"` — used by every
convolution here, because "valid" padding would erode resolution at each of the
23 convolutions and break the project's central invariant (§3.6, §6.12).

The measurement path pads differently: `gaussian_local_filter` uses
`mode="reflect"`, because zero-padding a `[0,1]` image at the border invents black
content and would produce a false-positive delta ring around every painting. L05
discusses only zero padding; this is a case the lecture does not cover.

## 0.5 Stride and dilation

**Stride** — how far the window jumps. L05 slide 19: "the number of pixels we
move filter to the right/down"; slide 28 gives the output-size formula
(`N=7, K=3, S=2 → 3`). Stride 1 in all feature-extracting convolutions; stride 2
for downsampling in [unet_v2.py](scripts/unet_v2.py) (`use_strided_conv=True`)
and in every `Conv2DTranspose`.

**Dilation** — L05 slide 19, "to have a larger receptive field": spread the
filter taps apart so context widens without downsampling. Used in the
`unet_dilated` / `unet_v2_dilated` variants (an ASPP-style bottleneck,
`scripts/aspp.py`); it produced no benefit and some training instability — see
§8.8.

## 0.6 Receptive field

**The course uses this term in two senses, and mixing them up is an easy way to
lose marks.**

1. **Per-neuron, one layer** — L05 slide 5: "the spatial extent of this
   connectivity is a hyperparameter called the receptive field of the neuron
   (equivalently **this is the filter size**)". In this sense ours is `3×3`.
2. **Cumulative** — L05 slide 64's "effective receptive field": the region of the
   *input image* influencing one deep value. Each `3×3` convolution adds 2 pixels;
   each stride-2 downsample doubles the growth rate of everything after it.

Sense 2 computed for [unet.py](scripts/unet.py):

| After | Effective receptive field | Stride |
|---|---|---|
| encoder level 1 | 6 × 6 | 2 |
| encoder level 2 | 16 × 16 | 4 |
| encoder level 3 | 36 × 36 | 8 |
| encoder level 4 | 76 × 76 | 16 |
| bottleneck | **140 × 140** | 16 |

A bottleneck unit sees `140×140` of a `400×400` training patch — about 12% of its
area. Why that much context is necessary: §6.10. Why stacked `3×3` rather than one
large kernel: §3.2.

## 0.7 Pooling / downsampling

Reducing spatial resolution. L05 slide 33 places it in Goodfellow's three-stage
layer (convolution → detector → pooling); slide 34, it "reduces the
dimensionality of the feature maps"; slide 36 works the example `[224×224×64]` →
`[112×112×64]`, "notice that the volume depth is preserved". Slide 37 lists the
alternatives we never benchmarked: average, mixed, Lᴾ, stochastic,
spatial-pyramid, RoI.

Two mechanisms here, and L08 slide 37 names exactly this pair ("Downsampling:
Pooling, strided convolution"):

- **Max pooling** (`MaxPool2D(2)`) — no parameters. Default in
  [unet.py](scripts/unet.py), [resunet.py](scripts/resunet.py),
  [attention_unet.py](scripts/attention_unet.py).
- **Strided convolution** — downsamples *and* learns how to.
  `use_strided_conv=True` in [unet_v2.py](scripts/unet_v2.py).

Pooling's translation invariance is a benefit for classification and half a
liability for us — argued in §3.4.

## 0.8 Upsampling

Going back up in resolution. L08 slides 38–41 present four mechanisms; L05 slide
58 gives ZFNet's earlier intuition — "a convnet model that uses the same
components (filtering, pooling) but **in reverse**". We use two of the four.

**Nearest neighbour** (L08 slide 38) — copy each value into a `2×2` block:

```
1 2      1 1 2 2
3 4  →   1 1 2 2
         3 3 4 4
         3 3 4 4
```

**"Bed of nails"** (L08 slide 38) — place each value, zero the rest:

```
1 2      1 0 2 0
3 4  →   0 0 0 0
         3 0 4 0
         0 0 0 0
```

**Max unpooling** (L08 slide 39) — during pooling remember *which* position held
the maximum ("use positions from pooling layer"); on the way up put the value
exactly there and leave the rest at zero:

```
     max pooling                        max unpooling
     (remember which element was max)   (reuse those positions)

     1 2 6 3                             0 0 2 0
     3 5 2 1  →  5 6      … network …    0 1 0 0
     1 2 2 1     7 8      1 2      →     0 0 0 0
     7 3 4 8              3 4            3 0 0 4
```

Note what survives: the *positions* come from the encoder, the *values* from the
decoder. The original 5, 6, 7, 8 are gone, and every non-maximal input value was
never recorded at all. This is why we reject it — §6.13.

**Transposed convolution** (L08 slides 40–41, labelled *deconvolution*) — a
learned convolution over the sparse grid. The name is a misnomer: it does not
invert a convolution, it is a convolution with forward and backward passes
swapped.

In this project: `Conv2DTranspose(f, 2, strides=2, padding="same")` by default
everywhere; bilinear resize + `Conv2D` as the alternative in
[unet_v2.py](scripts/unet_v2.py) (`use_upsample_conv=True`, via `_Upsample2x`).
The choice is argued in §6.13.

## 0.9 Depthwise and 1×1 convolution

**Depthwise** — filters each channel independently, breaking L05 slide 5's
"always full along the entire depth" rule on purpose, to be much cheaper. **No
lecture in scope covers it.** It appears only inside borrowed components: the
EfficientNetB0 backbone ([efficientnet_unet.py](scripts/efficientnet_unet.py))
and the `q,k,v` projections and gated feed-forward of `RestormerBlock`
([unet_restormer.py](scripts/unet_restormer.py)) — the "Dconv" in *MDTA*,
Multi-**Dconv** Head Transposed Attention.

**1×1** — no spatial mixing, only a learned recombination across channels;
effectively a per-pixel linear layer. Implied by L05 slide 23's "number of filters
allows us to increase or decrease channel size", and central to GoogLeNet's
Inception module (L05 slides 66–70). Every architecture here ends with one,
`Conv2D(1, 1, activation="sigmoid")`, collapsing 64 full-resolution channels to
the single IR value (2 channels for the `*_nll` variants). It also appears three
times inside `_attention_gate` and as the projection shortcut in
`_residual_block` — the latter with consequences, §4.8.

## 0.10 Feature map, channels, code, bottleneck

- **Feature map** — the `H × W × C` output of a layer. L05 slide 9 names it;
  slide 2 frames the network as transforming "the 3D input volume to a 3D output
  volume of neuron activations".
- **Channels** — independent learned features per position. L05 slides 22–23:
  "**number of filters allows us to increase or decrease channel size**", which is
  what `filters=[64, 128, 256, 512]` does.
- **Code** — L08 slide 2's `h`, the representation at the narrowest point.
- **Bottleneck** — that point. Here `bottleneck=1024` channels at `H/16 × W/16`.

By dimension count this "bottleneck" is not one — §6.3.

## 0.11 Skip connection, residual, concatenation

A **skip connection** routes an encoder feature map to the matching decoder
level, bypassing the bottleneck. UNet joins them by **concatenation** —
`Concatenate()([x, skip])` stacks along the channel axis, so nothing is discarded
and the next convolution learns the combination.

Distinct from a **residual** connection, which *adds* rather than concatenates,
inside a block rather than across the U. L05 slides 74–77 cover this as ResNet:
"use network layers to fit a **residual mapping** instead of directly trying to
fit a desired underlying mapping" (slide 76), with "every residual block has two
3×3 conv layers" (slide 77) — which matches `_residual_block` in
[resunet.py](scripts/resunet.py) exactly.

Skips are how the position that pooling discards gets handed back (§3.4), and why
UNet keeps detail a plain encoder–decoder loses. They would be pathological if
this were an autoencoder — §6.1.

## 0.12 Batch / Group Normalization

Batch Normalization normalizes each channel to zero mean and unit variance over
the current mini-batch, then rescales by two learned parameters. **L05 slides
44–48, 78.** The lecture's canonical normalizer — but its batch statistics are
noisy at small batch sizes, and this project trains at `BATCH_SIZE = 8`.

The from-scratch architectures therefore use **`GroupNormalization`** instead
(Wu & He, 2018): the same normalize-then-rescale operation, but over groups of
channels within each sample, so it is independent of batch size. One per conv
block, ~18 in `unet`, with `use_bias=False` on the preceding convolution. The
pretrained EfficientNetB0 encoder keeps its own BatchNorm (the ImageNet running
statistics are part of the transferred knowledge). Why the switch, and the four
ways L05 slide 48 lands on the original choice: §3.5.

## 0.13 Dropout

Randomly zeroing activations during training. `SpatialDropout2D` drops entire
**channels**, the right granularity for convolutional features since neighbouring
pixels in a feature map are correlated. **L07 slides 30–33**; L05 slide 54
(AlexNet, 0.5) and slide 78 (ResNet, "no dropout used") record both ends of the
historical argument.

Here: `dropout_rate` defaults to `0.0`, exists only in
[unet_v2.py](scripts/unet_v2.py), and is confined to the bottleneck and first
decoder block. Both reasons for that scope, and its actual (unused) status: §5.4.

## 0.14 Activations

- **ReLU** — after every normalization layer, throughout. L05 slide 31: "preferred
  activation function in CNN is ReLU… leaves outputs with positive values as is,
  replaces negative values with 0". L04 slide 18 makes the assignment explicit
  per architecture family — §2.2. Applied through a small `_ReLUFix` wrapper layer
  rather than `layers.ReLU()`, to work around a tensorflow-metal GPU bug — see
  [code-review.md](code-review.md) §3.3.
- **Sigmoid** — output only, squashing to `[0, 1]` to match the normalized IR
  target. The one exception is `unet_residual`, whose head instead applies
  **`tanh`** to a signed residual added to `mean(R,G,B)` — the only `tanh` in the
  project (§5.7). Note the difference from L05 slide 40, where sigmoid represents "a
  probability distribution over a binary variable": ours is not a probability but
  a bounded regression output (§0.15). The `log_var` channel of the `*_nll` models
  has no sigmoid; it is clipped by `ClipLogVar`
  ([nll_layers.py](scripts/nll_layers.py)) to `[-6, 6]`, since a log-variance must
  be free to be negative.
- **`tanh`** — L06's recurrent activation (slides 30, 66, 74) and the one
  activation in any lecture in scope that appears **nowhere** here (verified: no
  `tanh` in `scripts/`). Why: §2.2, §4.5.
- **GELU** — once, inside `RestormerBlock`'s gated feed-forward. Arrives with the
  borrowed component; no lecture in scope covers it.

## 0.15 Prediction head, loss, metric

The last layer and its loss define the task:

- **Classification head** — `C` channels, softmax, cross-entropy, `argmax`. L05
  slides 39–40 for whole images; L08 slide 34 per pixel.
- **Regression head** — 1 channel, sigmoid, a distance-based loss, no `argmax`.
  Ours.
- **Distributional head** — 2 channels `(mu, log_var)`, a likelihood loss. The
  `*_nll` models. No lecture in scope covers this; §5.7 explains why the project
  needed it.

A **loss** is what gradient descent minimizes and must be differentiable; a
**metric** is what a human reads and need not be. L03 slide 20 notes that
objective, criterion, cost, loss and error all name the same thing — so
loss-versus-cost carries no information, but loss-versus-metric does. Here they
are *deliberately* different: `combined_loss` = MAE + (1 − SSIM) is optimized
while `mae`/`ssim`/`psnr` are reported, and `MuMAEMetric`/`MuSSIMMetric`/
`MuPSNRMetric` read **only** the `mu` channel so a heteroscedastic model stays
comparable to a deterministic one on the same metrics.

## 0.16 Mini-batch, epoch, backpropagation, DAG

- **DAG** — layers as nodes, tensors as edges, no cycles; the acyclicity gives a
  topological ordering, which both the forward pass and backpropagation walk
  (L06 slide 2). All ten architectures are DAGs built with the Keras functional
  API. §4.2 notes the pleasing detail that slide 2's own illustration is our
  residual block.
- **Mini-batch** — `BATCH_SIZE = 8`. L03 slide 44 gives 50–256 as typical; §1.3
  and §3.5 both flag ours.
- **Epoch** — one pass over the training split. `EPOCHS = 100`, though
  EarlyStopping (`patience=20`) usually stops sooner.
- **Backpropagation** — L04 slides 25–52 derive it; L05 slide 84 and L08 slide 3
  place it as the standard training method. Optimizer: Adam, `lr = 1e-4`,
  `weight_decay = 1e-5`, per-element gradient clipping (`clipvalue = 1.0`, see
  [code-review.md](code-review.md) §3.5), with `ReduceLROnPlateau` — the automated,
  gentler form of both L05 recipes' manual "÷10 when validation plateaus"
  (slides 54, 78).

## 0.17 Gating

Multiplying a signal element-wise by a learned coefficient to control how much
passes through; a sigmoid coefficient lies in `[0, 1]` and reads as a soft
switch. The mechanism behind every LSTM gate — **L06 slides 66–74**, where
`i`, `f`, `o` come from sigmoids and `g` from a `tanh`.

Two gates here, neither temporal: `_attention_gate`
([attention_unet.py](scripts/attention_unet.py)), a sigmoid map multiplying the
encoder skip; and the GDFN half of `RestormerBlock`, `gelu(x1) * x2`, unbounded.
How closely they match the LSTM form: §4.9.

## 0.18 What the lectures introduce that this project does not use

| Concept | Course | Status here |
|---|---|---|
| **Fully connected layer** | L05 slide 39 | none. It fixes the input resolution, which would break the central invariant. L05 slide 70 (GoogLeNet "removes FC layers completely") and slide 77 (ResNet "no FC layers at the end") go the same way (§3.6) |
| **Softmax** | L05 slide 40 | unused — the output is a continuous reflectance, not a class (§0.15) |
| **Average / stochastic / pyramid pooling** | L05 slide 37 | never benchmarked; max pooling only (§0.7) |
| **`tanh` (as a hidden activation), recurrence** | L06 | none — no sequence to process. `tanh` appears once, as `unet_residual`'s bounded output head (§4.1, §4.5) |
| **Dropout** | L07 slides 30–33 | implemented in `unet_v2`, off by default; weight decay used instead (§5.4) |
| **Adversarial / GAN augmentation** | L07 slides 6, 7, 9 | deprioritized: fabrication is worse than omission (§5.3) |
| **Neural style transfer** | L07 slide 8 | untried, though the closest on-domain precedent uses it (§5.3) |

---

# Part 1 — Machine learning basics (L03)

## 1.1 The i.i.d. assumption is the real name of our split invariant

**L03 slide 54**: "The train and test data are generated by a probability
distribution over datasets called the **data generating process**. We typically
make a set of assumptions known collectively as the **i.i.d. assumptions**. These
assumptions are that the examples in each dataset are **independent** from each
other, and that the train set and test set are **identically distributed**, drawn
from the same probability distribution as each other."

This is the most rigorous justification available for the first of the project's
core invariants, which we have so far stated only informally: split by artwork,
because leakage silently invalidates every metric. That is a rule plus a warning,
with no account of *why*. Slide 54 supplies the reason.

**Independence.** The 1167 matched pairs are not 1167 independent samples. They
are sections cut from 29 artworks, so all 35 sections of `natmorta1` share a
support, a pigment set, a varnish, an ageing history and one acquisition session.
Under a random section-level split, a test section's nearest training neighbour is
a physically adjacent piece of the same painting — and the **i** in i.i.d. fails.
`grouped_train_val_test_split` ([dataset.py](scripts/dataset.py)) restores
independence at the only level where it holds: the artwork. What the project calls
leakage is exactly a violation of the independence assumption, and the reason it
"silently invalidates every metric" is that the test error then estimates
interpolation between near-duplicates rather than generalization.

**Identical distribution — deliberately violated, by design.** The second half of
the assumption is where `mockup_aware_train_val_test_split` gets interesting. The
mockup groups named in `settings.MOCKUP_ARTWORK_IDS` are synthetic single-pigment
paint-on-support samples, **not** drawn from the same distribution as real aged
artworks. Holding a mockup group out as test data would therefore measure
performance on a distribution nobody cares about, while starving training of
samples that exist purely to be learned from. Splitting them at the pair level and
sending only 5% to test (`MOCKUP_TEST_RATIO`) is the correct response to a *known*
breach of the "identically distributed" half: keep them on the training side,
where distributional mismatch is a feature, and out of the evaluation, where it
would be a bias.

Reading the two split functions this way — one enforcing the first `i`, the other
managing a deliberate breach of the `d` — explains why they differ, rather than
just recording that they do.

> **Note.** An earlier data state had none of the six configured mockup IDs
> (`tblu`, `tbianco`, `tbruno`, `tgiallo`, `trosso`, `tverde`) present in `data/`,
> which made `mockup_aware_train_val_test_split` silently equivalent to
> `grouped_train_val_test_split`. The data and the configuration have since been
> reconciled — all six mockup groups are present — so the split now does what the
> argument above describes. See §8.11.

## 1.2 Capacity, overfitting, and where this project sits

**L03 slide 55**: "The factors determining how well a machine learning algorithm
will perform are its ability to: 1. Make the training error small. 2. Make the
**gap** between training and test errors small… Underfitting occurs when the model
is not able to obtain a sufficiently low error value on the training set.
Overfitting occurs when the gap between the training error and test error is too
large. We can control whether a model is more likely to overfit or underfit by
altering its **capacity**."
**L03 slide 56**: "Models with high capacity can overfit by **memorizing
properties of the training set** that do not serve them well on the test set."
**L03 slides 57, 59** and **L07 slide 2** (Goodfellow Figure 5.3) draw the
capacity-versus-error curve with its underfitting and overfitting zones either side
of an optimal capacity. **L07 slide 3** compresses it: "model complexity is the
number of independent parameters to be fit (degrees of freedom); complex model ⇒
more sensitive to data ⇒ more likely to overfit."

`unet` has **31,049,409** parameters. The dataset is **1167** matched pairs from
29 artworks, of which the training fold holds **820**. That ratio puts us
unambiguously to the right of optimal capacity, in slide 55's overfitting zone. §6.3 reaches the same place by a different route —
counting the bottleneck's dimension — and §3.8 expresses it in the lecture's own
units by comparing against AlexNet.

**Slide 56's phrase deserves attention, because here it has a specific and
dangerous form.** A model that memorizes a particular painting's IR appearance
produces a *small* residual everywhere on it — **including over a genuine
underdrawing**. Overfitting in this project does not merely inflate the test
error, it **erases the signal**. That is a stronger reason to care about capacity
than the usual one, and it is stated nowhere in the repository.

## 1.3 Mini-batch size

**L03 slide 44**: mini-batch gradient descent "reduces the **variance of the
parameter updates**, which can lead to more stable convergence… **Common
mini-batch sizes range between 50 and 256**, but can vary for different
applications."

`BATCH_SIZE = 8` is an order of magnitude below that band. AlexNet trained at 128
(L05 slide 54), ResNet at 256 (L05 slide 78).

The reason for 8 is real and physical: large inputs through a 1024-channel
bottleneck on a single Metal GPU. The course material raises two independent
warnings about this parameter — slide 44 on gradient-update variance here, L05
slide 48 on BatchNorm statistics in §3.5. The second was resolved by switching to
`GroupNormalization`, which is batch-size-independent (§3.5, §8.4); the first
stands, but with `EPOCHS = 100` and early stopping the update-variance cost is
tolerated rather than fixed.

## 1.4 Loss, cost, objective — one thing, four names

**L03 slide 20**: "The function we want to minimize or maximize is called the
**objective function** or **criterion**. When we are minimizing it, we may also
call it the **cost function**, **loss function**, or **error function**."

Recorded so that the codebase's interchangeable use of these terms reads as
sanctioned rather than sloppy. The distinction that *does* carry information is
loss versus metric — §0.15.

## 1.5 Cross-validation

**L03 slide 57**: "If model overfits, i.e. it is too sensitive to data, it will be
**unstable** — use split training/test or **k-fold cross validation**."
**L03 slides 58–59** then demonstrate model selection by cross-validation.

The project uses both. Day-to-day work runs on a single fixed split seeded by
`settings.SEED`; on top of that, `scripts/kfold.py` implements a **grouped k-fold**
(k = 3) — each fold holds out different whole artworks for validation, keeps all
mockups in training, and leaves the three ground-truth paintings external to every
fold. It was run on the two leading heteroscedastic models to put an error bar on
the detection metrics.

The result: fold-to-fold standard deviation of ≈ 0.01 AUROC, far below any ranking
gap. Slide 57's instability does not materialise at this scale — the single fixed
split was adequate, and cross-validation *confirms* its point estimates rather
than deflating them (§8.7, and [final-comments.md](final-comments.md) §5).

---

# Part 2 — Feed-forward networks and backpropagation (L04)

## 2.1 Why non-linearity at all

**L04 slides 9–12**: "The mapping between input and output is usually
**non-linear**. To better draw decision boundaries we need to add non-linearity
within our networks."
**L04 slide 13**: "The activation function allows the neural network to learn a
**non-linear pattern** between inputs and target output variable."

This is the premise beneath L08 slide 9's "nonlinear generalization of PCA"
(§6.5), and here it has identifiable physical content. The RGB→IR mapping is
non-linear because pigment IR reflectance is not an affine function of visible
colour, layer thickness modulates transparency non-linearly, and binder and
varnish contribute wavelength-dependent absorption. A linear model — which is
what a linear autoencoder, i.e. PCA, gives (L08 slides 6–8) — cannot represent any
of that. The 18 ReLU layers are what buy it.

## 2.2 The activation criteria, and the rule that settles the `tanh` question

**L04 slide 14** gives five criteria: keep the output "restricted to a certain
limit" to avoid computational blow-up; "**vanishing gradient problem — activation
functions should not shift the gradient towards zero**"; computationally
inexpensive, "since activation functions are applied after every layer millions of
times"; differentiable; and "a neural network will **almost always have the same
activation function in all hidden layers**".

Our configuration satisfies all five in the obvious way: one activation, ReLU,
everywhere in the hidden layers — cheap, differentiable almost everywhere, and
non-saturating for positive inputs.

**L04 slide 18** then states the assignment outright:

> "Multilayer Perceptron (MLP): **ReLU** activation function. Convolutional Neural
> Network (CNN): **ReLU** activation function. Recurrent Neural Network: **Tanh
> and/or Sigmoid** activation function."

§4.5 argues from first principles that we use ReLU and not `tanh` because there is
no recurrence for saturation to protect. Slide 18 gives the same conclusion as a
flat rule indexed by architecture family, and it is the better thing to cite,
because it is the course's own answer. §4.5 is *why* the rule holds; slide 18 is
the rule.

## 2.3 Vanishing gradients along depth, and the four enablers

**L04 slide 53**: "the lower you get in the network, the more the gradient
vanishes — **small × small × small = smaller**."
**L04 slide 54**: "Back-propagation is difficult for multiple hidden layers due to
the vanishing gradient… However, it was later found that DNN can also be
successfully trained: use **many labelled data**; **train longer** (possible with
GPUs); **better weight initialisation** (new methods were developed — Xavier);
**regularise with dropout**. Sometimes the use of rectified linear units improves
things, too."

Slide 53 is the feed-forward counterpart of L06's recurrent analysis (§4.7): the
same multiplicative compounding, along depth instead of along time. The
distinction §4.7 draws still holds — our 23 factors are 23 *different* matrices,
not one matrix raised to a power — but the compounding is real, and it is what the
~18 normalization layers ("improves gradient flow", L05 slide 48) and ResUNet's
additive shortcut (§4.8) are for.

Slide 54's four enablers make a useful scorecard:

| L04 slide 54 enabler | This project |
|---|---|
| many labelled data | **no** — 1167 patches from 29 artworks. This absence is what forces transfer learning (§3.7) and the self-supervised proxy task (§6.9) |
| train longer, with GPUs | yes — `tensorflow-metal`, `EPOCHS = 100` |
| better weight initialisation | **yes** — `he_normal` on every ReLU-preceding conv, the pairing L07 slide 37 prescribes (§5.5) |
| regularise with dropout | **no** — `dropout_rate` defaults to `0.0` and exists only in `unet_v2` (§0.13, §5.4); weight decay `1e-5` is used instead (§5.2) |
| rectified linear units | yes — everywhere |

Three of four met. Data is the one genuine absence, for the reasons in §3.7 and
§6.9; dropout was passed over in favour of weight decay.

---

# Part 3 — Convolutional neural networks (L05)

L05 supplies most of Part 0's vocabulary. This part takes its architecture-level
content: the progression from AlexNet to ResNet, and what our ten architectures
inherit from it.

## 3.1 Our conv blocks are VGG blocks

**L05 slide 63**, VGG-16/19 details: "**Only 3×3 CONV stride 1, pad 1 and 2×2 MAX
POOL stride 2**."

That one line is a complete specification of `_conv_block` plus `MaxPool2D(2)` in
[unet.py](scripts/unet.py): `Conv2D(filters, 3, padding="same")` *is* 3×3 stride 1
pad 1 (§0.4, since `(K−1)/2 = 1`), and `MaxPool2D(2)` *is* 2×2 stride 2. The
project's default architecture is VGG at the block level — which is what UNet was
in 2015, so this is inheritance, not coincidence.

## 3.2 Why stacked 3×3 rather than one large kernel

**L05 slide 64**: "a stack of three 3×3 conv (stride 1) layers has the same
effective receptive field as one 7×7 conv layer, **but deeper, more
non-linearities, and fewer parameters**: `3·(3²C²)` vs `7²C²` for C channels per
layer."

That is `27C²` against `49C²`. `_conv_block` uses two rather than three, which is
the same trade one step smaller: two `3×3` match one `5×5`'s receptive field at
`18C²` instead of `25C²`, with an extra non-linearity in between. The cumulative
consequence is the receptive-field table in §0.6, and the reason four levels are
needed at all is §6.10.

## 3.3 The channel schedule

**L05 slide 77**, on the full ResNet architecture: "periodically, **double # of
filters and downsample spatially using stride 2**."

`filters=[64, 128, 256, 512]` with a `MaxPool2D(2)` after each level is that
schedule with pooling substituted for the strided convolution;
[unet_v2.py](scripts/unet_v2.py)'s `use_strided_conv=True` switches to the exact
form the slide describes. L05 slide 23's "number of filters allows us to increase
or decrease channel size" is the mechanism.

The intent is standard: as resolution halves, channel count doubles, so
representational capacity is roughly conserved while the receptive field grows.
Note the consequence computed in §6.3 — after four halvings and doublings plus a
jump to 1024, the bottleneck holds *more* values than the input.

## 3.4 Pooling's invariance is half a liability here

**L05 slide 35**: "Pooling helps the representation become slightly **invariant to
small translations** of the input — if input is translated by small amount, values
of most pooled outputs don't change… its function is to progressively reduce the
spatial size of the representation to reduce the amount of parameters and
computation, and hence to also control overfitting."

For classification that is pure gain. For this project it is half a liability, and
naming why resolves what would otherwise look like an architectural
inconsistency.

We need to know **exactly where** a residual appears, to the pixel, because the
output is a map a conservator will read as evidence. Translation invariance is
precisely what we cannot afford at the output. Yet we downsample four times, for
two reasons and no others: it is what grows the effective receptive field (§0.6,
§3.2), and full-resolution convolutions throughout would be unaffordable (L08
slide 35, §6.14).

**Skip connections are the resolution of that tension** (§0.11): the encoder
discards position to gain invariance and receptive field, and the skip path hands
the position back at every scale. This is the deep reason UNet's skips are not
optional decoration here, and it is also why max unpooling — which passes
positions but discards values — is the wrong trade (§6.13).

## 3.5 BatchNorm: one slide, four hits, and the one that forced GroupNorm

§0.12 defines the operation. **L05 slide 48** is the single most load-bearing
slide in this mapping, because four of its bullets land directly on the original
configuration:

| L05 slide 48 says | In this project |
|---|---|
| "usually inserted after convolutional layers and **before** non linearity" | exactly the `Conv2D → Norm → ReLU` order, in every conv block |
| "**acts as regularization** during training" | the slide behind §6.3 — the normalization layers are a genuine regularizer, since capacity cannot be reduced. §5.1 notes this |
| "**behaves differently during training and testing**: this is a very common source of bugs!" | the general form of the dropout/normalizer variance shift that confines `dropout_rate` to two blocks in [unet_v2.py](scripts/unet_v2.py) — §5.4 |
| "**small size of the mini batch may affect the BatchNorm**" | `BATCH_SIZE = 8` — an order of magnitude below AlexNet's 128 (slide 54) and ResNet's 256 (slide 78) |

**The fourth row was a real problem, and it was fixed.** Batch statistics
estimated from 8 samples are noisy, and that noise enters ~18 layers per network.
The from-scratch architectures were switched from `BatchNormalization` to
**`GroupNormalization`** (Wu & He, 2018), which normalizes over channel groups
within each sample and is therefore independent of batch size — it makes the first
three rows above still hold (same position in the block, still a regularizer,
still a train/test-mode layer) while removing the small-batch failure mode
entirely. The pretrained EfficientNetB0 encoder keeps BatchNorm, since its running
statistics are part of the transferred ImageNet knowledge. `LayerNormalization`
(which `RestormerBlock` uses internally) would have been an alternative. See §8.4.

## 3.6 ResNet, degradation, and variable input sizes

**L05 slide 73**: "Is learning better networks as easy as stacking more layers? An
obstacle… was the notorious problem of vanishing/exploding gradient."
**L05 slide 75**: "56-layer model performs worse on **both test and training**
error → the deeper model performs worse, but **it's not caused by overfitting!**"
**L05 slide 76**: "use network layers to fit a **residual mapping** instead of
directly trying to fit a desired underlying mapping."
**L05 slide 77**: "every residual block has two 3×3 conv layers… no FC layers at
the end… **(in theory, you can train a ResNet with input image of variable
sizes)**."

Four things this project takes from those slides:

1. **Slide 75's diagnosis is the one to keep straight.** The degradation problem
   is an *optimization* failure — the deeper model is worse on the training set
   too. Our own depth concern (§1.2, §6.3) is the opposite kind: not too deep to
   optimize, but too high-capacity for 1167 patches. Different problem, different
   remedy. Confusing them would lead to adding residual connections to fix
   overfitting, which is not what they do.
2. **Slide 76's residual mapping** is `_residual_block` in
   [resunet.py](scripts/resunet.py) — with the caveat in §4.8 about its shortcut.
3. **Slide 77's "two 3×3 conv layers" per block** matches ours exactly.
4. **Slide 77's parenthesis is the project's central invariant, stated by the
   lecture.** "In theory, you can train a ResNet with input image of variable
   sizes", because there are no FC layers. That is the second of the project's core
   invariants — fully-convolutional models, no hard-coded spatial dimensions — and
   the entire reason a `3674×2834` painting can be predicted by a model trained on
   `400×400` patches. The lecture offers it as a curiosity in parentheses; here it
   is load-bearing (§6.11).

## 3.7 Transfer learning: the lecture asks our exact question

**L05 slides 81–83, 85–86** all carry the same subtitle: "**Can we use CNNs when
number of labeled samples is low?**" — with slide 82's finding that "neural
features are much more discriminative than hand-crafted features" and slide 81's
that off-the-shelf CNN representations "consistently outperform s.o.a. on multiple
tasks".

This is the project's binding constraint stated as the lecture's own motivating
question. We have 1167 patches from 29 artworks, and
[efficientnet_unet.py](scripts/efficientnet_unet.py) answers it exactly as slides
81–83 recommend: a **frozen** ImageNet-pretrained encoder (`freeze_encoder=True`
by default) with only the decoder trained. With this much data you do not learn an
encoder from scratch, you borrow one.

Two notes:

- The standard recipe has a **second phase** — unfreeze and fine-tune end-to-end
  at a lower learning rate once the decoder converges. This is implemented
  (`efficientnet_unet_ft`, `freeze_encoder=False`, warm-started from the frozen
  checkpoint) and was run: it *underperforms* the frozen baseline, because the
  paired dataset is too small to fine-tune an ImageNet backbone without eroding it.
  See §8.10.
- The backbone is EfficientNetB0 (2019). L05 slide 79 cites Bianco, Cadene,
  Celona and Napoletano's benchmark analysis of architecture complexity versus
  accuracy — the natural reference for choosing a replacement, and the
  professor's own paper.

## 3.8 Parameter counts in the lecture's terms

**L05 slide 52**: "AlexNet has about 660K units, **61M parameters**, and over 600M
connections. Notice: the convolutional layers comprise most of the units and
connections, but the **fully connected layers are responsible for most of the
weights**."
**L05 slide 70**: GoogLeNet — "only 5 million params! (removes FC layers
completely)… 12× less params than AlexNet".

Counted directly from the builders:

| Architecture | Parameters |
|---|---|
| `unet` / `unet_v2` (defaults) | 31,049,409 |
| `unet_nll` | 31,049,474 |
| `attention_unet` | 31,398,049 |
| `resunet` | 32,454,017 |
| `unet_restormer` | 41,603,785 |

(`efficientnet_unet` is omitted because instantiating it downloads ImageNet
weights — which is also why the test suite excludes it.)

Read against slide 52 this is informative. We sit at roughly half of AlexNet's
61M, but with **zero** fully-connected layers — where AlexNet kept most of its
weights. Every one of our ~31M parameters is convolutional. So by the measure
slide 52 actually cares about this is a substantially larger convolutional model
than AlexNet, and six times GoogLeNet's entire budget, fitted on an 820-patch
training fold rather than ImageNet's 14,197,122 images (slide 49). That is §1.2's and §6.3's
capacity-versus-data mismatch expressed in the lecture's own units.

`unet_restormer`'s extra 10.5M is the single `RestormerBlock` at the 1024-channel
bottleneck: one attention block costs a third again as much as the entire rest of
the network.

---

# Part 4 — Recurrent neural networks (L06)

## 4.1 The short answer

There is no recurrent network in this project and no plan for one. Verified rather
than assumed: `grep -rn "tanh\|LSTM\|GRU\|SimpleRNN\|recurrent" scripts/` returns
nothing.

The reason is the data. L06 slide 3 says RNNs "are designed to process
**sequences** of data `x₁,…,xₙ`" and introduce "cycles and a notion of **time**".
A painting has neither. It is a 2-D spatial object photographed twice, in two
bands, simultaneously. There is no ordering along which a hidden state could be
carried, and inventing one — raster-scanning the pixels, say — would impose an
arbitrary asymmetry, making the pixel above "past" and the pixel below "future",
on a signal that is isotropic in exactly the way the project's stroke-coherence
metric relies on ([stroke_stats.py](scripts/stroke_stats.py)).

Everything below is therefore either a structural analogy that clarifies a
decision we did make, or a mechanism whose *purpose* we achieve another way.

## 4.2 DAGs and cycles: the lecture draws our code

**L06 slide 2**: "Standard Neural Networks are DAGs (Directed Acyclic Graphs).
That means they have a topological ordering. The topological ordering is used for
activation propagation, and for gradient back-propagation. They process one input
instance at a time."
**L06 slide 3**: "Recurrent networks introduce **cycles** and a notion of time."

All ten architectures are on the slide-2 side of that line. And slide 2's own
illustration of what a feedforward DAG looks like is `Conv3x3 → ReLU → Conv3x3 →
+`, with the input routed around to the sum — which is `_residual_block` in
[resunet.py](scripts/resunet.py), line for line. The lecture's example of a DAG is
a block we shipped.

## 4.3 Parameter sharing over time vs over space — and one crucial asymmetry

**L06 slide 6**: deep RNNs stack layers vertically, annotated "**Same parameters
at this level**".
**L06 slide 29**: "Notice: the same function and the same set of parameters are
used at every time step."

This is the deepest genuine connection in the lecture. Both an RNN and a CNN are
**parameter-sharing schemes over an index set** — the RNN over *time*, the
convolution over *spatial position* (L05 slide 13). Same idea, different index. In
both cases the payoff is identical: because the parameters do not depend on the
index, the model accepts inputs of a size it never saw in training. L06 slide 4
states it for sequences — the unrolled DAG's "size depends on the input sequence
length" — and it is why our `(None, None, 3)` input works on a `3674×2834`
painting after training on `400×400` patches.

**But there is an asymmetry that decides the whole inference story**, and a loose
reading of the analogy gets it backwards:

| | Unrolled RNN | Fully-convolutional CNN |
|---|---|---|
| Shares parameters over | time steps | spatial positions |
| Parameter count as input grows | **constant** | **constant** |
| Graph *depth* as input grows | **grows linearly** (L06 slide 4) | **constant** — 23 conv layers, always |
| Tensor shapes as input grows | constant per step | grow |
| Consequence | longer input ⇒ longer gradient chain ⇒ vanishing/exploding (§4.7) | larger input ⇒ more memory, same gradient chain |

Processing a `3674×2834` image does **not** make our network deeper. That single
fact is why whole-image inference on images sixty-five times the area of a
training patch shows none of L06's pathologies, and why the open question in
[code-review.md](code-review.md) §5.5 is about memory and border context, not about
optimization stability.

## 4.4 The architecture taxonomy: where our task sits

**L06 slides 21–25** enumerate the shapes:

| Slide | Shape | Example given |
|---|---|---|
| 21 | one-to-one | feed-forward network |
| 22 | one-to-many | image captioning: image → sequence of words |
| 23 | many-to-one | sentiment classification: sequence → label |
| 24 | many-to-many (encoder–decoder) | machine translation: seq of words → seq of words |
| 25 | many-to-many (aligned) | video classification on frame level |

Our task is not in this taxonomy, because it is not sequential. But it has a clear
spatial analogue, and picking the right one is illuminating: RGB→IR is the spatial
counterpart of **slide 25**, the *aligned* many-to-many — one output per input
position, same extent, position-preserving. It is emphatically **not** slide 24's
encoder–decoder shape.

**This distinction explains UNet's skip connections.** In slide 24's machine
translation the encoder compresses the entire input into one state and the decoder
generates from it; input and output lengths differ and there is no positional
correspondence between them. That architecture has an information bottleneck with
*no alignment*, and the historical fix was attention — a learned alignment
restored on top.

UNet looks superficially like slide 24 (encoder, bottleneck, decoder) but its task
is slide 25's: pixel `(i, j)` in must become pixel `(i, j)` out. The alignment is
not something to be learned, it is **given**. Skip connections are how the
architecture exploits that gift — they carry position-indexed features straight
across at every scale, so the bottleneck never has to encode *where* anything was.
In that sense UNet's skips do for dense prediction what attention does for
seq2seq: they defeat the encoder–decoder bottleneck. But they are cheaper and
stronger, because alignment is free here and had to be inferred there.

## 4.5 The recurrence formula and the output projection

**L06 slides 28–30**: a recurrence `hₜ = f_W(hₜ₋₁, xₜ)` applied at every step,
where "the state consists of a single 'hidden' vector `h`" (slide 30), followed by
an output projection from that state.

Two mappings:

- The output projection `yₜ = W_hy·hₜ + b` is applied identically at every time
  step. Our `Conv2D(1, 1, activation="sigmoid")` (§0.9) is the same object over a
  different index: a learned linear map from the hidden representation to the
  output, applied identically at every *position*.
- **`tanh` versus ReLU.** L06's vanilla RNN uses `tanh`, and LSTM uses it twice
  more (slides 66, 74 — "the output will be a filtered version by the `tanh` of
  the updated cell state"). We use it nowhere. The reason is §4.3's asymmetry:
  `tanh` is bounded and zero-centred, which matters when the *same* activation is
  applied to its own output dozens of times and the values must not drift. Along
  our 23-layer path each activation is applied once, with a BatchNorm before it
  re-centring the distribution, so the saturating tails buy nothing and cost
  gradient. L04 slide 18 states the resulting rule directly (§2.2).

One honest note. `ClipLogVar` ([nll_layers.py](scripts/nll_layers.py)) hard-clips
`log_var` to `[-6, 6]`, and a hard clip has exactly the failure mode of a
saturated `tanh`: zero gradient outside the range. The difference is that clipping
is only reached in regimes we already consider pathological (a variance below
`e⁻⁶` or above `e⁶`), whereas `tanh` saturates inside the normal operating range.
It is a bound of last resort, not an activation.

## 4.6 Truncated BPTT and tiled inference

**L06 slide 43**: full BPTT is "forward through entire sequence to compute loss,
then backward through entire sequence to compute gradient."
**L06 slide 44**: truncated BPTT instead "run[s] forward and backward through
**chunks** of the sequence instead of whole sequence — carry hidden states forward
in time forever, but only backpropagate for some smaller number of steps."

This is the closest methodological analogue in the lecture to a real, still-open
question here. Truncated BPTT is what you do when the thing you would like to
process whole does not fit, and it is a **deliberate approximation with a known
bias**: severing the chunk boundary discards dependencies that cross it.

The spatial counterpart is meaning #4 of "sliding window" (§0.2). The
tiled-inference path this project once used processed a large painting in tiles
and Gaussian-blended them, and the blending existed for precisely the reason slide
44's truncation is approximate — a tile boundary severs spatial context, and the
blend hides the seam rather than restoring what was lost.

The parallel sharpens [code-review.md](code-review.md) §5.5 rather than resolving
it. Whole-image inference has *no* boundary-severing bias: it is the spatial
equivalent of full BPTT, the exact computation. What it still does not settle is
whether a model trained on `400×400` patches behaves the same when its normalizer
statistics and receptive-field context meet a `3674×2834` input — a
distribution-shift question, not an approximation question. The comparison remains
unrun.

## 4.7 Vanishing and exploding gradients, and three kinds of clipping

**L06 slides 53–63** are devoted to vanilla RNN gradient flow (the mathematics is
in figures, so it is described here rather than quoted).
**L06 slide 79** gives the summary in text: "Backward flow of gradients in RNN can
explode or vanish. **Exploding is controlled with gradient clipping. Vanishing is
controlled with additive interactions (LSTM).**"

The RNN's problem has a specific cause: backpropagating through `T` steps
multiplies by the *same* recurrent matrix `T−1` times, so the gradient is governed
by one matrix's singular values raised to a power — exponential in sequence
length.

Our 23-layer path does not have that structure, for two independent reasons:

1. **Different matrices.** Each of the 23 convolutions has its own weights, so
   there is no exponential in a single spectrum. L04 slide 53's
   "small × small × small" compounding still applies (§2.3), but without the
   single-operator amplification.
2. **~18 normalization layers.** L05 slide 48 lists "improves gradient flow" among
   normalization's benefits, and re-standardizing activations ~18 times along the
   path (`GroupNormalization`, §3.5) intervenes directly on the quantity that
   compounds.

L05 slide 73 is where the two lectures meet on this: "is learning better networks
as easy as stacking more layers? An obstacle… was the notorious problem of
vanishing/exploding gradient." The CNN answer to that obstacle is §4.8.

**We use no gradient clipping**, deliberately: `clipnorm`/`clipvalue` are
available on `tf.keras.optimizers.Adam` and [trainer.py](scripts/trainer.py) sets
neither. With no recurrence there is no exponential amplification to control, and
Adam's per-parameter scaling plus `ReduceLROnPlateau` handle ordinary instability.

**A disambiguation in the spirit of §0.2**, because "clipping" appears three times
in this codebase meaning three different things:

| Kind | What is clipped | Where | Purpose |
|---|---|---|---|
| **Gradient clipping** (L06 slide 79) | the gradient vector, during backprop | **nowhere** | would control exploding gradients — not needed |
| **Activation clipping** | the `log_var` channel, forward pass | `ClipLogVar`, [nll_layers.py](scripts/nll_layers.py) | numerical stability of `exp(0.5·log_var)` |
| **Value clipping** | pixel values, on data | `tf.clip_by_value(rgb, 0, 1)` in [augmentation.py](scripts/augmentation.py) | keep jittered RGB in valid range |

Only the first is what slide 79 means. The other two are unrelated operations that
share a verb.

## 4.8 Slide 78: the lecture hands us ResUNet

**L06 slide 78**: "Recall: 'PlainNets' vs. ResNets — similarity with CNNs.
**ResNet is to PlainNet what LSTM is to RNN, kind of.**"
**L06 slide 77** draws the contrast that makes the analogy work: the vanilla RNN
state passes through `f → f → f`, a **multiplicative** chain, while LSTM (ignoring
forget gates) is `f + , f + , f +` — an **additive** path alongside.
**L06 slide 79**: "Common to use LSTM: their **additive interactions improve
gradient flow**."

This is the lecture explicitly supplying the justification for
[resunet.py](scripts/resunet.py). The README motivates ResUNet as "residual blocks
in encoder/decoder; better gradient flow on small datasets" — correct, but it
reads as a citation to He et al. Slide 78 says something stronger and more
transferable: the additive shortcut is *the same fix* as LSTM's cell state, and it
addresses *the same failure*. A residual block is a gradient highway; so is an
LSTM.

**And here the mapping earns its keep, because our implementation does not match
the picture.** Slide 77's additive path is uninterrupted — the state is added
straight through. [resunet.py](scripts/resunet.py)'s shortcut is not:

```python
shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
shortcut = GroupNormalization(groups=num_groups(filters))(shortcut)
...
x = layers.Add()([x, shortcut])
```

The shortcut passes through a learned `1×1` convolution and a normalization layer
on **every block, unconditionally**. That is He et al.'s *projection* shortcut,
which the original paper uses only where channel counts change. So the
"uninterrupted
additive path" slide 77 identifies as the mechanism is, in our ResUNet,
interrupted at every block by a learned transform. The gradient highway is real
but tolled.

By contrast `RestormerBlock` ([unet_restormer.py](scripts/unet_restormer.py)) has
the pure form:

```python
x = inputs + self._attention(self.norm1(inputs))
x = x + self._feed_forward(self.norm2(x))
```

Two true identity paths, exactly slide 77's shape. Whether an identity shortcut
where channels already match would help was checked directly — it turns out every
`_residual_block` in this architecture *always* changes channel count (one block
per depth level, widening on the way down and narrowing on the way up), so a
conditional identity path would never fire. The unconditional projection stays.
See §8.9.

## 4.9 Gating: LSTM's mechanism applied to space instead of time

**L06 slide 66**: from `x` and `h`, one matrix produces four vectors `i f o g`
through `sigmoid, sigmoid, sigmoid, tanh`.
**L06 slides 68–74**: the forget layer decides "what information we are going to
throw away from the cell state"; the input gate, "what new information we are
going to store"; the output, "a filtered version by the `tanh` of the updated cell
state". Every one is a sigmoid multiplied into a signal.

`_attention_gate` in [attention_unet.py](scripts/attention_unet.py) is
structurally the same object:

| | LSTM forget gate (slides 66–68) | `_attention_gate` (Oktay et al.) |
|---|---|---|
| Inputs | current input `x`, previous state `h` | encoder skip `x`, decoder gating signal `g` |
| Coefficient | `f = sigmoid(W·[h, x])` | `ψ = sigmoid(conv(ReLU(conv(x) + conv(g))))` |
| Applied as | `f ⊙ cₜ₋₁` | `ψ ⊙ x` |
| Index set | time | **space** — one coefficient per pixel |
| Question answered | "how much of the past do I keep?" | "how much of this skip connection is relevant here?" |

Same computation, different index — §4.3's pattern once more. The `1×1`
convolutions play the role of the LSTM's weight matrix, and the sigmoid produces
the same `[0, 1]` soft switch. Reading Oktay's gate as "an LSTM forget gate over
pixels instead of timesteps" is, as far as this mapping can tell, exactly right.

The second gate, `RestormerBlock`'s GDFN, is a looser relative: `gelu(x1) * x2`
splits a tensor in half and lets one half modulate the other. Multiplicative
gating, but unbounded — no sigmoid, so no "how much passes through" reading.

## 4.10 Autoregressive generation, and why we do not

**L06 slide 42**: "At test-time sample characters one at a time, feed back to
model."

Worth recording as a rejected option rather than an irrelevant one, because a
spatial version exists — PixelRNN and PixelCNN generate an image pixel by pixel,
each conditioned on those already produced.

We predict in a single forward pass. Beyond the cost argument — an autoregressive
pass over `3674×2834` pixels is roughly ten million sequential steps — there is a
domain reason, the same one [code-review.md](code-review.md) §6 gives for
deprioritizing an adversarial term: **for a forensic and scholarly instrument, a
plausible-looking fabricated mark is worse than a missed one.** Feeding a model's
own output back as input is the mechanism by which small errors compound into
confident invention. A single-shot prediction can be wrong, but it cannot talk
itself into a hallucinated underdrawing. Given that the product is a residual a
conservator will read as evidence, that asymmetry decides it.

## 4.11 When recurrence would actually become relevant

So the "not applicable" verdict is bounded rather than absolute: there is one
plausible route. [code-review.md](code-review.md) §6 raises a multi-band /
multispectral extension (MST++). A spectral stack — RGB, IR, UV, X-ray, or many
narrow bands — has an ordered axis along which values are correlated, and that
axis is far more sequence-like than space is. Sequence models are a legitimate
family for it.

They would still probably lose. The state of the art on spectral reconstruction is
attention-based, not recurrent, and this project already has the relevant
machinery: `RestormerBlock`'s MDTA computes attention **across channels**, which
is precisely attention over a spectral axis. If a multispectral extension ever
happens, that block is the natural place to put it — not an LSTM.

---

# Part 5 — Regularization (L07)

L07 speaks most directly to this project's central problem — 31M parameters
against an 820-patch training fold — so this part is less analogy and more
scorecard.

## 5.1 The definition, and the canonical four

**L07 slide 22**: "We define regularization as **any modification we make to a
learning algorithm that is intended to reduce its generalization error but not its
training error**."
**L07 slide 23** lists the methods: "Parameter Norm Penalties → **weight decay**;
**Dropout**; **Early stopping**; Someone includes **data augmentation** in the
list…"

Where this project stands on each — the most useful single table in this file:

| Slide 23 method | Status here | Where |
|---|---|---|
| **Early stopping** | **yes, fully**, and implemented exactly as slides 40–41 describe | `EarlyStopping(patience=20, min_delta=5e-4)` + `ModelCheckpoint(save_best_only=True)` — §5.6 |
| **Data augmentation** | **yes**, with a domain-specific asymmetry the lecture's taxonomy explains | [augmentation.py](scripts/augmentation.py) — §5.3 |
| **Weight decay** | **yes** — `1e-5` in the Adam optimiser, matching L05's ResNet recipe | §5.2 |
| **Dropout** | **no** — `dropout_rate=0.0` by default, only in `unet_v2`, never enabled | [unet_v2.py](scripts/unet_v2.py) — §5.4 |

Plus one the lecture does not list but L05 slide 48 does: normalization
"acts as regularization during training", and there are ~18 `GroupNormalization`
layers per network (§3.5). So the project's regularization is early stopping,
augmentation, weight decay, GroupNorm, and a frozen pretrained encoder.

Note what slide 22's definition **excludes**. The artwork-grouped split (§1.1) is
*not* regularization: it does not modify the learning algorithm, it fixes the
evaluation. The distinction matters, because no amount of regularization repairs a
leaky split and no split repairs an over-capacity model.

## 5.2 Weight decay

**L07 slides 24–25**: "Many regularization approaches are based on limiting the
capacity of models… by adding a **parameter norm penalty** to the objective
function… L2 regularization is also known as ridge regression or Tikhonov
regularization. In deep learning community it is commonly known as **weight
decay**. This regularization strategy drives the weights closer to the origin."
**L07 slide 28** works through the L2 gradient interpretation.

`compile_model` / `compile_model_nll` pass `Adam(weight_decay=settings.WEIGHT_DECAY)`
with `WEIGHT_DECAY = 1e-5` — decoupled weight decay (AdamW), applied to every
weight in every architecture. The value is taken directly from L05's ResNet recipe
("weight decay of 1e-5", slide 78); AlexNet used `5e-4` (slide 54) but at a far
larger batch and dataset.

The course material warrants this from two directions: slide 23 lists weight decay
**first** among the most-used methods, and both of L05's worked recipes use it.
For a project whose central problem is excess capacity (§1.2, §6.3) it is the
cheapest regularizer available — one optimiser argument, no architecture change.
See §8.5.

## 5.3 Data augmentation, and the rule that explains our asymmetry

**L07 slide 5**: "data augmentation approaches overfitting **from the root of the
problem, the training dataset**… under the assumption that more information can be
extracted from the original dataset through augmentations."
**L07 slide 6**, the taxonomy: "these augmentations artificially inflate the
training dataset size by either data warping or oversampling. **Data warping
augmentations transform existing images such that their label is preserved.** This
encompasses augmentations such as **geometric** and **colour** transformations,
random erasing, adversarial training, and neural style transfer."

Slide 6's four words — **"their label is preserved"** — are the exact
justification for project invariant #5, which the repository states as a rule
without a principle ("spatial transforms apply to RGB *and* IR with a shared box;
photometric jitter applies to RGB only").

Worked through slide 6's own two categories:

| Slide 6 category | Our transform | Label preserved if… | Therefore |
|---|---|---|---|
| **Geometric** | horizontal/vertical flip, random crop | applied to RGB **and** IR with the *same* parameters | flip both; one shared crop box, obtained by concatenating RGB+IR on the channel axis before cropping |
| **Colour** | brightness / contrast jitter | applied to RGB **only** | jittering the IR target would produce a pair whose IR is *not* the true IR of that RGB — a corrupted label |

The asymmetry is not a heuristic about physics that happens to work. It is slide
6's label-preservation constraint applied to a task whose label is an *image*: a
geometric transform moves the label and so must be applied to it, while a
photometric transform on the input leaves the label untouched — and applying one
to the label would break the very correspondence the model is learning. The
project's physical framing ("IR reflectance is a physical property, not an
illumination artefact") is the domain reason the same conclusion holds; slide 6 is
the general principle. This is worth promoting into the repo — §8.3.

The remaining items in slide 6's list are rejected or untried, and two connect to
decisions already on record. **Adversarial training** and the GAN route (slide 7)
are deprioritized in [code-review.md](code-review.md) §6 because "a
plausible-looking but fabricated mark would be worse than a missed one" — the same
argument §4.10 gives against autoregressive generation. **Neural style transfer**
(slide 8, offered for "few-shot learning — augmenting tiny datasets") is more
interesting than it looks, because the closest on-domain precedent in the
project's own bibliography — Cann et al. 2021, recovering underdrawings on works
by Leonardo — is a style-transfer method. Untried; noted in
[code-review.md](code-review.md) §6 alongside the GAN route.

Note finally that **L07 slide 4** lists "**increase the amount of data**, using
augmentation, data generation or **new real data (data-centric AI)**" as a remedy
*distinct* from regularization. The mockup groups are exactly that: physically
created new real data, made to densify the pigment manifold (§6.7). The project
did the data-centric thing before doing the regularization thing (see §1.1 and
§8.11 on the data/configuration reconciliation that made the mockup split active).

## 5.4 Dropout: an ensemble, and only at training time

**L07 slide 30** gives the intuition — 50 people planning a conspiracy: "strategy
A: plan a big conspiracy involving 50 people, likely to fail; strategy B: plan 10
conspiracies each involving 5 people, likely to succeed."
**L07 slide 32**: "Dropout trains an **ensemble** consisting of all sub-networks
that can be constructed by removing non-output units from an underlying base
network… **Dropout is only used during training.**"
**L07 slide 33**: with dropout probability `p`, each activation is replaced by a
random variable.

Slide 32's last sentence is the root of the constraint shaping
[unet_v2.py](scripts/unet_v2.py). Dropout is active in training and inactive at
inference, so activation statistics differ between the two phases. L05 slide 48
names the general hazard ("behaves differently during training and testing: this
is a very common source of bugs"); Li & Xu (README ref [16]) name the specific
mechanism, variance shift, for the dropout–BatchNorm interaction in particular.
The project's normalizer is now `GroupNormalization`, which does not carry
train-time running estimates and so is less exposed to this than BatchNorm was —
but `dropout_rate` was still scoped conservatively to only the bottleneck and the
first decoder block, the two places where overfitting risk is most concentrated
and fine underdrawing-relevant detail is least at stake.

The honest status: dropout is **off** in every checkpoint this project has
produced. It is an implemented, tested, unused option — which leaves early
stopping and augmentation doing nearly all the declared regularization work, with
GroupNorm and weight decay doing the rest (§5.1, §5.2).

## 5.5 Weight initialization: the wrong default for our activation

**L07 slide 35**: "The goal of **Xavier initialization** is to initialize the
weights such that the **variance of the activations is the same across every
layer**. This constant variance helps prevent the gradient from exploding or
vanishing." Its derivation assumes, explicitly, "we use the **tanh()** activation
function, which is approximately linear with small inputs".
**L07 slide 37**: "Xavier initialization is designed to work well with **tanh or
sigmoid** activation functions. **For ReLU activations, the He initialization is
more effective.** He and colleagues argue that the Xavier initialization **does not
work well with the ReLU activation function**, and instead propose an
initialization of `σ² = 2/N`."

Every ReLU-preceding convolution in the from-scratch architectures sets
`kernel_initializer="he_normal"` — the He initialisation slide 37 prescribes for
ReLU, `σ² = 2/N`. This was a deliberate change from the Keras default
(`GlorotUniform` / Xavier), which slide 37 singles out as ineffective for ReLU:
Xavier is derived under a `tanh` assumption (slide 35) and under-scales the
variance for a rectifier.

Two notes. The ~18 `GroupNormalization` layers already mitigate a poor
initialisation — re-standardizing activations at every block directly fixes the
variance drift initialisation schemes exist to prevent — so the change is a
refinement, not a rescue; and L05 slide 78 records ResNet itself trained with
Xavier and ReLU. But He is the pairing the lecture recommends and it costs one
argument per layer. See §8.6.

## 5.6 Early stopping: the slide describes our callbacks

**L07 slide 40** quotes Goodfellow's *Deep Learning* section 7.8: "we often observe that training error
decreases steadily over time, but validation set error begins to rise again… we
can obtain a model with better validation set error by **returning to the
parameter setting at the point in time with the lowest validation set error**.
**Every time the error on the validation set improves, we store a copy of the
model parameters.** When the training algorithm terminates, we **return these
parameters, rather than the latest parameters**."

That paragraph is a specification of the exact callback pair in
[trainer.py](scripts/trainer.py), and the correspondence is worth making explicit
because the two callbacks are easy to mistake for redundant:

| Slide 40 sentence | Callback |
|---|---|
| "every time the error on the validation set improves, we store a copy of the model parameters" | `ModelCheckpoint(monitor="val_loss", save_best_only=True)` — persists best weights to `best_model.keras` |
| "return these parameters, rather than the latest parameters" | every evaluation loads `best_model.keras`, never the in-memory end-of-run model |
| stop when it stops improving | `EarlyStopping(monitor="val_loss", patience=20, min_delta=5e-4)` |

The `ModelCheckpoint` is the load-bearing half: it writes the best weights to disk,
which is what every downstream step reads. `EarlyStopping` only decides *when* to
stop (its `restore_best_weights` is off, since the checkpoint already holds the
best weights). All three callbacks monitor `val_loss` uniformly. This is the one
regularizer in slide 23's list the project implements completely and canonically.

## 5.7 Universality — and why our mapping is not a function

**L07 slides 42–44**: "No matter what the function, there is guaranteed to be a
neural network so that for every possible input `x`, the value `f(x)` (or some
close approximation) is output from the network… this universality theorem holds
even if we restrict our networks to have just a single hidden layer" (Cybenko,
1989).
**L07 slide 45**, caveat 1: not exact computation, "rather, we can get an
approximation that is as good as we want", improved by adding hidden neurons.
**L07 slide 47**, caveat 2: "the class of functions which can be approximated…
are the **continuous** functions."

The theorem is reassuring about capacity and says nothing about our actual
difficulty — and articulating why sharpens §6.6's argument considerably.

Universality guarantees the approximation of a **function**: one output per input.
§6.6's central physical claim is that RGB→IR **is not a function**. Pigments
indistinguishable in visible light have different IR reflectance, so the same RGB
value legitimately maps to several IR values. **There is no `f` for the theorem to
approximate.**

This reframes what the deterministic architectures are doing. Given a one-to-many
relation and a squared- or absolute-error objective, a network that can represent
any function converges to the *conditional mean* of the fibre — the best
single-valued approximation of a multi-valued relation. It is not failing to
learn; it is succeeding at a problem that was mis-specified. Adding capacity
cannot help, which is exactly what slide 45's "increase the number of hidden
neurons" cannot fix here: the limitation is in the object being modelled, not in
the approximator.

The `*_nll` models sidestep it by changing the object. Predicting `(mu, log_var)`
turns a one-to-many relation into a genuine single-valued function — from RGB to
the *parameters of a distribution* over IR. Universality then applies again, to
that function. **This is the strongest available argument for the heteroscedastic
head**, stronger than the physical framing in [README.md](README.md) and the
geometric framing in §6.6, because it identifies the deterministic model's ceiling
as a category error rather than a shortfall. See §8.1.

Caveat 2 is a smaller note in the same direction: a real pentimento boundary is
close to a discontinuity in the IR image, and slide 47 excludes discontinuous
functions. In practice "a continuous approximation is good enough", but it does
predict that the sharpest edges are where the prediction is weakest — and since
the residual is the product (§6.2), prediction weakness at edges is precisely what
inflates `raw_delta` along every contour and motivated `structural_delta` (§6.5).

## 5.8 Unbalanced data and the evaluation metrics

**L07 slide 48**: "Classes have often unequal frequency… **majority class
classifier can be 99% correct but useless.** Should we rebalance the data set for
training? Some say yes, but only if the ratio is extremely high."
**L07 slide 49**: "Problems with the accuracy: **assumes equal costs for
misclassification**; **assumes relatively uniform class distribution**."
**L07 slides 50–51**: precision, recall/sensitivity/true-positive-rate,
specificity, **false positive rate = 1 − specificity**, from the confusion matrix.
**L07 slide 52**: the F-measure, "note, however, that the F-measures **do not take
the true negatives into account**".

These four slides are the theoretical grounding for
[detection.py](scripts/detection.py), which §6.9 introduces without one.

- **Our problem is exactly slide 48's.** Hidden-detail pixels are a small fraction
  of a painting. A detector reporting "no hidden detail" everywhere scores
  extremely well on accuracy — slide 48's "99% correct but useless" in its purest
  form.
- **Which is why accuracy appears nowhere in this project.** Slide 49's second
  bullet is the reason, and the code acts on it:
  [detection.py](scripts/detection.py) reports AUROC and average precision, never
  accuracy.
- **AUROC is built from slide 51's two rates.** The ROC curve plots sensitivity
  (`TP / actual positive`) against false positive rate (`FP / actual negative`),
  both defined on that slide. Because each is normalized by its own class total,
  AUROC is prevalence-independent — which is why the code review calls it the
  primary ranking.
- **Average precision needs prevalence beside it, for slide 52's reason.**
  Precision (slide 50, `TP / predicted positive`) ignores true negatives, so unlike
  AUROC it moves with class balance: AP's chance level *is* the prevalence.
  [detection.py](scripts/detection.py) reports prevalence alongside AP precisely so
  the number can be read against its own baseline — a direct application of slide
  52's caveat.
- **Slide 48's rebalancing question we answer "no"**, but not because the ratio is
  insufficiently extreme. The imbalance is in the *evaluation* mask, not the
  training labels: the proxy task (§6.9) trains on IR regression, where there is no
  class imbalance to rebalance, and the imbalance appears only when the residual is
  scored against hand-drawn masks. Rebalancing is not applicable — a cleaner answer
  than "the ratio is not extreme enough".

---

# Part 6 — Autoencoders and semantic segmentation (L08)

The lecture the project is actually built on. The first half supplies the
topology; the second half supplies the task family.

## 6.1 Encoder–decoder: what we take, what we deliberately break

**L08 slide 2**: an autoencoder is "a neural network that is trained to attempt to
copy its input to its output. Internally, it has a hidden layer `h` that describes
a **code** used to represent the input… an **encoder** function `h = f(x)` and a
**decoder** that produces a reconstruction `r = g(h)`."

All ten architectures have that topology — [unet.py](scripts/unet.py) is an
encoder `[64,128,256,512]` → bottleneck 1024 → symmetric decoder — but they are
**not autoencoders**: the input is RGB, the target is IR. This is cross-modal
*translation*, not reconstruction, and the difference is not cosmetic:

- **The bottleneck is not there to prevent copying.** In an autoencoder,
  compression is the regularizer that blocks the identity function. Here input and
  target live in different modalities, so the identity is not even an admissible
  solution. The bottleneck exists to aggregate *context* (§6.10).
- **UNet's skip connections would be pathological in an autoencoder.** They are a
  shortcut letting `g(f(x)) = x` bypass the code entirely — precisely the failure
  L08 slide 9 warns about ("learn to copy without extracting useful information
  about the distribution of the data"). Here they are *beneficial*: the skip path
  carries spatial structure that must be **re-mapped** into IR reflectance, not
  copied (§3.4).
- **No leakage into the signal.** The model never sees `real_IR` as an input. The
  delta is computed downstream, outside the network.

## 6.2 The central inversion: the failure *is* the product

**L08 slide 2**: *"If an autoencoder succeeds in simply learning to set g(f(x)) = x
everywhere, then it is not especially useful. Instead, autoencoders are designed to
be unable to learn to copy perfectly."*

That sentence describes this project literally, with one reversal. We optimize for
faithful reconstruction (MAE / SSIM / PSNR), but **the useful output is the
residual** — where reconstruction fails. The slide says the
autoencoder is designed to fail at copying; we say the network should predict the
IR correctly *everywhere except* where hidden content exists, and that localized
failure is the result.

Everything methodologically hard about this project follows: telling **"it failed
because there is an underdrawing"** apart from **"it failed because the model is
mediocre there"**. A large delta is not informative by itself. Hence the
three-stage refinement:

| Signal | What it quotients away | Where |
|---|---|---|
| `raw_delta = \|real − pred\|` | nothing — conflates everything | baseline |
| `structural_delta = 1 − structure` | local gray-level and contrast shifts | [delta_analysis.py](scripts/delta_analysis.py), §6.5 |
| `structural_z = structural_delta / σ` | + the pigment's intrinsic ambiguity | [calibration.py](scripts/calibration.py), §6.6 |

It is also why §1.2's overfitting concern is sharper than usual: a memorized
painting yields a small residual *over the underdrawing too*.

## 6.3 Undercomplete or overcomplete? The arithmetic says overcomplete

**L08 slide 5**: "One way to obtain useful features from the autoencoder is to
constrain `h` to have smaller dimension `M` than `x`… Learning an undercomplete
representation forces the autoencoder to capture the most salient features."
**L08 slide 9**: *"if the encoder and decoder are allowed too much capacity, the
autoencoder can learn to perform the copying task without extracting useful
information about the distribution of the data. A similar problem occurs… in the
**overcomplete** case in which the hidden code has dimension greater than the
input."*

Run the numbers for a `400×400` patch:

```
input:      400 × 400 × 3                   =  480,000 values
bottleneck: (400/16) × (400/16) × 1024
          =  25 × 25 × 1024                 =  640,000 values

ratio = 1.333×
```

The "bottleneck" is **1.33× the input**. In slide 5's dimensional sense this
network is **overcomplete**, not undercomplete — and it has skip connections on
top. We are squarely in the regime slide 9 flags as problematic, and §1.2 reaches
the same conclusion from the capacity side, §3.8 from the parameter-count side.

This justifies the project's countermeasures — all of them *non-architectural*,
because compressing the bottleneck would destroy the high-frequency content that
**is** the signal:

- ~18 `GroupNormalization` layers (§3.5) — L05 slide 48's "acts as regularization"
- weight decay `1e-5` (§5.2)
- EarlyStopping + ReduceLROnPlateau (§5.6)
- Augmentation (§5.3)
- A **frozen** ImageNet encoder in
  [efficientnet_unet.py](scripts/efficientnet_unet.py) (§3.7)
- `dropout_rate` in [unet_v2.py](scripts/unet_v2.py), off by default (§5.4)

## 6.4 Why the loss is not `‖x − x̃‖²`

**L08 slides 6–8** put squared error at the centre of the PCA connection: the
reconstruction error is "the sum over all these unrepresented directions of the
**squared** differences of the datapoint from the mean" (slide 6), "the squared
distance between red and green points" (slide 7), and a linear autoencoder
minimizing "the squared reconstruction error" is "exactly what PCA does" (slide 8).

We use MSE nowhere. Two reasons:

1. **L2 produces blurry, mean-reverting predictions.** On a painting this is fatal
   *because the residual is the product* (§6.2): a blurry prediction inflates
   `raw_delta` along every edge, and edge noise drowns the signal. Hence
   `combined_loss` = MAE + (1 − SSIM) — MAE punishes outliers less harshly than
   MSE, and the SSIM term is explicitly structural.
2. **We need control in the frequency domain.** `combined_loss_advanced` (for
   `efficientnet_unet` only) adds a Laplacian pyramid term and an FFT-magnitude
   term. Note the parallel with the lecture's second half: the
   `_downsample`/`_upsample_to` helpers in [losses.py](scripts/losses.py) are the
   same machinery as L08 slides 36–41, **applied inside the loss instead of inside
   the network**. It decomposes the error into frequency bands and weights fine
   detail more heavily, rather than letting the network trade away high-frequency
   accuracy to minimize low-frequency error.

## 6.5 PCA and local standardization: `structural_delta` is a correlation

**L08 slides 6–8**: PCA projects onto `M` orthogonal directions; the reconstruction
error is the squared distance in the unrepresented directions (slide 7's red point
versus green point).

There is an exact mathematical link here, not an analogy. In
[delta_analysis.py](scripts/delta_analysis.py) the `structure` term is

```
structure = (cov + c3) / (std_real · std_pred + c3)
```

which is the (regularized) **local Pearson correlation** between real and
predicted IR over an `11×11` Gaussian window. Correlation is covariance *after*
standardizing both variables by their local mean and standard deviation — the same
centring-and-whitening PCA performs globally before projecting, done here per
window.

**Consequence:** local standardization quotients away the first and second
moments, so `structural_delta = 1 − structure` is **invariant to any local affine
change of gray level**. A more absorbent support, non-uniform illumination or a
different IR exposure shift a region's mean and scale without changing its
structure, so they produce no signal.

Verified against the project's own functions on synthetic data — a prediction that
is the same structure under an affine change (`pred = 0.7·real + 0.12`), versus one
with unrelated structure but matched statistics:

| | affine change, same structure | unrelated structure |
|---|---|---|
| `raw_delta` (mean) | **0.0429** — fires | 0.1706 |
| `luminance` (mean) | 0.9981 | 0.9965 |
| `contrast` (mean) | 0.9414 | 0.9838 |
| `structure` (mean) | 1.0000 | 0.0143 |
| `structural_delta` (mean) | **0.00000** — silent | 0.98570 |

`raw_delta` fires on the innocuous affine change; `structural_delta` is exactly
zero on it and near-saturated on the genuine structural difference. Note also that
`luminance` and `contrast` *do* register the affine change (0.998 and 0.941 against
0.997 and 0.984) — which is precisely why the design discards those two components
and keeps only `structure`.

This is why `structural_delta` and `raw_delta` fall into two sharply separated
detection tiers ([final-comments.md](final-comments.md) §3): not because the
structural signal is more sophisticated, but because it is blind to an entire
class of physical false positives.

The other side of that invariance: it is also why a purely z-scored loss term
cannot stand alone — if you are invariant to the absolute level you have no anchor
on the predicted value. That is why we decided never to raise the z-score weight
without leaving the MAE anchor in place.

## 6.6 Manifold hypothesis: the foundation of the heteroscedastic models

**L08 slides 10–11**: a manifold is "a connected region… from any given point the
manifold locally appears to be a Euclidean space"; the manifold hypothesis assumes
"most of `ℝⁿ` consists of invalid inputs, and that interesting inputs occur only
along a collection of manifolds containing a small subset of points".
**L08 slide 15**: "the encoder converts coordinates in the input space to
coordinates on the manifold. The decoder does the inverse mapping."

This is the strongest theoretical justification for the `*_nll` half of the
project.

**The setup.** Observable `(RGB, IR)` pairs lie on a manifold in the joint space.
The model learns `IR = f(RGB)`, assuming that manifold is the graph of a function
over RGB.

**The problem.** That assumption is false, on physical grounds: the RGB→IR map is
**one-to-many**. Pigments indistinguishable in visible light have markedly
different IR reflectance. In the slide's language, the projection of the joint
manifold onto RGB is **not injective** — the fibre above a given RGB value has
non-zero extent. A deterministic model is forced to average over it, and §5.7
sharpens this into the observation that there is no function for the universality
theorem to approximate at all.

**The solution.** The four `*_nll` models predict `(mu, log_var)`. `mu` is the
position on the manifold; `sigma = exp(0.5·log_var)` is **the local thickness of
the fibre**, learned end-to-end from RGB and context. So

```
z = (real_IR − mu) / sigma
```

is the distance from the manifold **measured in units of the manifold's own local
thickness** — a Mahalanobis-style distance rather than a Euclidean one.

**Why it matters.** Return to slide 7: the PCA reconstruction error is the squared
*ambient* distance between red and green points. That measure is correct only if
the manifold has uniform thickness. `raw_delta` is exactly that ambient Euclidean
distance, and it errs in two symmetric ways:

- large delta + large `sigma` (a colour the model has seen vary a lot)
  → small `z`, **not an anomaly**, just known ambiguity
- modest delta + small `sigma` (a historically stable colour)
  → large `z`, **a genuine candidate**

And the epistemologically decisive point: `sigma` is predicted **from RGB alone**,
never conditioned on `real_IR` at inference. It cannot "explain away" a real
anomaly by inflating itself where convenient, because it does not have the
information to do so.

**The degenerate case and its guard.** A large *constant* `sigma` would score as
perfectly calibrated (nominal coverage, `z_std ≈ 1`) while reducing `z` to a
rescaled `raw_delta` — useless. This is why
[calibration.py](scripts/calibration.py) reports `sharpness` and `dispersion`
**alongside** coverage and ENCE: `dispersion ≈ 0` invalidates a good calibration
score instead of confirming it.

## 6.7 Manifold and the split strategy

Manifold reasoning applied to the data. §1.1 argues the split invariants from
L03's i.i.d. assumptions, which is the rigorous form; L08's manifold vocabulary
adds the geometric intuition, briefly:

- Sections of one painting are *neighbouring points* on the manifold — same
  pigments, same support, same acquisition. Grouping them is what makes them one
  sample rather than 35.
- The mockup groups are **not** artworks to generalize to. They exist to
  **densify the manifold** at specific pigment coordinates — that is, to help
  *define* it. That is the geometric statement of §1.1's "deliberately not
  identically distributed", and L07 slide 4's "new real data (data-centric AI)"
  (§5.3) is the same idea from the regularization side.

See §1.1 and §8.11: an earlier data/config mismatch had made the mockup mechanism
inert; it has since been reconciled and now works as described.

## 6.8 Deep autoencoders: capacity is not the constraint

**L08 slides 16–21**: slide 16 defines a deep autoencoder as "an encoder composed
of convolutional layers, with a decoder composed of transposed convolutions or
other interpolating layers"; slides 17–21 show MNIST reconstruction results at a
range of code sizes. (Those five slides are images, so the exact code dimensions
are not quotable from the PDF text — the message, not the numbers, is what
transfers.) The message: reconstruction quality improves as the code grows, and a
convolutional autoencoder beats PCA at equal code size.

Relevance here: with a 640,000-value code (§6.3) we are far to the right of that
curve. **Reconstruction capacity is not the binding constraint.** The binding
constraint is data — 1167 patches from 29 artworks — and the risk is the opposite
of underfitting. That is why every choice in §6.3 pushes toward regularization and
transfer learning, and none toward more capacity.

## 6.9 Where we sit in the segmentation taxonomy — and why we trained no segmenter

**L08 slides 25–27**: classification ("no spatial extent") / semantic segmentation
("no objects, just pixels") / object detection / instance segmentation. Slide 27:
"**Paired training data: for each training image, each pixel is labeled** with a
semantic category. At test time, classify each pixel of a new image."

Our task is **dense prediction** like semantic segmentation — output the same size
as the input, one decision per pixel — but a **regression**, not a classification:

| | Semantic segmentation (L08) | deep_layers |
|---|---|---|
| Output | `C × H × W` scores → `argmax` (slide 34) | `1 × H × W` sigmoid, or 2 channels `(mu, log_var)` |
| Loss | per-pixel cross-entropy | MAE + (1−SSIM) / Laplacian+FFT / Gaussian NLL |
| Supervision | a hand-labeled per-pixel mask | the paired IR image |

**And here is the single most important design motivation in the whole mapping.**
We have **three** hand-annotated masks: `GT01_Map.png`, `GT02_Map.png`,
`GT03_Map.png`. With three masks, supervised semantic segmentation of underdrawings
is out of the question — slide 27 asks for every pixel labeled on every training
image, and we do not have that annotation budget, nor will we: annotating requires
a conservator, and a certainty historical paintings do not offer.

So the architectural choice is dictated by the annotation budget: **the supervised
task is replaced with a self-supervised proxy task.** Predicting IR from RGB
requires no annotation — the label *is* the IR image, and there are 1167 pairs.
The residual of that proxy task becomes the detection map (§8.2).

The circle closes in [detection.py](scripts/detection.py): AUROC and average
precision against a binary mask **are** segmentation metrics, grounded in L07
slides 48–52 (§5.8). So we evaluate a per-pixel binary segmentation, produced by a
regression head rather than a classification head, against the three masks
available — weakly-supervised segmentation, where supervision enters only at
evaluation and never during training.

Honest corollary, already documented: three masks sort the signals into *tiers*
(`structural` at 0.76–0.81 versus everything else at 0.53–0.59), not a fine-grained
ranking. Hence the second, independent axis —
[stroke_stats.py](scripts/stroke_stats.py), structure-tensor coherence, requiring
**no reference at all**: an underdrawing is made of oriented, elongated strokes,
prediction noise is isotropic. Two references failing in different ways, and their
rank correlation (Spearman 0.523) is the real result.

## 6.10 "Impossible to classify without context"

**L08 slide 29**: an isolated patch cannot be classified without context. *"Q: how
do we include context?"*

This justifies the entire depth of the architecture. An RGB pixel **does not
determine** its IR value: the same brown may be raw umber over gesso, or a thin
glaze over charcoal. Disambiguating requires brushstroke texture, stroke direction,
surrounding shapes — that is, **receptive field**, `140×140` at the bottleneck
(§0.6). Hence:

- Four encoder levels, built from stacked `3×3` convolutions (§3.2)
- The additive attention gates in
  [attention_unet.py](scripts/attention_unet.py) — CNN *spatial gating* (§4.9),
  **not** self-attention, so no global receptive field
- The `RestormerBlock` in [unet_restormer.py](scripts/unet_restormer.py): the
  **only** point in the project with a global receptive field (§6.14)

## 6.11 Sliding window vs fully convolutional

**L08 slides 31–32**: extract a patch, classify its centre pixel with a CNN.
*"Problem: Very inefficient! Not reusing shared features between overlapping
patches."*
**L08 slide 33**: encoding the whole image is intuitive, but classification
architectures reduce spatial size while segmentation needs output = input.
**L08 slides 34–35**: fully convolutional without downsampling fixes the size, but
"convolutions at original image resolution will be very expensive".
**L08 slides 36–37**: the synthesis — downsampling and upsampling **inside** the
network (Long/Shelhamer/Darrell FCN; Noh et al.).

The project sits entirely on slides 36–37, codified as invariant #2: inputs are
`(None, None, 3)`, spatial H/W is never hard-coded, `pad_to_multiple` pads to a
multiple of 16, and one `model.predict()` handles the whole image. L05 slide 77
states the same property from the ResNet side (§3.6).

Relevant history: a tiled-inference path with Gaussian blending (meaning #4 in
§0.2) **did exist** and was retired once every notebook had converged on
whole-image inference. Slide 32 is the argument in favour of that retirement. But
it must be stated precisely: **the reason for tiling was never the slide's
reason.** The slide is about reusing shared features between overlapping patches in
dense prediction; here it was the resolution mismatch between training (`400×400`)
and inference (up to `3674×2834`) plus memory. So the slide's argument does not
settle the question — which [code-review.md](code-review.md) §5.5 records as open,
and what §4.6
sharpens from the truncated-BPTT side.

## 6.12 Output size = input size, and why `_ResizeToMatch` exists

**L08 slide 33**: *"classification architectures often reduce feature spatial sizes
to go deeper, but semantic segmentation requires the output size to be the same as
input size."*

This slide describes a bug the project actually hit. EfficientNetB0 is a
**classification** network: its stride-2 depthwise convolutions round spatial
dimensions down. With a `400×400` input:

```
skip  block5c_project_bn  at H/16  →  400 // 16 = 25
bottleneck top_activation at H/32  →  400 // 32 = 12      (400/32 = 12.5, floored)
Conv2DTranspose(stride=2)          →   12 × 2  = 24
Concatenate([24, 25])              →   crash — off by 1
```

[_ResizeToMatch](scripts/efficientnet_unet.py) resolves that ±1-pixel discrepancy
at runtime with no static-shape assumptions. It is the concrete price of reusing a
classification backbone for dense prediction, and it would need re-verifying if the
backbone were swapped, since different backbones round differently or not at all.

The same pattern recurs three times: `_ResizeToMatch`,
[_Upsample2x](scripts/unet_v2.py) (needed because `layers.UpSampling2D` inspects
the *static* shape and fails on `(None, None, C)`), and
[ClipLogVar](scripts/nll_layers.py) — all three
`register_keras_serializable`-decorated. They are the cost of holding the
"fully convolutional at any resolution" invariant.

## 6.13 Upsampling: our choices against the lecture's menu

§0.8 defines the four mechanisms. Here is which we picked and why.

**Bed of nails / nearest neighbour** — no zero-insertion. `unet_v2` with
`use_upsample_conv=True` uses **bilinear** resize + `Conv2D`, the smooth version of
the slide's nearest neighbour.

**Transposed convolution** — the default: `Conv2DTranspose(f, 2, strides=2,
padding="same")`. A technical clarification is worth making, because the README
cites Odena et al. on checkerboard artifacts: **checkerboard artifacts arise when
the stride does not divide the kernel size**, producing uneven overlap on the
sparse grid slide 38 draws. Counting how many times each output position is
written:

| kernel | stride | `kernel % stride` | interior overlap counts | uniform? |
|---|---|---|---|---|
| **2** | **2** | **0** | `1 1 1 1 1 1 …` | **yes** |
| 3 | 2 | 1 | `2 1 2 1 2 1 …` | no |
| 4 | 2 | 0 | `2 2 2 2 2 2 …` | yes |
| 5 | 2 | 1 | `2 2 3 2 3 2 …` | no |

With kernel 2 and stride 2 the division is exact, the overlap uniform, and the
classic checkerboard already avoided by construction. So `use_upsample_conv` in
[unet_v2.py](scripts/unet_v2.py) is **not fixing a known artifact** — it tests
whether a smoother fixed interpolation beats a learned upsampling anyway. Which is
why it is an ablatable flag rather than a fused change: `build_unet_v2()` with no
arguments is architecturally identical to `build_unet()`, so the ablation is clean.

**Max unpooling — deliberately not used**, and the most consequential of the three
decisions. SegNet-style max unpooling propagates only the *positions* of the
maxima: cheap in memory, but it discards the values and everything non-maximal
(§0.8). UNet concatenates **the entire encoder tensor** instead.

The domain reason: the target signal — charcoal, chalk and lead-point strokes —
lives in the **high-frequency band**, and it is the only thing we care about. And
because the residual is the product (§6.2), a blurry prediction inflates
`raw_delta` along every edge and drowns the signal. We cannot afford to throw away
high-resolution information on the skip path to save memory. This is the same trade
§3.4 identifies: we give up translation invariance to keep position, and full
concatenation is what buys it.

## 6.14 The cost of full resolution, and the Restormer's placement

**L08 slide 35**: *"convolutions at original image resolution will be very
expensive."*

The `RestormerBlock` is inserted at the **bottleneck only**, with two layers of
motivation:

1. **Position** — at the bottleneck the resolution is lowest (`H/16`), so even
   ordinary self-attention would be as cheap there as it can be.
2. **Mechanism** — MDTA computes attention across **channels** rather than pixels,
   so its cost is **linear** in `H·W`, not quadratic. On a `3674×2834` image at
   inference this is not academic: quadratic self-attention would be flatly
   impossible.

Double safety, and consistent with the declared scoping decision: a narrow
attention block rather than a full transformer, because with 29 artworks a
data-hungry architecture is not defensible ([code-review.md](code-review.md) §6,
§6.3). It is also the project's only global receptive field (§6.10), and it costs
10.5M of the 41.6M parameters in `unet_restormer` (§3.8).

---

# Part 7 — Summary tables

One table per lecture, in lecture order. `§` references point into this file.

## L03 — Machine Learning Basics

| Concept (slide) | Where in the project | Why / what we chose |
|---|---|---|
| **i.i.d. assumptions (54)** | `grouped_train_val_test_split` | sections of one painting are **not independent** — the precise name for "leakage" (§1.1) |
| Identically distributed (54) | `mockup_aware_train_val_test_split` | mockups are *knowingly* off-distribution → keep in train, out of test (§1.1) |
| Capacity, over/underfitting (55–56) | 31M params, 820-patch training fold | firmly in the overfitting zone (§1.2) |
| "Memorizing the training set" (56) | — | a memorized painting yields a small residual *over the underdrawing too* (§1.2, §6.2) |
| Mini-batch 50–256 typical (44) | `BATCH_SIZE = 8` | one of two independent flags on this parameter (§1.3, §3.5, §8.4) |
| Loss = cost = objective (20) | used interchangeably | sanctioned; loss-vs-*metric* is the distinction that matters (§0.15) |
| k-fold cross-validation (57–59) | grouped **k = 3** (`kfold.py`) on top of the fixed split | fold-to-fold std ≈ 0.01 AUROC — the single split was adequate (§1.5, §8.7) |

## L04 — Feed-Forward Networks

| Concept (slide) | Where in the project | Why / what we chose |
|---|---|---|
| Input→output mapping is non-linear (9–13) | 18 ReLU layers | pigment IR reflectance is not affine in visible colour (§2.1) |
| Activation criteria (14) | one activation everywhere | "almost always the same activation in all hidden layers" — satisfied (§2.2) |
| **CNN ⇒ ReLU, RNN ⇒ tanh/sigmoid (18)** | ReLU throughout; `tanh` only in `unet_residual`'s head | the course's own one-line answer to §4.5's argument (§2.2) |
| Vanishing gradient along depth (53) | ~18 GroupNorms + ResUNet's shortcut | same compounding as L06, but 23 *different* matrices (§2.3, §4.7) |
| The four enablers of deep training (54) | 3 of 4 met | data ✗, GPU ✓, init ✓ (He), dropout ✗ (weight decay instead), ReLU ✓ (§2.3) |

## L05 — Convolutional Neural Networks

| Concept (slide) | Where in the project | Why / what we chose |
|---|---|---|
| Local connectivity, filter size (5) | every `Conv2D`; the Gaussian window in `delta_analysis` | learned everywhere except the measurement window (§0.1) |
| Convolution replaces matrix multiply (6, 9) | `_conv_block` in every architecture | weight sharing → any input size (§0.3) |
| Parameter sharing (13, 53) | all 23 conv layers | translation equivariance makes arbitrary-size inference possible (§0.3, §6.11) |
| Padding `(K−1)/2` (19, 28) | `padding="same"`; `mode="reflect"` for measurement | zero-padding a `[0,1]` image would invent a false delta ring (§0.4) |
| Dilation (19) | `unet_dilated` / `unet_v2_dilated`, ASPP bottleneck | widens context without downsampling — tried, no benefit, unstable (§0.5, §8.8) |
| Stacked 3×3 beats one 7×7 (64) | two `3×3` per block → `140×140` at the bottleneck | fewer parameters, more non-linearities (§3.2, §0.6) |
| Pooling, invariance to translation (34–36) | `MaxPool2D(2)`; strided conv in `unet_v2` | invariance is **half a liability** here; skips hand position back (§3.4) |
| ReLU preferred (31) | after every GroupNorm, via `_ReLUFix` | (§0.14, code-review §3.3) |
| Fully connected layer (39) | **none** | it fixes input resolution and would break invariant #2 (§0.18, §3.6) |
| Softmax (40) | **unused** | output is a continuous reflectance, not a class (§0.15) |
| Normalization before nonlinearity (48) | `Conv2D → GroupNorm → ReLU`, ~18 times | slide's position kept; normalizer swapped for batch-independence (§3.5) |
| "Small mini-batch may affect BatchNorm" (48) | `BATCH_SIZE = 8` | **resolved** — switched to GroupNorm, which is batch-independent (§3.5, §8.4) |
| Dropout 0.5 (54) → no dropout (78) | `dropout_rate=0.0` by default | we sit with ResNet; GroupNorm + weight decay take the role (§0.13, §5.4) |
| LR ÷10 on plateau (54, 78) | `ReduceLROnPlateau` (factor 0.25, patience 6) | same idea, automated (§0.16) |
| VGG: only 3×3 s1 p1 + 2×2 maxpool s2 (63) | `_conv_block` + `MaxPool2D(2)` | our default architecture *is* VGG at block level (§3.1) |
| Double filters, downsample by 2 (77) | `filters=[64,128,256,512]` | capacity conserved as resolution falls (§3.3) |
| Degradation is not overfitting (75) | — | our depth problem is the *opposite* kind; don't confuse the remedies (§3.6) |
| Residual mapping (76–77) | `_residual_block` in `resunet.py` | but the shortcut is a projection, not identity (§4.8, §8.9) |
| "Variable input sizes" in parentheses (77) | invariant #2, `pad_to_multiple` | a curiosity in the lecture; load-bearing here (§3.6) |
| Transfer learning with few labels (81–83) | frozen EfficientNetB0 encoder | the lecture asks our exact question (§3.7) |
| AlexNet 61M, mostly in FC layers (52, 70) | ~31M, **all convolutional** | a larger conv model than AlexNet, on 1167 images (§3.8) |

## L06 — Recurrent Neural Networks

| Concept (slide) | Where in the project | Why / what we chose |
|---|---|---|
| DAG, topological ordering (2) | all ten architectures | slide 2's own diagram *is* `_residual_block` (§0.16, §4.2) |
| Cycles, a notion of time (3) | **nothing** | a painting is not a sequence; no axis to carry state along (§4.1) |
| Unrolled size depends on input length (4) | — | our graph depth is **constant** in image size; only shapes grow (§4.3) |
| "Same parameters at this level" (6, 29) | convolution shares over space, not time | same idea, different index set (§4.3) |
| Architecture taxonomy (21–25) | spatial analogue of slide 25 (aligned) | **not** slide 24's encoder–decoder: alignment is given, so skips beat attention (§4.4) |
| Output projection `y = W·h + b` (30) | `Conv2D(1, 1, sigmoid)` | same map, per position instead of per step (§4.5) |
| `tanh` (30, 66, 74) | only in `unet_residual`'s output head | saturation only pays when an activation feeds itself; the residual head is a bounded-output exception (§0.14, §4.5) |
| Truncated BPTT (44) | tiled inference, retired | whole-image = full BPTT = no boundary bias (§4.6) |
| Gradient clipping (79) | `Adam(clipvalue=1.0)` | not for recurrence — for the MS-SSIM gradient singularity (code-review §3.5); other unrelated "clips" exist (§4.7) |
| Vanishing gradients (53–63, 79) | 23 *different* matrices + ~18 GroupNorms | not the same structure as `W^(T−1)` (§4.7) |
| Additive vs multiplicative path (77) | `RestormerBlock` has the pure form | ResUNet's projection shortcut interrupts it (§4.8) |
| **"ResNet is to PlainNet what LSTM is to RNN" (78)** | `resunet.py` | the lecture supplies our justification, stronger than the README's (§4.8) |
| LSTM gates (66–74) | `_attention_gate`; GDFN | a forget gate over pixels instead of timesteps (§0.17, §4.9) |
| Autoregressive sampling (42) | **rejected** | a fabricated mark is worse than a missed one (§4.10) |

## L07 — Regularization in Deep Learning

| Concept (slide) | Where in the project | Why / what we chose |
|---|---|---|
| Definition of regularization (22) | — | the grouped split is **not** regularization: it fixes evaluation, not the algorithm (§5.1) |
| The canonical four (23) | early stopping ✓, augmentation ✓, weight decay ✓, dropout ✗ | plus ~18 GroupNorms acting as regularizer (§5.1) |
| **Weight decay / norm penalties (24–28)** | `Adam(weight_decay=1e-5)` | value taken from L05's ResNet recipe (§5.2, §8.5) |
| Augmentation attacks the root cause (5) | [augmentation.py](scripts/augmentation.py) | more information from the same artworks (§5.3) |
| **"Label is preserved" (6)** | geometric → both channels; colour → RGB only | the *principle* behind invariant #5 (§5.3, §8.3) |
| Adversarial / style-transfer augmentation (6–9) | rejected / untried | fabrication is worse than omission; style transfer is the Cann et al. precedent (§5.3) |
| "New real data, data-centric AI" (4) | the mockup groups | data-centric before regularization (§5.3, §6.7) |
| Dropout = ensemble, train-time only (30–33) | `dropout_rate=0.0`, `unet_v2` only | passed over for weight decay; GroupNorm carries no train-time running stats (§5.4) |
| **Xavier assumes tanh; ReLU wants He (35, 37)** | `he_normal` on every ReLU-preceding conv | the pairing slide 37 prescribes (§5.5, §8.6) |
| Early stopping (40–41) | `ModelCheckpoint` + `EarlyStopping(restore_best_weights)` | the slide is a specification of our exact callback pair (§5.6) |
| **Universality theorem (42–47)** | the `*_nll` models | RGB→IR **is not a function** — the deterministic ceiling is a category error (§5.7) |
| Continuous functions only (47) | `structural_delta` | pentimento edges are near-discontinuities → weakest prediction at contours (§5.7, §6.5) |
| "99% correct but useless" (48–49) | accuracy used **nowhere** | hidden-detail pixels are a tiny fraction of a painting (§5.8) |
| Sensitivity / FPR / precision (50–51) | AUROC in [detection.py](scripts/detection.py) | rates normalized by their own class total → prevalence-independent (§5.8) |
| F-measure ignores true negatives (52) | AP reported **with prevalence** | AP's chance level *is* the prevalence (§5.8) |

## L08 — Autoencoders + Semantic Segmentation

| Concept (slide) | Where in the project | Why / what we chose |
|---|---|---|
| Encoder `f` / decoder `g` (2) | all ten architectures | topology adopted, but cross-modal — not an autoencoder (§6.1) |
| "unable to copy perfectly" (2) | the delta *is* the product | localized failure is the result, not a defect (§6.2) |
| Minibatch GD + backprop (3) | Adam, `BATCH_SIZE=8`, `EPOCHS=100` | (§0.16) |
| Undercomplete: `dim(h) < dim(x)` (5) | bottleneck 640k > 480k input | **overcomplete** in fact: we regularize, we do not compress (§6.3) |
| Squared reconstruction error = PCA (6–8) | MSE used nowhere | blur → inflated delta at every edge → unusable residual (§6.4) |
| PCA standardization (6–8) | `structure` = local Pearson correlation | local affine invariance = immunity to substrate shifts (§6.5) |
| Too much capacity → copying (9) | ~18 GroupNorm, weight decay, EarlyStopping, frozen encoder | non-architectural regularization only (§6.3) |
| Manifold hypothesis (10–11, 15) | the `*_nll` models, `sigma`, `z` | RGB→IR is one-to-many: `sigma` is the fibre thickness (§6.6, §6.7) |
| Deep conv autoencoder (16–21) | UNet + `Conv2DTranspose` | capacity is not the constraint; data is (§6.8) |
| Semseg needs every pixel labeled (27) | only 3 masks exist | → self-supervised RGB→IR proxy task (§6.9) |
| "Impossible to classify without context" (29) | 4 levels, gates, Restormer | pigment is ambiguous for an isolated pixel (§6.10) |
| Sliding-window inefficiency (32) | tiled inference retired; whole-image used | valid argument, not the original motivation; see code-review §5.5 (§6.11) |
| `argmax` over `C` scores (34) | sigmoid over 1 channel | dense **regression**, not dense classification (§0.15) |
| Full-res convs too expensive (35) | Restormer at the bottleneck, linear MDTA | inference runs up to `3674×2834` (§6.14) |
| Down+upsample inside the network (36–37) | the whole UNet family | (§6.11) |
| Unpooling / bed of nails / max unpooling (38–39) | max unpooling **rejected** | discards high frequency, which is the signal (§0.8, §6.13) |
| Deconvolution (40–41) | `Conv2DTranspose(2, s=2)` | kernel/stride divisible → no checkerboard by construction (§6.13) |
| output size = input size (33) | `_ResizeToMatch`, `pad_to_multiple` | the price of a classification backbone (§6.12) |

---

# Part 8 — Findings: how the design questions were resolved

The concept-by-concept mapping above raised a set of questions about the training
recipe and the evaluation. This part records how each was settled. §8.1–8.3 were
documentation gaps, now closed by [README.md](README.md),
[final-comments.md](final-comments.md) and this file. §8.4–8.11 were technical and
each was acted on.

| # | Question raised | Outcome |
|---|---|---|
| 8.1 | The universality + manifold argument for `sigma` | written up (§5.7, §6.6, README) |
| 8.2 | The annotation budget as the *cause* of the architecture | written up (§6.9, README) |
| 8.3 | Label preservation as the *principle* behind the augmentation asymmetry | written up (§5.3, README) |
| 8.4 | BatchNorm at `BATCH_SIZE = 8` | switched to `GroupNormalization` |
| 8.5 | No weight decay | added, `1e-5` in the optimiser |
| 8.6 | Xavier init with ReLU | switched to `he_normal` |
| 8.7 | No cross-validation | grouped k-fold (k = 3) run; single split confirmed adequate |
| 8.8 | Dilated convolution untried | tried (ASPP bottleneck); no benefit, training instability |
| 8.9 | ResUNet's unconditional projection shortcut | conditional identity path would be dead code — kept as is |
| 8.10 | Two-phase fine-tuning unused | implemented and run; underperforms the frozen encoder |
| 8.11 | Dataset integrity / inert mockup split | data and configuration reconciled |

## 8.1 The universality and manifold argument for `sigma`

The physical framing ("pigments alike in visible light can have markedly different
IR reflectance") is correct but local. Two stronger framings are now recorded: the
**logical** one (§5.7 — RGB→IR is not a function, so the universality theorem has
nothing to approximate, and the deterministic ceiling is a category error rather
than a shortfall) and the **geometric** one (§6.6 — a non-injective projection of
the joint manifold, `z` as a distance in units of local fibre thickness). The
measured payoff, separately, is small — see §8.10 and
[final-comments.md](final-comments.md) §6.

## 8.2 The annotation budget as the cause of the architecture

Choosing a self-supervised RGB→IR proxy task because three masks cannot support
supervised segmentation (§6.9, against L08 slide 27) is the most defensible design
decision in the project. It is now stated as such in [README.md](README.md) and in
§6.9.

## 8.3 Label preservation as the principle behind the augmentation asymmetry

Spatial transforms apply to RGB *and* IR with a shared box; photometric jitter
applies to RGB only. The principle behind the rule is L07 slide 6's "data warping
augmentations transform existing images such that **their label is preserved**"
(§5.3) — a geometric transform moves an image-valued label and so must be applied
to it; a photometric transform on the input must not touch it.

## 8.4 GroupNorm instead of BatchNorm

L05 slide 48 ("small size of the mini batch may affect the BatchNorm") and L03
slide 44 (50–256 typical) both flag `BATCH_SIZE = 8`. The from-scratch
architectures were switched from `BatchNormalization` to `GroupNormalization`
(Wu & He, 2018), which normalizes over channel groups within each sample and is
batch-size-independent. The pretrained EfficientNetB0 encoder keeps its BatchNorm,
since the running statistics are part of the transferred ImageNet knowledge. A
pilot on `unet` improved every reconstruction metric under GroupNorm with no
instability. See §3.5.

## 8.5 Weight decay added

L07 slide 23 lists parameter norm penalties first; both L05 recipes use one.
`compile_model` now passes `Adam(weight_decay=1e-5)` — decoupled weight decay on
every parameter, the value taken directly from L05's ResNet recipe (slide 78). See
§5.2.

## 8.6 He initialisation

L07 slide 37 states Xavier "does not work well with the ReLU activation function"
and recommends He. Every ReLU-preceding convolution now sets
`kernel_initializer="he_normal"`. See §5.5.

## 8.7 Grouped k-fold cross-validation

L03 slide 57 prescribes k-fold for exactly the instability a sensitive model
shows. `scripts/kfold.py` implements a grouped k-fold (k = 3): each fold holds out
whole artworks, all mockups stay in training, the three ground-truth paintings are
external to every fold. Run on the two leading heteroscedastic models, it gave a
fold-to-fold standard deviation of ≈ 0.01 AUROC — far below any ranking gap. The
single fixed split was adequate and cross-validation confirms its point estimates.
See §1.5 and [final-comments.md](final-comments.md) §5.

## 8.8 Dilated convolution — tried, negative

L05 slide 19 introduces dilation as the standard way to widen the receptive field
without downsampling — that is, without the position loss §3.4 identifies as this
project's core tension. `scripts/aspp.py` adds a parallel-dilation (ASPP-style)
bottleneck, used by `unet_dilated` and `unet_v2_dilated`. Result: no fidelity or
detection benefit over the plain bottleneck, and genuine optimisation instability
— the dilated variants peaked early and then degraded. The wide concat-then-project
block appears to make the loss landscape harder. Applying dilation only at the
bottleneck (rather than trading a pooling stage for it) is the conservative form;
the more invasive version would need the skip/decoder scheme redesigned.

## 8.9 ResUNet's projection shortcut — dead-code check

L06 slides 77–78 identify the *uninterrupted additive path* as the mechanism by
which residual connections help; `_residual_block` in
[resunet.py](scripts/resunet.py) routes its shortcut through a learned `1×1`
convolution and a normalization on every block, unconditionally. Making that
conditional on matching channel counts was checked and rejected: every block in
this architecture always changes channel count (one block per depth level,
widening or narrowing every time), so a conditional identity path would never
fire. The unconditional projection stays. A true identity path would require a
structural change (a second, same-width block per level) outside the scope of the
question. See §4.8.

## 8.10 Two-phase EfficientNet fine-tuning — implemented, negative

L05 slides 81–83 motivate transfer learning for exactly this low-label regime, and
the standard recipe is to train the decoder frozen and then unfreeze and fine-tune
end-to-end at a lower learning rate. `efficientnet_unet_ft` /
`efficientnet_unet_nll_ft` implement this (`freeze_encoder=False`, warm-started
from the frozen checkpoint, `FINETUNE_LEARNING_RATE = 1e-5`). Result: the
fine-tuned checkpoints consistently *underperform* the frozen baseline on every
metric — the paired dataset is too small to fine-tune an ImageNet backbone without
eroding the features that made it useful. See [final-comments.md](final-comments.md)
§8.

## 8.11 Dataset integrity

An earlier data state had two problems this mapping surfaced: two pairs with
mismatched RGB/IR dimensions (which made `load_image_pairs` raise), and a
`MOCKUP_ARTWORK_IDS` list that matched nothing in the filenames (which made
`mockup_aware_train_val_test_split` silently equivalent to the plain grouped
split). Both have been reconciled — the data now loads, and all six configured
mockup groups (`tblu`, `tbianco`, `tbruno`, `tgiallo`, `trosso`, `tverde`) are
present, so the mockup-aware split does what §1.1 and §6.7 describe. Counts:
`data/rgb` and `data/ir` hold 1168 files each with matching stems, `extract_artwork_id`
yields 29 distinct artworks.
