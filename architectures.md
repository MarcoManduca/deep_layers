# The architectures, measured against the baseline

What each variant changes with respect to `unet`, why it was tried, and what
came out of it. Every claim is traced back to the code.

The overview of the models is in [README.md](README.md); the full quantitative
results in [final-comments.md](final-comments.md).

---

## 1. The reference: `unet`

Everything else is measured from here.

```
input (None, None, 3)
  │
  ├─ conv_block(64)   ──skip──────────────────────────┐
  │  MaxPool2D(2)                                     │
  ├─ conv_block(128)  ──skip───────────────────┐      │
  │  MaxPool2D(2)                              │      │
  ├─ conv_block(256)  ──skip────────────┐      │      │
  │  MaxPool2D(2)                       │      │      │
  ├─ conv_block(512)  ──skip─────┐      │      │      │
  │  MaxPool2D(2)                │      │      │      │
  └─ conv_block(1024)            │      │      │      │   ← bottleneck
        Conv2DTranspose(512) ──► concat ┘      │      │
        conv_block(512)                        │      │
        Conv2DTranspose(256) ──► concat ───────┘      │
        conv_block(256)                               │
        Conv2DTranspose(128) ──► concat ──────────────┘
        conv_block(128)
        Conv2DTranspose(64)  ──► concat
        conv_block(64)
        Conv2D(1, 1, activation="sigmoid")
```

The block, identical at every stage ([unet.py:34-44](scripts/unet.py#L34-L44)):

```
Conv 3×3 → GroupNorm(32) → ReLU → Conv 3×3 → GroupNorm(32) → ReLU
```

Two ReLUs, not one: without the intermediate non-linearity the two convolutions
would collapse into a single linear operator.

Convolutions use `padding="same"`, `use_bias=False`, `he_normal`. The model is
fully convolutional: 23 convolutions along the input→output path, and inputs of
arbitrary size as long as they are multiples of 16.

**Departures from the original U-Net** (Ronneberger, Fischer, Brox, MICCAI 2015),
all dictated by the task:

| original | here | why |
|---|---|---|
| *unpadded* convolutions, skips must be cropped | `padding="same"` | the residual is computed pixel by pixel: input and output must line up |
| no normalization | GroupNorm | batch 8 |
| 2 channels + softmax (segmentation) | 1 channel + sigmoid (regression) | we predict a value, not a class |

The initialization, by contrast, is unchanged: Ronneberger already uses a
Gaussian with standard deviation `√(2/N)` — He initialization before it had that
name.

---

## 2. The shared recipe

Fifteen models, **one single recipe held constant**. That is what makes the
comparison an ablation rather than fifteen disconnected attempts.

| element | value | note |
|---|---|---|
| initialization | `he_normal` on every conv preceding a ReLU | `σ² = 2/N` |
| hidden activation | ReLU | via `_ReLUFix`, a workaround for a `tensorflow-metal` bug |
| output activation | sigmoid | the codomain decides the activation |
| normalization | `GroupNormalization(32)` | batch-size independent |
| optimizer | Adam, lr `1e-4`, `weight_decay=1e-5`, `clipvalue=1.0` | the per-element clip guards against the singularity of the MS-SSIM gradient |
| batch size | **8** | this is the reason for GroupNorm |
| deterministic loss | `0.16·Charbonnier + 0.84·(1−MS-SSIM)` | §10 |
| heteroscedastic loss | `β`-weighted Laplace NLL, `β = 0.5` | §11 |
| callbacks | `ModelCheckpoint`, `EarlyStopping` (patience 20), `ReduceLROnPlateau` | all on `val_loss`, capped at 100 epochs |

### The two exceptions, both justified

1. **The EfficientNet encoder keeps its own `BatchNormalization`** — the ImageNet
   statistics are part of the transferred knowledge. Only the decoder uses
   GroupNorm.
2. **`unet_residual` has a `tanh` head + clip**, not sigmoid — the residual must
   be allowed to go negative.

Marginally: the Restormer block internally uses `LayerNormalization` and `GELU`,
which come along with the borrowed component.

> **A historical note worth knowing about.** Earlier rounds routed the loss
> through a per-architecture `ARCH_LOSSES` registry, which gave EfficientNet
> `combined_loss_advanced` (MAE + Laplacian + FFT). **That registry is gone.**
> [`trainer.compile_model`](scripts/trainer.py#L121) now uses `combined_loss`
> for *every* deterministic architecture, EfficientNet included — which is
> precisely what makes the comparison an ablation. `combined_loss_advanced` is
> still implemented and tested, simply not wired to any training path.

---

## 3. Summary table

| axis | model | what changes | hypothesis tested | outcome |
|---|---|---|---|---|
| baseline | `unet` | — | — | detection 0.71, the best deterministic model |
| conv block | `resunet` | residual block | a linear path for the gradient helps | **best fidelity of the set**, MAE ≈ 0.11 |
| conv block | `attention_unet` | gate on the skips | filtering the skips helps | detection 0.70, on par with the baseline |
| up/down sampling | `unet_v2` | learned stride-2, resize+conv, dropout | learned downsampling, no checkerboard | within the noise band |
| bottleneck | `unet_dilated` | ASPP bottleneck | receptive field without pooling | no benefit + instability |
| bottleneck | `unet_v2_dilated` | `unet_v2` + ASPP | do the effects add up? | as above |
| bottleneck | `unet_restormer` | transposed attention | is global context needed? | within the band |
| output head | `unet_residual` | residual over `mean(R,G,B)` | learning the difference is easier | **an instructive failure**: 0.67 = the floor |
| pretrained | `efficientnet_unet` | frozen ImageNet encoder | transferable features | PSNR 16.2–16.5, in the pack |
| pretrained | `efficientnet_unet_ft` | unfrozen encoder, lr 1e-5 | does fine-tuning help? | **worse on every metric** |

---

## 4. `resunet` — residual block

### What changes

Only `_conv_block` → `_residual_block`. Nine blocks replaced, topology
unchanged.

```
                    ┌──── Conv1×1(f) → GN ────────────────┐   shortcut
                    │                                     │
   x ───────────────┤                                     ▼
                    └─ Conv3×3(f) → GN → ReLU →          Add
                       Conv3×3(f) → GN ──────────────────►│
                                                          ▼
                                                        ReLU
```

Main branch with no final ReLU, shortcut with no activation, `Add`, and **only
then** the ReLU.

### Why

Instead of making the block learn a mapping `H(x)`, we make it learn the
**residual** `F(x) = H(x) − x`. If the optimal transformation is close to the
identity, `F` only has to produce a small correction — and driving `F` to zero
is trivial, whereas making two convolutions reproduce the identity exactly is
surprisingly hard.

The mechanical reason that matters most: the derivative of a sum with respect to
each addend is 1, so the gradient propagates along the shortcut **without
attenuation**, in parallel with the convolutional branch.

### Why the ReLU sits after the `Add`

This is ResNet's *post-activation* convention. Putting it at the end of the main
branch would interrupt the identity path with a non-linearity at every block —
and the clean route back to earlier layers, which is the whole point, would be
lost.

```
unet     :  Conv → GN → ReLU → Conv → GN → ReLU
resunet  :  Conv → GN → ReLU → Conv → GN → Add(shortcut) → ReLU
```

### The caveat: the shortcut is never the identity

In canonical ResNet the pure identity is used when the channel counts match, and
a 1×1 projection only when they differ. Here the projection is **always**
applied, because the number of channels always changes:

| where | in | out |
|---|---|---|
| encoder | 3, 64, 128, 256 | 64, 128, 256, 512 |
| bottleneck | 512 | 1024 |
| decoder | `2f` (upsampled ⧺ skip) | `f` |

An input that already has `filters` channels **never occurs**, so the projection
is never wasted. But the consequence has to be understood: **nowhere in this
network is there a pure identity shortcut** — the "derivative exactly 1"
property is weakened by a `Conv1×1` + GN. It remains, nonetheless, a much
shorter path than the main branch.

The docstring also states the real fix: a **second constant-width block per
level**, as in ResNet *stages*. It was not implemented — it is a structural
change, out of scope.

Cost of the shortcut: a 1×1 costs roughly one ninth of a 3×3 at equal channel
count, so ~5% of the block's parameters.

### Outcome

- **Best pixel fidelity of the set**: MAE ≈ 0.11, PSNR ≈ 18.6 dB.
  `resunet_nll` goes down to MAE ≈ 0.09.
- Deterministic detection **0.68**, below `unet` (0.71) — the two metric families
  disagree, and that is the point of the results slide.
- **With `α = 0.50` it is the only genuine gain of the loss sweep**: fidelity
  (MAE 0.114 → 0.110) and detection (0.681 → 0.706) improve together.
- With `α = 0.84` it **collapses** (SSIM 0.09): a stability risk, not just a
  degradation.
- **`resunet_nll` is the recommended model for detection**: `structural_z`
  0.716 ± 0.011 per fold, **0.728** with the 3-fold ensemble, and clearly better
  than `attention_unet_nll` on the difficult painting GT03 (0.690 vs 0.653).

---

## 5. `attention_unet` — gate on the skips

### What changes

A single point, inside the decoder
([attention_unet.py:116-119](scripts/attention_unet.py#L116-L119)):

```
unet:            Conv2DTranspose ──┐
                                   ├─► Concatenate ─► conv_block
                 skip ─────────────┘

attention_unet:  Conv2DTranspose ──┬──────────────────┐
                                   │                  ├─► Concatenate ─► conv_block
                                   ▼                  │
                 skip ────────► attention_gate ───────┘
```

Upsampling first, **then** the gate: the decoder signal must already be at the
skip's resolution.

### The gate, in one sentence

It builds a **mask** — a single-channel image with values between 0 and 1, the
same size as the feature map — and **multiplies the skip by that mask**. Where
it is 1 the skip passes through untouched; where it is 0 it is switched off.

```python
theta_x = Conv2D(f//2, 1)(skip)         # compress the skip
phi_g   = Conv2D(f//2, 1)(g)            # compress the decoder signal
add     = ReLU(theta_x + phi_g)         # sum them: ADDITIVE attention
psi     = sigmoid(Conv2D(1, 1)(add))    # → a single map, values in [0,1]
return  skip * psi                      # apply the mask
```

With concrete shapes, at the first decoder stage on a 400×400 input:

```
skip : 50 × 50 × 512      g : 50 × 50 × 512
  ↓ Conv1×1                 ↓ Conv1×1
     50 × 50 × 256             50 × 50 × 256
        └────────── + ─────────┘
                   ReLU
                    ↓ Conv1×1
              50 × 50 × 1
                   sigmoid
                    ↓
        skip × mask  →  50 × 50 × 512
```

### The three properties that matter

1. **Additive attention**, not dot-product: compatibility is computed by summing
   the projections, not multiplying them.
2. **Single-channel output**: the multiplication broadcasts, so all 512 channels
   at a position receive **the same weight**. This is **spatial** attention — it
   says *where*, not *which feature*.
3. **Sigmoid, not softmax**: every position gets a value independent of the
   others. Technically this is a **gate**, not attention in the transformer
   sense.

### The point almost nobody notices

**All three convolutions are 1×1**, so the weight of pixel `(i,j)` depends only
on `skip(i,j)` and `g(i,j)`: **zero spatial context inside the gate**.

Context arrives by another route: `g` is a decoder feature map, so it has
already passed through the bottleneck and carries a wide receptive field with
it. The semantics are:

> The decoder, which has seen the broad context, decides pixel by pixel how much
> of the encoder's fine local information to let through.

**Corollary worth having ready for the exam:** the gate does **not** give the
network a long-range receptive field. The only model with true attention and a
global receptive field is `unet_restormer`.

### Cost and notes

Three 1×1 convs per gate, four gates: about **350,000 parameters** against the
model's ~31 M. The cheapest modification of the set.

The gate's convolutions are **bare**: no GroupNorm, and no explicit initializer,
hence Glorot. A small inconsistency with the project's rule — `theta_x` and
`phi_g` precede a ReLU and "would like" He.

### Outcome

- Detection **0.70**, essentially on par with `unet`: the gate gains nothing
  measurable.
- `attention_unet_nll` is one of the two recommended models: 0.699 ± 0.008 per
  fold, **0.719** with the ensemble — but `resunet_nll` is at least its equal.
- **Fragile with respect to the loss**: raising `α` drops it **below the
  `mean(R,G,B)` floor**. Its strength depends on an MS-SSIM-dominated loss. In
  the run with the *generic* split it diverged (`val_loss` 0.46 against ~0.22):
  that checkpoint is a failure, not a result.

---

## 6. `unet_v2` — sampling and regularization

### What changes

Three modifications, **independently switchable**, all `False`/`0.0` by default
— so `build_unet_v2()` with no arguments is **identical** to the baseline.

> **The trained checkpoint has all three toggles on**:
> `use_strided_conv=True, use_upsample_conv=True, dropout_rate=0.2`
> ([023_training_variants.ipynb](notebooks/023_training_variants.ipynb), cell 6).

### Toggle 1 — `use_strided_conv`: how we go down

```python
# False (baseline)
MaxPool2D(2)

# True (used)
Conv2D(f, 3, strides=2, padding="same", use_bias=False, he_normal)
GroupNormalization(32)
ReLU
```

The **stride** is the step by which the kernel moves: with `stride=2` it skips
every other position, so the output has half the side length.

```
stride 1 →  positions  0  1  2  3  4  5  6  7     = 8 outputs
stride 2 →  positions  0     2     4     6        = 4 outputs
```

| | `MaxPool2D(2)` | `Conv2D(f, 3, strides=2)` |
|---|---|---|
| parameters | **0** | f × f_in × 9 |
| operation | the **maximum** of the 2×2 block | a **learned weighted** sum over a 3×3 window |
| windows | partition, no overlap | kernel 3 > stride 2 → **they overlap** |
| non-linearity | none | GroupNorm + ReLU |

**MaxPool selects, the conv combines.** MaxPool discards 3 values out of 4 by a
fixed criterion; the conv fuses all of them with learned weights. The price:
extra parameters, and the loss of the small translation invariance that pooling
gives for free.

### Toggle 2 — `use_upsample_conv`: how we go back up

```python
# False (baseline)
Conv2DTranspose(f, 2, strides=2, padding="same")     # neither norm nor activation

# True (used)
_Upsample2x()                                        # bilinear ×2, 0 parameters
Conv2D(f, 3, padding="same", use_bias=False, he_normal)
GroupNormalization(32)
ReLU
```

A **transposed convolution** takes each input pixel, multiplies it by the whole
kernel, and **stamps** the result onto a window of the output:

```
input:   [a]  [b]                    kernel 2×2, stride 2

output:  [a·k00  a·k01][b·k00  b·k01]
         [a·k10  a·k11][b·k10  b·k11]
```

The **checkerboard** appears when the kernel is not divisible by the stride (the
classic case: kernel 3, stride 2): the stamps overlap unevenly and a
high-frequency lattice comes out (Odena, Dumoulin & Olah, *Distill* 2016).

With `kernel=2, stride=2` the stamps tile exactly, so *that* artefact does not
arise. A related defect remains, though: every 2×2 block comes from **a single**
input pixel, so adjacent blocks have no continuity constraint between them —
block edges can appear.

Resize+conv separates the responsibilities: **bilinear interpolation** (smooth
by construction) to enlarge, **3×3 conv** (which crosses block boundaries) to
transform.

**Why this matters more here than elsewhere:** any periodic lattice in the
prediction goes straight into the residual, where it *looks like structure* —
and `structural_delta` measures exactly local structural disagreement. It would
be a **false-positive generator** on the metric that counts.

*Detail:* `_Upsample2x` is a custom layer because `UpSampling2D` reads the
**static** shape and fails on `(None, None, C)` inputs. It reads the shape at
runtime with `tf.shape` instead. Same pattern as `_ResizeToMatch` in
`efficientnet_unet.py`.

### Toggle 3 — `dropout_rate = 0.2`

`SpatialDropout2D(0.2)` **after** the conv block, and only in two places: the
**bottleneck** and the **first decoder block**.

Dropout zeroes a random fraction of the activations **during training only**,
rescaling the survivors by `1/(1−p)`. At inference it zeroes nothing.

**`SpatialDropout2D` zeroes whole channels**, not individual pixels — and that
is the right granularity: in a feature map neighbouring pixels are strongly
correlated, so zeroing one changes almost nothing. Zeroing a channel removes a
complete **feature detector**.

| where | channels | zeroed per step | rescaling |
|---|---|---|---|
| bottleneck | 1024 | ~205 | ×1.25 |
| first decoder block | 512 | ~102 | ×1.25 |

Confined there because the bottleneck is where the overfitting risk is
concentrated, whereas the shallow levels hold the relevant fine detail. There is
also a technical reason: dropout and normalization have a known *variance shift*
conflict (Li et al., CVPR 2019, ref [16] in the README).

### Outcome

Within the noise band, not singled out in the write-up.

**Limitation to declare if asked**: the docstring says the three parameters are
exposed separately so they can be ablated in isolation, but **only the
configuration with all three on was ever trained**. If `unet_v2` had shown an
effect, we would not know which of the three produced it. It did not show one,
so the point has no bite.

---

## 7. `unet_dilated` and `unet_v2_dilated` — ASPP bottleneck

### What changes

**One line**, the bottleneck block
([unet_dilated.py:53](scripts/unet_dilated.py#L53)):

```python
# unet
x = _conv_block(x, 1024)

# unet_dilated
x = dilated_bottleneck(x, 1024, dilation_rates=(1, 2, 4, 8))
```

### The dilated convolution

A normal 3×3 kernel covers three adjacent pixels per side. A **dilated**
convolution takes the same nine weights and **spaces them out**:

```
dilation 1           dilation 2             dilation 4

  X X X              X . X . X              X . . . X . . . X
  X X X              . . . . .              . . . . . . . . .
  X X X              X . X . X              . . . . . . . . .
                     . . . . .              . . . . . . . . .
                     X . X . X              X . . . X . . . X
                                            . . . . . . . . .
                                            . . . . . . . . .
                                            . . . . . . . . .
                                            X . . . X . . . X
```

The dots are pixels the kernel **does not look at** — hence the name *à trous*,
"with holes". **There are still nine weights and nine multiplications**: all
that changes is how wide the covered area is, `(2r+1)×(2r+1)`.

### Why it was needed

Up to this point the only way the network has of seeing further is **pooling** —
which is also what **destroys the exact position** of thin strokes, i.e. the
signal we want to detect. The two are tied together: more context = less
precision.

Dilation **unties them**: context without shrinking anything.

### The "pyramid"

Four branches **in parallel** on the same input. At the bottleneck of a 400×400
image we are at 25×25, and every pixel there is worth 16 original pixels:

| branch | covers (bottleneck px) | ≈ original px |
|---|---|---|
| dilation 1 | 3×3 | 48×48 |
| dilation 2 | 5×5 | 80×80 |
| dilation 4 | 9×9 | 144×144 |
| dilation 8 | 17×17 | **272×272** |

The dilation-1 branch is an ordinary conv, included on purpose so that the
finest local scale is not lost.

```python
branches = []
for rate in (1, 2, 4, 8):
    b = Conv2D(1024, 3, dilation_rate=rate, padding="same")(x)
    b = GroupNormalization(32)(b)
    b = ReLU(b)
    branches.append(b)

x = Concatenate()(branches)        # 25 × 25 × 4096
x = Conv2D(1024, 1)(x)             # remix and go back to 1024
x = GroupNormalization(32)(x)
x = ReLU(x)
```

The final 1×1 conv is not a bookkeeping trick to make the shapes work out: it is
where the network **learns how much weight to give each scale**. Its output has
the same shape as a regular `_conv_block`, which is why the block is a drop-in
replacement.

### Cost

Dilation itself is **free** — same kernel, same weights. What costs is doing
four of them instead of one:

| | bottleneck parameters |
|---|---|
| regular `_conv_block` (2 convs in series) | ≈ 14 M |
| ASPP bottleneck (4 branches + projection) | ≈ 23 M |

### The known defect

At high dilation the kernel samples a sparse grid, and adjacent pixels can read
**disjoint** input sets: the result can be a checkered incoherence, the
*gridding artifact*. It is a known property of the method, not a bug in the
implementation — and a plausible candidate for the instability observed.

### Outcome

**No benefit** in either fidelity or detection, plus genuine optimization
instability.

`unet_v2_dilated` is this same bottleneck dropped into `unet_v2`, with the same
three toggles on.

---

## 8. `unet_restormer` — transposed attention

### What changes

**One extra block** right after the bottleneck:

```python
x = _conv_block(x, 1024)
x = RestormerBlock(dim=1024, num_heads=8)(x)     # ← the only addition
```

### The problem it solves

In classical self-attention every position looks at **all** the others: that
requires an `N × N` table. And this is the wall:

| image | positions at the bottleneck | table entries |
|---|---|---|
| 400×400 crop | 25 × 25 = **625** | ~390,000 |
| painting GT01, 3674×2834 | 230 × 178 ≈ **41,000** | **~1.7 billion** |

Inference on a whole painting is a project invariant. With classical attention
it would be impossible.

### The reversal

> not *"which **positions** resemble which positions"*
> but *"which **channels** resemble which channels"*

With `dim=1024` and 8 heads, each head works on 128 channels: the table is
**128 × 128, always**, whatever the image size.

### The steps

Input `25 × 25 × 1024`.

**1 · Produce `q`, `k`, `v`**

```
Conv1×1 → 3072 channels    (remix the channels)
Depthwise 3×3              (add local context)
split into three           → q, k, v, each 25 × 25 × 1024
```

*A depthwise conv looks at the neighbours **keeping the channels separate** —
one small filter per channel. A cheap way to add spatial context.*

**2 · Reshape** — no computation, just a change of shape:

```
from   25 × 25 × 1024
to     8 heads × 128 channels × 625 pixels
```

Each channel is no longer an image, but a row of 625 numbers.

**3 · Compare every channel with every other channel**

```
q · kᵀ  →  8 × 128 × 128
```

Each cell is the product of two channels **summed over all pixels**: it measures
how much two channels fire together across the whole image.

**This is where the global receptive field comes from**: the sum runs over the
entire image.

**4 · Softmax**, with a learned per-head `temperature`.

**5 · Recombine** — `table · v`, reshape back, final `Conv1×1`.

**The result:** every output channel is a **mixture of the input channels**,
with weights computed by looking at the whole image.

**6 · Residual sum**: `x = input + attention(input)`.

### Worked numerical example

A 2×2 image, 3 channels. Unrolled:

```
A = [ 9, 0, 0, 9 ]
B = [ 8, 1, 1, 8 ]
C = [ 0, 9, 9, 0 ]
```

```
A · B  =  9×8 + 0×1 + 0×1 + 9×8  =  144    ← high: they fire together
A · C  =  9×0 + 0×9 + 0×9 + 9×0  =    0    ← never together
```

A `3 × 3` table, which **does not depend on how many pixels there are**. If the
image became 100×100 the rows would grow from 4 to 10,000 numbers, but the table
would stay `3 × 3`. **That is why the cost is linear.**

After the softmax, `row A ≈ [0.55, 0.45, 0.00]`, so:

```
A_new  =  0.55·A + 0.45·B + 0.00·C
```

*"Channel A is reconstructed using itself and B, ignoring C."*

*In the real block `q` and `k` are L2-normalized before the product, so the
table measures the **shape** of the co-activation and not its intensity.*

### The second half: GDFN

```
Conv1×1  →  4096 channels        (expand)
Depthwise 3×3                    (local context)
split in half  →  x1, x2         (2048 each)
GELU(x1) × x2                    (x2 acts as a valve on x1)
Conv1×1  →  back to 1024
```

One half produces the signal, the other decides **how much of it to let
through**. Same principle as `attention_unet`'s mask, but applied to channels
instead of positions.

### Comparison with the gate

| | `attention_unet` gate | Restormer block |
|---|---|---|
| produces | a 1-channel **spatial mask** | a **recombination of the channels** |
| says | *where* to look | *which features* to mix |
| does it look at neighbours? | **no**, entirely pointwise | yes, and **the whole image** |
| receptive field | local | **global** |
| normalization | independent sigmoid per pixel | softmax, weights compete |
| where it acts | on the skips | at the bottleneck |

### Outcome

Within the band. It is nonetheless the ready answer to *"did you try
transformers?"*: a single block, at the bottleneck, deliberately minimal — the
question was *"does global context help?"*, not *"shall we replace the
architecture?"*.

---

## 9. `unet_residual` — residual head

### What changes

Inside, it is **literally `unet`**: it imports the same `_conv_block` from
`scripts.unet`. Only **the last step** changes.

```
RGB ──┬──────────► U-Net (identical) ──► Conv1×1 tanh ──► residual ∈ [−1, +1]
      │                                                         │
      └──► mean(R,G,B) ──► grey ∈ [0, 1] ───────────► Add ◄─────┘
                                                        │
                                            straight-through clip
                                                        │
                                                   IR ∈ [0, 1]
```

### The idea, with one pixel

RGB = `(0.6, 0.5, 0.4)` → grey = `0.5`.

| | what the network must produce |
|---|---|
| `unet` | **0.55** — the IR value, from scratch |
| `unet_residual` | **+0.05** — only the difference |

And with a charcoal stroke underneath (real IR 0.2): `unet` must say **0.20**,
`unet_residual` **−0.30**.

The nature of the job changes: from *"guess the value"* to *"guess the
correction"*. Visible grey and IR are largely correlated, so most of the answer
is already there **for free** — the grey path does not even have a parameter.
This is *global residual learning* (VDSR, DnCNN).

### Not to be confused with `resunet`

```
resunet         ──► shortcut INSIDE every block, around two convs
                    ×9 times,  it serves the GRADIENT

unet_residual   ──► shortcut around the WHOLE network, only at the end
                    ×1 time,   it serves to make the TARGET easier
```

In `resunet` the blocks are modified and the head is the usual sigmoid. In
`unet_residual` the blocks are untouched and only the head changes.

### The three pieces

**1 · `tanh` instead of `sigmoid`** — the residual must be **signed**: an IR
pixel can be darker than the grey (an underdrawing showing through) or lighter.
With a sigmoid the network could only *add*. On that head the initialization is
Glorot, not He, because `he_normal` presupposes a ReLU.

**2 · The flat mean, not luminance** — `(R+G+B)/3`, not
`0.299R + 0.587G + 0.114B`. Those weights model the sensitivity of the **human
eye** to visible light and say nothing about reflectance in the near infrared.
With no reason to privilege one channel, the unweighted mean is the neutral
choice.

**3 · The straight-through clip** — the grey lies in [0,1] and the residual in
[−1,+1], so their sum can land anywhere between −1 and 2. It must be brought
back into [0,1] because the loss (`max_val=1.0`) requires it.

An ordinary clip has **zero derivative** outside the interval: a pixel that came
out at 1.4 would receive no gradient and would be **dead forever**. And this is
not hypothetical — the grey alone already spans all of [0,1].

```python
x + stop_gradient(clip(x) − x)
```

| | |
|---|---|
| **forward** | `x + (clip(x) − x) = clip(x)` — exactly clipped |
| **backward** | `stop_gradient` zeroes the second term → derivative **1** |

With a pixel at 1.4 and target 0.9: the ordinary clip gives gradient 0 and stays
stuck; the straight-through version pushes it down by 0.5. The sigmoid of the
other architectures is smooth but **saturates** — a tiny gradient, not a null
one.

### Outcome

**A failure, and an instructive one.** Its `structural_delta` matches **exactly
the `mean(R,G,B)` floor** — 0.67 — at *every* value of `α`.

With `α = 0.16` the `tanh` head **collapses onto the identity**: the grey path
already satisfies nearly everything an MS-SSIM-dominated loss asks for, so there
is not enough gradient pressure left to build a residual.

The failure did not come from where the docstring feared it would (a residual
too large to learn, because pigments with the same grey value can be IR-opaque
or IR-transparent) but from the opposite side: **the loss never asked the
network to try**.

---

## 10. `efficientnet_unet` and `_ft` — pretrained encoder

### What changes

The other variants change one piece of `unet`. This one **throws the whole
encoder away**.

```
              ENCODER (EfficientNetB0, frozen)        DECODER (new)

RGB ─► ×255 ─► ┌─ H/2   16 channels ──────────────────────────────► 16
               ├─ H/4   24 channels ───────────────────────► 32
               ├─ H/8   40 channels ─────────────► 64
               ├─ H/16  112 channels ───► 128
               └─ H/32  1280 channels ─► 256
                                                    ──► Conv1×1 sigmoid ─► IR
```

### The idea

863 pairs against **1.2 million ImageNet images**. The early layers of any CNN
learn generic things — edges, textures, gradients: there is nothing specifically
pictorial about an edge detector.

The encoder stays **frozen** (`backbone.trainable = False`); **only the decoder**
is trained.

### Difference 1 — it goes one level deeper

| | minimum depth | bottleneck channels |
|---|---|---|
| `unet` | H/16 | 1024 |
| `efficientnet_unet` | **H/32** | **1280** |

And the channels are distributed the opposite way round:

| level | `unet` | EfficientNet |
|---|---|---|
| shallowest | 64 | **16** |
| | 128 | 24 |
| | 256 | 40 |
| | 512 | 112 |
| bottleneck | 1024 | 1280 |

EfficientNet is **thin at the surface and fat in depth**. The concrete
implication: the skips carry **far less fine information** (16 and 24 channels
against 64 and 128) — precisely what thin strokes need.

### Difference 2 — why the decoder does not mirror the encoder

Three reasons, in order of weight.

**EfficientNet's numbers are not comparable.** Its blocks are **MBConv**
(*inverted residual*): internally they **expand the channels 6×**, do a
depthwise conv, and project back down.

```
MBConv:   24 channels ──expand ×6──► 144 ──depthwise──► 144 ──project──► 24
                                      ↑                                   ↑
                             working width                     what you read off
```

So "24 channels" is not the width EfficientNet reasons with: it is the
**compression point** between two blocks. The decoder's blocks, by contrast, are
plain `Conv3×3 → GN → ReLU`, where the stated number **is** the working width.
Copying 16 and 24 would mean copying a number without its meaning.

**The decoder has the harder job**: the encoder has to *summarize*, the decoder
has to *reconstruct* — and it is the part the residual is computed on. In fact
the chosen decoder is **wider than the mirror** at the middle stages (128 against
112, 64 against 40, 32 against 24).

**The 1280 → 256 compression concentrates the whole budget.** Without it, the
decoder's first conv would be 1280→1280 with a 3×3 kernel, ~14.7 M parameters on
its own. Working the numbers out on the actual decoder (estimate):

| piece | parameters |
|---|---|
| 1280 → 256 compression | **≈ 3.5 M** |
| the four upsampling stages | ≈ 0.7 M |
| final stage + head | ≈ 0.01 M |
| **total trainable** | **≈ 4.3 M** |

**Over 80% of the trainable capacity sits in that single initial compression.**
Doubling that first number would almost double the whole trainable model, to buy
capacity at the lowest resolution — that is, where it is least needed.

### The three implementation details

**`inputs * 255.0`** — the dataset is normalized to [0,1], but the pretrained
EfficientNet expects **[0,255]**. Feeding it inputs 255 times smaller would make
the pretrained features meaningless.

**`_ResizeToMatch`** — going down, EfficientNet **rounds down** (25 → 12),
whereas `Conv2DTranspose(stride=2)` doubles exactly (12 → 24). The decoder
arrives with 24 pixels and the skip has 25: `Concatenate` would fail. The layer
reads the skip's size at runtime and resizes bilinearly.

**The encoder keeps its BatchNorms** — the ImageNet statistics are part of the
transferred knowledge. And since the encoder is frozen, those BatchNorms run in
inference mode with those statistics, which is exactly what we want.

*(The decoder goes down to 16 channels and 32 does not divide 16: this is the
case the `num_groups` helper exists for — it lowers the number of groups instead
of raising an error.)*

### The `_ft` variant

Not a different model: a **second phase**.

| | phase 1 | phase 2 (`_ft`) |
|---|---|---|
| encoder | frozen | **unfrozen** |
| learning rate | 1e-4 | **1e-5** |
| starting weights | random decoder | the phase-1 checkpoint |

**Why two phases:** at the start the decoder is random and produces large,
disorderly gradients. With the encoder already unfrozen, those gradients would
**destroy the pretrained features** before they could be of any use.

### Outcome

- `efficientnet_unet` groups with `unet` and `attention_unet`:
  PSNR **16.2–16.5**.
- `efficientnet_unet_ft` **does not beat the frozen baseline on any metric**: 863
  pairs are not enough to fine-tune an ImageNet network without eroding the very
  features that made it useful.

It is the only architecture that is **half somebody else's**: an encoder with
separable convolutions, squeeze-and-excite, Swish and BatchNorm; a decoder with
GroupNorm, ReLU and He.

---

## 11. The orthogonal axis: the heteroscedastic head

This is not a different family of architectures — it is a **different head**,
crossed with the five backbones carried forward: `unet_nll`, `resunet_nll`,
`attention_unet_nll`, `efficientnet_unet_nll`, `efficientnet_unet_nll_ft`.

| | deterministic | heteroscedastic |
|---|---|---|
| the network produces | a **value** | the **parameters of a distribution** |
| output | 1 channel, sigmoid | 2 channels: `μ` (sigmoid) + `log b` (linear, clipped) |
| the objective is to | minimize a **distance** | maximize a **likelihood** |

```python
mu      = Conv2D(1, 1, activation="sigmoid",  name="mu")(x)
log_var = Conv2D(1, 1, activation="linear", name="log_var_raw")(x)
log_var = ClipLogVar(log_var_min, log_var_max, name="log_var")(log_var)
outputs = Concatenate()([mu, log_var])
```

**Why the second channel has no activation**: positivity is already guaranteed
by the parametrization — the network predicts `log b`, and the loss computes
`b = exp(log b)`, positive by construction. A sigmoid would squash `log b` into
(0,1), i.e. `b ∈ (1, e)`: on data in [0,1] the model could **never** express
confidence. The clip to [−6, 6] is not an activation, it is a numerical guard.

> **Mind the name.** The layer is called `ClipLogVar` and the settings are
> `NLL_LOG_VAR_*`: these are **historical** names, from when that channel was a
> Gaussian log-variance. Since Round 2 every NLL architecture is trained as the
> **log-scale of a Laplace**. `config.py` documents this. In the presentation,
> say **scale**, not variance.

From there we get the standard deviation and the detection signal:

```
σ  =  b · √2

structural_z  =  structural_delta / smoothed(σ)
```

---

## 12. The two losses

### Deterministic — `combined_loss`

```
L  =  0.16 · Charbonnier  +  0.84 · (1 − MS-SSIM)
```

**Charbonnier** is a smoothed L1:

```
ρ(x) = √(x² + ε²) − ε        with  ε = 1e-3,  x = IR_true − IR_predicted
```

| regime | behaviour |
|---|---|
| `\|x\| ≫ ε` | `ρ ≈ \|x\|` → **like L1**: robust, optimum at the median, sharp edges |
| `\|x\| ≪ ε` | `ρ ≈ x²/2ε` → **like L2**: smooth gradient near the optimum |

Gradient `ρ'(x) = x / √(x² + ε²)`: continuous everywhere and bounded in (−1, 1).
Pure L1 jumps from −1 to +1 at zero; L2 is unbounded and, under uncertainty, its
minimum is the **mean** of the hypotheses — that is, **blur**.

With data in [0,1], `ε = 1e-3` is worth ~0.25 grey levels out of 255: **below
JPEG quantization**. This is not a compromise between L1 and L2 — it is **L1
with a mathematical defect removed**.

**MS-SSIM** decomposes local similarity into luminance × contrast ×
**structure**, over a 5-scale pyramid. The structure term is **invariant to
local shifts in grey level**.

**Why both are needed:**

| test | Charbonnier | MS-SSIM |
|---|---|---|
| add 0.1 to every pixel | = 0.1, **sees the error** | stays very high, the structure term is **identical** |
| blur the image | stays modest | **collapses** |

Charbonnier **anchors the absolute scale** — indispensable, because the residual
is computed as a difference. MS-SSIM **defends the structure** — indispensable,
because the signal is shapes.

**Why 0.84 on MS-SSIM**: it is the `Mix` configuration of Zhao et al. (2016), and
it matches the objective — the deliverable is a map of shapes. **It is also the
explanation of the 16–19 dB PSNR**: we optimized structure at 84%, not absolute
value.

The sweep over `α ∈ {0.16, 0.50, 0.84}` confirms that the weight was tested, not
copied: raising `α` improves fidelity but `unet` drops from 0.71 to 0.67 (the
floor); `resunet` at 0.50 is the only genuine gain; at 0.84 `resunet` collapses.

> **Precise attribution**: the `Mix` of Zhao et al. uses **L1**, not Charbonnier.
> The substitution is ours. The correct phrasing: *"the Mix configuration of Zhao
> et al., with the L1 term replaced by its smooth Charbonnier surrogate"* — and
> that is why LapSRN is reference [4] on the slide.

### Heteroscedastic — `laplace_nll_loss`

```
L  =  mean over all pixels of:

          sg(b)^β  ·  ( |y − μ|/b  +  log b )

      └──────────┘    └─────────┘    └────┘
       reweighting     error/scale    penalty

with  β = 0.5,   b = exp(log b),   sg = stop_gradient
```

It follows directly from the Laplace density:

```
p(y | μ, b)  =  (1 / 2b) · exp( −|y − μ| / b )

−log p  =  |y − μ|/b  +  log b  +  log 2
                                   └── constant, dropped
```

**The tension between the two terms:**

| term | wants | why |
|---|---|---|
| `\|y − μ\| / b` | `b` **large** | dividing by a large number makes the error cheap |
| `log b` | `b` **small** | it punishes declared uncertainty |

Without the second, the optimum would be `b → ∞`: *"I know nothing, so I am
never wrong."* With both, differentiating with respect to `b` with
`d = |y − μ|`:

```
 −d/b²  +  1/b  =  0       →       b = d
```

**The optimal `b` is exactly the error committed.** And **there is no supervision
on `b` whatsoever**: uncertainty emerges as a by-product of the likelihood — the
network learns to say where it will be wrong without anyone having shown it.

**Why Laplace and not Gaussian:**

1. **Consistency**: the L1 family beats L2 in image restoration — the same
   reasoning that led to Charbonnier, here inside the probabilistic scaffolding.
   A Gaussian would give `(y−μ)²/(2σ²) + ½·log σ²`: same structure, but L2.
2. **Heavier tails**, and that is the right prior: the residual contains
   anomalies *by construction* — they are the signal. A Gaussian would treat them
   as near-impossible events and would inflate `b` to accommodate them,
   corrupting the uncertainty map.

**The weight β** breaks a vicious circle. The gradient towards `μ` is divided by
`b`:

```
   μ is bad here  ──►  the error is large  ──►  b rises
          ▲                                       │
          └────────  μ gets no gradient  ◄────────┘
```

Multiplying the loss by `b^β` with `b` **detached from the gradient** leaves the
optimum at convergence unchanged — only the gradient magnitude changes:

```
b^β  ·  (1/b)   =   b^(β−1)
```

| β | attenuation | meaning |
|---|---|---|
| 0 | `1/b` | pure NLL, the problem is fully present |
| **0.5** | **`1/√b`** | halfway, the best-calibrated point in the sweep |
| 1 | `1` | gradient independent of the uncertainty |

> **Precise attribution**: Seitzer et al. define β-NLL for the **Gaussian
> variance**. Applying it to the **scale of a Laplace** is a motivated
> transposition, declared in the docstring — not the published formula. The
> literal Gaussian version does exist in the repo (`beta_gaussian_nll_loss`),
> implemented and tested, simply not chosen.

### The unresolved asymmetry

The deterministic models are trained **84% on a structural criterion**. The NLL
models on a **purely pixel-wise** likelihood, with no structural term at all. And
yet it is precisely those that win on structural detection.

It is an open inconsistency, and the most obvious next experiment for the
project: combining the two terms.
