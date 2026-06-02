# Deep Layers

Deep learning pipeline for RGB → infrared (IR) image translation applied to **reflectography** of paintings. The model learns to predict the expected IR appearance of a painting from its RGB photograph; deviations between the real IR image and the prediction (`delta = |real_IR − predicted_IR|`) reveal underdrawings and preparatory sketches hidden beneath paint layers.

---

## Methodology

### Problem framing

Infrared reflectography is a non-invasive technique used in art conservation to examine the layers beneath a painting's surface. Carbon-based underdrawings (charcoal, chalk, lead-point) absorb IR radiation differently from paint layers, making them visible in IR images.

Rather than analysing IR images directly, this pipeline learns the mapping RGB → predicted IR from paired training data. The residual signal

```
delta = |real_IR − predicted_IR|
```

isolates regions where the observed IR deviates from what the model expects given the paint surface — a strong indicator of underlying structure.

### Architectures

Three encoder–decoder architectures are implemented and benchmarked:

| Architecture | Key feature |
|---|---|
| **UNet** | Standard skip connections; solid baseline |
| **ResUNet** | Residual blocks in encoder/decoder; better gradient flow on small datasets |
| **Attention UNet** | Additive attention gates on skip connections; suppresses irrelevant background features |

All models accept images of any spatial size at inference time (dynamic padding to the nearest multiple of 16).

### Training

- **Input**: RGB image `(H, W, 3)`, values normalised to `[0, 1]`
- **Target**: IR image `(H, W, 1)`, grayscale, values normalised to `[0, 1]`
- **Loss**: combined MAE + (1 − SSIM), weighted by `LOSS_ALPHA` (default 0.7)
- **Augmentation**: random horizontal/vertical flips applied to both channels; brightness/contrast jitter on RGB only (IR reflectance is a physical property, not an illumination artifact)
- **Data split**: by artwork ID — all sections of the same painting are assigned to a single fold, preventing leakage between train, validation, and test sets

---

## Project Structure

```
deep_layers/
├── data/
│   ├── ir/                     # Infrared images (not versioned)
│   └── rgb/                    # RGB images (not versioned)
├── models/                     # Saved checkpoints: {arch}/best_model.keras
├── logs/                       # TensorBoard event files
├── notebooks/
│   ├── 010_eda.ipynb           # Dataset exploration and split validation
│   ├── 020_training.ipynb      # Train all three architectures
│   ├── 030_evaluation.ipynb    # Quantitative comparison on the test set
│   └── 040_inference.ipynb     # Inference and delta/underdrawing visualisation
├── scripts/
│   ├── config.py               # Pydantic settings (paths, hyperparameters)
│   ├── dataset.py              # Pair loading, grouped split, tf.data pipeline
│   ├── augmentation.py         # TF-native augmentation
│   ├── unet.py                 # UNet model
│   ├── resunet.py              # ResUNet model
│   ├── attention_unet.py       # Attention UNet model
│   ├── losses.py               # Combined MAE + (1 − SSIM) loss
│   ├── metrics.py              # PSNR and SSIM Keras metric wrappers
│   ├── trainer.py              # Model factory, compilation, callbacks, checkpoint loading
│   └── visualization.py        # Plotting utilities
├── pyproject.toml              # Ruff and pytest configuration
├── requirements.txt            # Python dependencies
└── LICENSE                     # CC BY-SA 4.0
```

---

## Setup

**Requirements**: Python 3.11.5, pip.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

On Apple Silicon, `tensorflow-metal` (included in `requirements.txt`) enables GPU acceleration via the Metal backend automatically — no additional configuration is needed.

### Running the notebooks

Execute them in order from the `notebooks/` directory:

```
010_eda.ipynb       → explore the dataset and validate the split
020_training.ipynb  → train UNet, ResUNet, Attention UNet
030_evaluation.ipynb→ compare architectures on the test set
040_inference.ipynb → run predictions and inspect the delta heatmap
```

```bash
jupyter notebook notebooks/
```

### Linting

```bash
ruff check scripts/
ruff format scripts/
```

---

## License

This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](LICENSE) license.
