# Deep Layers

## Project Description
A computer vision project focused on training a GAN to generate infrared images (IRR) from RGB inputs, with applications in the field of reflectography. The model leverages advanced deep learning techniques to enhance the analysis of artworks and other reflective surfaces.

## Group Members
- [Simone Caglio](https://github.com/SimoneFisico)
- [Marco Manduca](https://github.com/MarcoManduca)

## Tech Spec
-  **Language:** [Python 3.11.5](https://www.python.org/downloads/release/python-3115/) or earlier
-  **Frameworks and Libraries:** This project uses several libraries for data analysis, computer vision, deep learning, data visualization such as `numpy`, `pandas`, `matplotlib`, `tensorflow`, `keras`, and more. (See [requirements.txt](./requirements.txt) for the full list of dependencies.)
- This script ran on macOS with Apple Silicon Chip. Follow the instruction linked above to install tensorflow with accelerate training with Metal on Mac GPUs: [tensorflow-plugin](https://developer.apple.com/metal/tensorflow-plugin/). Some packages may be install differently and it's probably necessary to update some reference in the import statements to run the code.

## Installation
### Prerequisites
-  **Python:** Make sure you have Python 3.11.5 or a compatible version installed.
-  **Package Manager:** You will need `pip` or `conda` to install the dependencies.
### Installation Steps
1.  **Clone the repository:**

```bash
git clone git@github.com:MarcoManduca/deep_layers.git
cd  deep_layers
```
2.  **Create a virtual environment (optional but recommended):**
- Using venv:
```bash
python3  -m  venv  venv
source  venv/bin/activate  # On Windows: venv\Scripts\activate
```
- Using conda:
```bash
conda  create  --name  deep_layers  python=3.11.5
conda  activate  deep_layers
```
3.  **Install the dependencies:**
- With pip:
```bash
pip  install  -r  requirements.txt
```
- With conda:
```bash
conda  install  --file  requirements.txt
```

## Usage
Once all dependencies are installed, you can start exploring and playing with the generator model saved in [models folder](./models/). You can also try some IRR image prediction from RGB image (it's necessary to save the RGB image in the [RGB test folder](./data/test/rgb/)) and run the [notebook](./script/model_testing.ipynb) code:
```bash
jupyter  notebook ./script/model_testing.ipynb
```

Evenctually you can re-train the models applying custom configuration of the hyperparameter defined in the [environment file configuration](.env) and running the [script](./script/main.py)
```bash
python3  ./script/main.py
```
## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
