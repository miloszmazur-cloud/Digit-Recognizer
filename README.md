# Pure NumPy Digit Recognizer

A lightweight feedforward neural network and interactive digit recognizer built from scratch using 
pure NumPy and standard Python libraries. Trained on MNIST image dataset with synthetic noise mapped
to an 11th "NOT" output class to prevent false positives.

---

## Features

- **Built From Scratch**: implements dense layers, forward propagation, sigmoid
  activation, and backpropagation using NumPy vector operations.
- **11th "NOT" Class**: output nodes include classes `0` through `9` alongside a dedicated `NOT`
  class representing "not a digit" to prevent false-positive digit hallucinations.
- **Preprocessing Pipeline**: drawing inputs are bounded and scaled from 300x300 to 28x28, and
  centered via center of mass to match the MNIST standard.
- **Training**: supports batch training on the OpenML MNIST digit dataset, as well as, interactive
  user-guided corrections directly in drawing canvas
- **Persistent Storage**: save and load model architectures, layer weights and biases using HDF5
  (`.h5`).

---

## Network Architecture

| Layer | Type | Nodes | Input Dim |
| :--- | :--- | :--- | :--- |
| Layer 0 | Dense (Hidden) | 128 | 784 |
| Layer 1 | Dense (Hidden) | 32 | 128 |
| Layer 2 | Dense (Output) | 11 | 32 |

(784 represents the flattened 28×28 input image)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/miloszmazur-cloud/Digit-Recognizer.git
   cd Digit-Recognizer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Launch the GUI

Run the drawing canvas and prediction interface:

   ```bash
   python ui.py
   ```

- Draw a digit inside the 300x300 canvas.
- Click **Run** to generate classification percentages across all 11 classes.
- Click **✓** next to chosen digit class to backpropagate a correction for the current drawing.
- Click **Save Model** to store updated weights into `model.h5`.

### 2. Train the Model from Scratch

Download the MNIST dataset via `scikit-learn` and run the offline training routine:

   ```bash
   # Run with default settings (2 epochs, seed 123)
   python train_model.py
   
   # Or customize training parameters via CLI arguments
   python train_model.py --epochs 5 --seed 42
   ```

**Supported Arguments:**

* `--epochs` *(int, default: `2`)*: Total cycles through the MNIST training dataset.
* `--seed` *(int, default: `123`)*: Random seed applied to both `numpy` and `random` for
  deterministic weight initialization and sample shuffling.

The script trains the network sample by sample, periodically injects random binary noise vectors to
train 11th `NOT` class, and exports the trained weights and biases to `model.h5`.

---

## Project Structure

   ```text
   ├── model.py          # Core neural network - Node/Layer classes
   ├── train_model.py    # Training routine using OpenML MNIST dataset
   ├── ui.py             # Tkinter-based canvas and interactive classification dashboard
   ├── model.h5          # Trained weights and biases based on OpenML MNIST (HDF5 format)
   ├── requirements.txt  # Python package dependencies
   ├── .gitignore        # Git exclusion list
   └── README.md         # Project documentation
   ```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.