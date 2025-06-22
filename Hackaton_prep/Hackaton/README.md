# NES-CRM1 Binding Prediction Project

## Project Overview

This project focuses on **predicting which residues in Nuclear Export Signal (NES) peptides bind to the CRM1 protein**, using a deep learning approach that relies only on sequence information. Instead of manually defined motifs or structural inputs, we use **pretrained protein language model embeddings (ESM2)** combined with a **BiLSTM classifier** to predict residue-level binding directly from peptide sequences.

## Why NES?

NES peptides are short, leucine-rich sequences that guide proteins out of the nucleus by binding to the CRM1 export receptor. Understanding which specific residues in NES peptides bind CRM1’s five hydrophobic pockets is important for elucidating nuclear export mechanisms and developing therapeutics, as CRM1 is a validated target in cancer treatment.

## Dataset and Labeling

The training data consists of **CRM1–NES peptide complex structures** predicted by **AlphaFold**. Each complex contains a CRM1 protein (typically chain A) and a NES peptide (chain B).

To generate residue-level labels:

   *We compute the **minimum heavy-atom distances** between each peptide residue and the CRM1 structure.

   *We then **label the top 5 closest residues** in each peptide as binding residues (label = 1), based on CRM1’s five known binding pockets.

   *All other residues are labelled as non-binding (label = 0).

This **top-5 labeling strategy** ensures consistency with CRM1’s binding capacity and simplifies training.

## Pipeline Summary

1. **Parse AlphaFold CRM1–peptide structures** (mmCIF or PDB format) using Biopython.

2. **Extract sequences** and filter for standard amino acids.

3. **Label binding residues** by computing the top-5 closest peptide residues to CRM1 atoms.

4. **Generate residue-level embeddings** using the ESM2 pretrained protein language model.

5. **Train a BiLSTM model** on these embeddings to predict binding probabilities per residue.

6. **Classify binding residues** at inference by selecting the top 5 scoring positions per peptide.

7. **Evaluate the model** using residue-level metrics like precision, recall, and F1 score.
---

## Installation

### Prerequisites

```bash
pip install numpy torch pandas matplotlib scikit-learn fair-esm biopython tqdm
```

### Required Python packages:
- `numpy` - Numerical computations
- `torch` - PyTorch deep learning framework
- `fair-esm` - ESM protein language model
- `biopython` - PDB structure parsing
- `scikit-learn` - Model evaluation metrics
- `tkinter` - GUI interface (usually included with Python)
- `tqdm` - Progress bars
- `matplotlib` - Plotting and visualization

## Usage

### 1. Data Processing

Process your PDB/mmCIF structure files to extract peptide sequences and binding masks:

```python
python data_processing.py
```

This script implements the labeling pipeline described above:
- Reads PDB/mmCIF files from `data/structures/af_positives/`
- Extracts peptide sequences (chain B) and CRM1 atoms (chain A)
- Computes distance-based binding masks using Euclidean distance
- Identifies top-5 closest residues as binding sites
- Saves processed data to `data/peptide_data.pkl`

### 2. Interactive GUI Application

Launch the trained model interface:

```python
# Open the Jupyter notebook
jupyter notebook NES_CRM1_Predictor_GUI.ipynb
```

**Using the GUI:**
1. Enter an amino acid sequence in the input field
2. Click "Predict" to run the BiLSTM model
3. View per-residue binding predictions
4. The top-5 most likely binding residues are identified


### 3. Grid Search Experiments

To optimize model hyperparameters, three different grid search scripts are provided:

#### 📄 `grid_search_binding_predictor_esm_extended.py`

* **Purpose:** Extended search using multiple architectural options
* **Features Tuned:** MLP usage, dropout, weighted loss, classifier depth
* **Run with:**

  ```bash
  python src/grid_search_scripts/grid_search_binding_predictor_esm_extended.py
  ```
* **Results saved to:** `results/results_ext.csv`

#### 📄 `grid_search_binding_predictor_esm_with_weight.py`

* **Purpose:** Simpler grid search with class weighting enabled
* **Run with:**

  ```bash
  python src/grid_search_scripts/grid_search_binding_predictor_esm_with_weight.py
  ```
* **Results saved to:** `results/results.csv`

#### 📄 `grid_search_binding_predictor_esm_without_weight.py`

* **Purpose:** Grid search without using class weights
* **Run with:**

  ```bash
  python src/grid_search_scripts/grid_search_binding_predictor_esm_without_weight.py
  ```
* **Results saved to:** `results/results.csv`

These grid searches systematically test parameter combinations for best performance on peptide binding prediction.

## Model Architecture

### ESM-2 + BiLSTM Approach
- **Input**: NES peptide amino acid sequence
- **Embedding**: ESM-2 protein language model (layer 6, 640D or 1280D)
- **Network**: Bidirectional LSTM for sequence context modeling
- **Output**: Per-residue binary classification (binding vs non-binding)

### Training Strategy
- **Top-k selection**: Focus on 5 most likely binding residues per peptide
- **Loss function**: Binary cross-entropy with optional class weighting

## Key Functions

### `data_processing.py`
- `load_structure()`: Parse PDB/mmCIF files using BioPython
- `compute_residue_min_dist()`: Calculate minimum distance to CRM1 atoms
- `get_binding_mask()`: Generate binary labels using distance thresholds
- `extract_peptides_and_masks()`: Batch process structure files

### GUI Application (`NES_CRM1_Predictor_GUI.ipynb`)
- `SequencePredictorApp`: Main GUI interface class
- `_predict()`: ESM-2 embedding generation + BiLSTM prediction
- `topk_predictions()`: Adaptive thresholding for top-5 binding sites

## Example Workflow

### Input Sequence:
```
PRTHYGQKAILFLPLPVSSD
```

### Prediction Output:
```
Per-residue contact predication:
Pos 1 (P): 0.000
Pos 2 (R): 0.000
Pos 3 (T): 0.000
Pos 4 (H): 0.000
Pos 5 (Y): 0.000
Pos 6 (G): 0.000
Pos 7 (Q): 0.000
Pos 8 (K): 0.000
Pos 9 (A): 0.000
Pos 10 (I): 1.000 ← Top-5 binding site
Pos 11 (L): 1.000 ← Top-5 binding site
Pos 12 (F): 1.000 ← Top-5 binding site
Pos 13 (L): 1.000 ← Top-5 binding site
Pos 14 (P): 1.000 ← Top-5 binding site
Pos 15 (L): 0.000
Pos 16 (P): 0.000
Pos 17 (V): 0.000
Pos 18 (S): 0.000
Pos 19 (S): 0.000
Pos 20 (D): 0.000
```

## Model Variants

The project includes two model configurations:

1. **With class weighting** (`model_emb_best_weight.pt`)
   - 640D ESM embeddings
   - Handles class imbalance in binding vs non-binding residues

2. **Without class weighting** (`model_emb_best_without_weight.pt`)
   - 1280D ESM embeddings
   - Standard binary classification approach

## Requirements

- Python 3.11
- CUDA-compatible GPU (recommended for ESM-2 inference)
- At least 8GB RAM for model loading
- PDB/mmCIF structure files for training data

---

## ⚙️ Troubleshooting

### 🔧 Common Issues

1. **`ModuleNotFoundError`**
   → Make sure all dependencies are installed:

   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA Memory Errors**
   → Switch to CPU inference or reduce batch size.

3. **BioPython Parsing Errors**
   → Check your PDB file formatting and ensure chain IDs are correctly specified.

4. **GUI Display Issues**
   → Ensure `tkinter` is properly installed. On some systems, it may require separate installation:

   ```bash
   sudo apt-get install python3-tk   # for Debian/Ubuntu
   ```

5. **ESM Model Loading Fails**
   → Make sure you have an active internet connection for the initial model download.

---

## Clean Slate for Environment Issues

Follow these steps to create a fresh virtual environment and register it for Jupyter:

### 1. Navigate to project directory

```bash
cd ../Hackaton
```

### 2. Create a virtual environment (choose one of the two)

```bash
# Option A: Using exact Python version
py -3.11 -m venv .venv    # Windows
python3.11 -m venv .venv  # Linux/macOS
```

### 3. Activate the virtual environment

```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4. Install project dependencies

```bash
pip install -r requirements.txt
```

### 5. (Optional) Register the venv as a Jupyter kernel

```bash
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

### 6. In Jupyter Notebooks

* Open your `.ipynb` file.
* Select the kernel: **Python (.venv)** from the top-right menu.

---


### Configuration Notes
- Update `crm1_chain_id` and `peptide_chain_id` in `data_processing.py` if your PDB files use different chain identifiers
- Adjust top-k value based on your specific research needs
- Model paths in the GUI notebook may need updating based on your file structure
