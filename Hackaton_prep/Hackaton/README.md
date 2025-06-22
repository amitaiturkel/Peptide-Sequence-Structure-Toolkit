# NES-CRM1 Binding Prediction Project

## Project Overview

This project aims to **predict which amino acids in Nuclear Export Signal (NES) peptides bind to the five specific CRM1 protein pockets**. Accurately identifying these binding residues helps understand NES-CRM1 interactions and can guide drug design or biological research.

## Why NES?

NES peptides are short sequences that mediate the export of proteins from the nucleus by binding to the CRM1 export receptor. Understanding the precise residues within these NES peptides that physically interact with CRM1's binding pockets is crucial for decoding nuclear export mechanisms.

## Dataset and Labeling

To train a machine learning model for this task, we need a **labeled dataset** indicating which amino acids bind CRM1 pockets.

* We start with the **NESDB dataset**, which contains experimentally validated NES peptide sequences.
* For each NES peptide, we collect its **3D structural information** from Protein Data Bank (PDB) files or AlphaFold predicted structures.
* The CRM1 protein's 3D structure is also available, highlighting its five binding pockets.

## How 3D Structural Data Helps Labeling

The 3D structures provide the **atomic coordinates** (x, y, z) of every residue in both the NES peptide and the CRM1 protein.

**Label generation pipeline:**

1. Extract coordinates of alpha carbon (CA) atoms of each amino acid in the NES peptide and in each CRM1 pocket.
2. Calculate the **Euclidean distance** between every NES residue and the CRM1 pocket atoms.
3. Assign a **binding label (1)** to a residue if it lies within a threshold distance (e.g., 4 Å) from any CRM1 pocket atom; otherwise, label it as **non-binding (0)**.

This process creates binary binding labels for each residue, essential for supervised training.

## Pipeline Summary

1. **Parse NESDB data** to get NES peptide sequences.
2. **Load PDB or AlphaFold structures** of peptide-CRM1 complexes.
3. **Extract 3D coordinates** of peptides and CRM1 pockets.
4. **Compute binding labels** based on distance thresholds.
5. **Generate ESM-2 embeddings** for each peptide residue sequence.
6. **Combine embeddings and labels** into a dataset CSV file.
7. **Train a residue-level binary classifier (BiLSTM)** to predict binding residues.
8. **Evaluate the model** on validation and test sets.

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

## Model Architecture

### ESM-2 + BiLSTM Approach
- **Input**: NES peptide amino acid sequence
- **Embedding**: ESM-2 protein language model (layer 6, 640D or 1280D)
- **Network**: Bidirectional LSTM for sequence context modeling
- **Output**: Per-residue binary classification (binding vs non-binding)

### Training Strategy
- **Positive samples**: Residues within distance threshold of CRM1 pockets
- **Distance-based labeling**: Euclidean distance ≤ 4Å from CRM1 atoms
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
MALKLAGLDI
```

### Prediction Output:
```
Per-residue contact probability:
Pos 1 (M): 0.125
Pos 2 (A): 0.892  ← Top-5 binding site
Pos 3 (L): 0.756  ← Top-5 binding site  
Pos 4 (K): 0.234
Pos 5 (L): 0.681  ← Top-5 binding site
Pos 6 (A): 0.445
Pos 7 (G): 0.123
Pos 8 (L): 0.567
Pos 9 (D): 0.798  ← Top-5 binding site
Pos 10 (I): 0.612 ← Top-5 binding site
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
Here's a cleaned-up and corrected version of your troubleshooting and environment setup guide, with consistent formatting and accurate commands for both Windows and Linux users:

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
- Adjust distance thresholds and top-k values based on your specific research needs
- Model paths in the GUI notebook may need updating based on your file structure

## Biological Significance

This tool addresses a key challenge in structural biology: predicting protein-protein interactions from sequence alone. By combining:
- **Evolutionary information** (ESM-2 embeddings)
- **Structural constraints** (distance-based labeling)
- **Sequence context** (BiLSTM modeling)

The model can identify functionally important residues that mediate nuclear export, potentially informing:
- Drug target identification
- Protein engineering efforts
- Understanding of nuclear transport mechanisms





**Note**: This tool is designed for research purposes. Experimental validation is recommended for critical applications involving NES-CRM1 interactions.