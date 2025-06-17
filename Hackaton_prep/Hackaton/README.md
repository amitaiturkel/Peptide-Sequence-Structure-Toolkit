# NES-CRM1 Binding Prediction Project

## Project Overview

This project aims to **predict which amino acids in Nuclear Export Signal (NES) peptides bind to the five specific CRM1 protein pockets**. Accurately identifying these binding residues helps understand NES-CRM1 interactions and can guide drug design or biological research.

---

## Why NES?

NES peptides are short sequences that mediate the export of proteins from the nucleus by binding to the CRM1 export receptor. Understanding the precise residues within these NES peptides that physically interact with CRM1’s binding pockets is crucial for decoding nuclear export mechanisms.

---

## Dataset and Labeling

To train a machine learning model for this task, we need a **labeled dataset** indicating which amino acids bind CRM1 pockets.

* We start with the **NESDB dataset**, which contains experimentally validated NES peptide sequences.
* For each NES peptide, we collect its **3D structural information** from Protein Data Bank (PDB) files or AlphaFold predicted structures.
* The CRM1 protein's 3D structure is also available, highlighting its five binding pockets.

---

## How 3D Structural Data Helps Labeling

The 3D structures provide the **atomic coordinates** (x, y, z) of every residue in both the NES peptide and the CRM1 protein.

**Label generation pipeline:**

1. Extract coordinates of alpha carbon (CA) atoms of each amino acid in the NES peptide and in each CRM1 pocket.
2. Calculate the **Euclidean distance** between every NES residue and the CRM1 pocket atoms.
3. Assign a **binding label (1)** to a residue if it lies within a threshold distance (e.g., 4 Å) from any CRM1 pocket atom; otherwise, label it as **non-binding (0)**.

This process creates binary binding labels for each residue, essential for supervised training.

---

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


## How to Use

1. Prepare raw NESDB sequences using `parse_nesdb.py`.
2. Download or prepare PDB/AlphaFold structure files.
3. Run `compute_labels.py` to generate residue-level binding labels.
4. Generate ESM-2 embeddings with `esm_embeddings.py`.
5. Combine embeddings and labels into training CSV with `dataset_builder.py`.
6. Train and evaluate the model with `train_and_eval.py`.

---

Feel free to ask if you want me to help with any specific part of the code or workflow!
