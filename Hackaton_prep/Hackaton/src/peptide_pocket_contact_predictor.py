"""
Peptide-Pocket Binding Prediction Pipeline
==========================================

This script implements an end-to-end pipeline for predicting which residues
in a peptide bind to predefined receptor pockets based on their 3D structure and
sequence information.

Flow Overview:
--------------
1. **Bio-structure parsing with Biopython:**
   Parse PDB files to extract 3D C-alpha coordinates of peptide residues and receptor pocket residues.

2. **ESM-2 embedding integration:**
   Generate rich residue-level embeddings for peptide sequences using the ESM-2 protein language model.

3. **Residue-level multi-label classifier (LSTM + Fully Connected):**
   Model peptide sequences as embeddings and predict a multi-label binding classification for each residue (binding to one or more of 5 pockets).

4. **Label generation from 3D structure (distance-based contacts):**
   Create ground-truth binding labels by calculating distances between peptide residues and pocket residues in 3D; residues within a threshold distance are labeled as binding.

5. **Training and evaluation routines (with F1 score):**
   Train the model using a binary cross-entropy loss with logits and evaluate performance with the micro-averaged F1 score to handle multi-label classification.

This pipeline combines structural bioinformatics with modern deep learning and protein language models to accurately predict peptide-pocket interactions at the residue level.

---

"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Hackaton_prep.src.esm_embeddings import get_esm_embeddings, get_esm_model
from Bio.PDB import PDBParser
from typing import List, Tuple
from sklearn.metrics import f1_score


class PeptidePocketDataset(Dataset):
    """
    Dataset holding peptide sequences, their binding labels, and embeddings.

    Args:
        sequences (List[str]): List of peptide sequences.
        labels (List[np.ndarray]): List of binary contact label matrices of shape (L, 5).
        embeddings (List[np.ndarray]): List of residue-level embeddings of shape (L, D).

    This dataset returns tensor pairs (embedding, labels) for training and evaluation.
    """

    def __init__(self, sequences: List[str], labels: List[np.ndarray], embeddings: List[np.ndarray]):
        self.sequences = sequences
        self.labels = labels  # shape: (L, 5) for each peptide
        self.embeddings = embeddings  # shape: (L, D) for each peptide

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx],
                                                                                     dtype=torch.float32)


class ResiduePocketClassifier(nn.Module):
    """
    BiLSTM + Fully Connected classifier for multi-label residue-pocket binding prediction.

    Architecture:
    - Bidirectional LSTM to model sequential dependencies in peptide embeddings.
    - Fully connected layers with ReLU activation and dropout for classification.

    Args:
        embedding_dim (int): Dimensionality of input embeddings (default 640 for ESM-2).
        hidden_dim (int): Number of hidden units in the LSTM layer.
        num_pockets (int): Number of receptor pockets to predict binding for (default 5).
    """

    def __init__(self, embedding_dim=640, hidden_dim=128, num_pockets=5):
        super().__init__()
        self.bi_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_pockets),
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: Raw logits of shape (batch_size, seq_len, num_pockets).
        """
        lstm_out, _ = self.bi_lstm(x)  # (B, L, 2*hidden_dim)
        logits = self.classifier(lstm_out)  # (B, L, num_pockets)
        return logits


def compute_contact_labels(peptide_coords: np.ndarray, pocket_coords: List[np.ndarray],
                           threshold: float = 4.0) -> np.ndarray:
    """
    Generate binary contact labels based on distance threshold between peptide residues and pocket residues.

    Args:
        peptide_coords (np.ndarray): Coordinates of peptide residues, shape (L, 3).
        pocket_coords (List[np.ndarray]): List of 5 arrays, each with pocket residue coordinates.
        threshold (float): Distance cutoff in Angstroms to define contact (default 4.0).

    Returns:
        np.ndarray: Binary contact matrix of shape (L, 5) indicating contacts.
    """
    L = peptide_coords.shape[0]
    labels = np.zeros((L, 5), dtype=int)
    for i, pocket in enumerate(pocket_coords):
        dists = np.linalg.norm(peptide_coords[:, None, :] - pocket[None, :, :], axis=-1)
        min_dist = np.min(dists, axis=1)
        labels[:, i] = min_dist <= threshold
    return labels


def extract_coords_from_structure(pdb_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Parse PDB file and extract C-alpha coordinates for peptide and 5 receptor pockets.

    Args:
        pdb_path (str): Path to PDB file containing peptide-receptor complex.

    Returns:
        Tuple:
            - peptide_atoms (np.ndarray): Peptide C-alpha coordinates (L, 3).
            - pocket_coords (List[np.ndarray]): List of 5 arrays with pocket C-alpha coordinates.

    Note:
        Chains labeled 'P' represent peptide.
        Chains labeled 'R1' to 'R5' represent pockets 1 to 5.
        Residue filtering is simplified and may need adjustment based on your PDB files.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)

    peptide_atoms = []
    pocket_coords = [[] for _ in range(5)]

    for model in structure:
        for chain in model:
            for res in chain:
                if 'CA' not in res:
                    continue
                atom = res['CA']
                res_id = res.get_id()[1]
                chain_id = chain.get_id()

                if chain_id == "P":
                    peptide_atoms.append(atom.coord)
                elif chain_id.startswith("R") and res_id in range(0, 20):  # Example condition for pockets
                    pocket_index = int(chain_id[1]) - 1
                    pocket_coords[pocket_index].append(atom.coord)

    return np.array(peptide_atoms), [np.array(p) for p in pocket_coords]


def train_model(model, dataloader, epochs=10, lr=1e-3):
    """
    Train the residue-pocket classifier.

    Args:
        model (nn.Module): ResiduePocketClassifier model instance.
        dataloader (DataLoader): DataLoader yielding batches of (embeddings, labels).
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimizer.

    Prints:
        Training loss per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


def evaluate_model(model, dataloader):
    """
    Evaluate the model's multi-label binding prediction performance using F1 score.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for evaluation data.

    Prints:
        Micro-averaged F1 score over all pockets.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x)
            preds = torch.sigmoid(logits) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    y_true = np.concatenate(all_labels).reshape(-1, 5)
    y_pred = np.concatenate(all_preds).reshape(-1, 5)
    f1 = f1_score(y_true, y_pred, average='micro')
    print(f"F1 Score: {f1:.3f}")


def main():
    """
    Main routine demonstrating:
    - Loading peptide sequences and PDB files.
    - Extracting ESM embeddings for sequences.
    - Parsing structures to extract coordinates.
    - Computing contact labels from 3D structure.
    - Creating dataset and dataloader.
    - Training and evaluating the model.

    Replace example sequences and pdb_files with your real data.
    """
    # Example peptide sequences and their corresponding PDB files
    peptide_sequences = ["ACDEFGHIK"]  # Replace with actual peptide sequences
    pdb_files = ["example_structure.pdb"]  # Replace with your PDB file paths

    # Load pretrained ESM-2 model and utilities for embedding extraction
    model_esm, alphabet, batch_converter, device = get_esm_model(embedding_size=640)

    embeddings, labels = [], []
    for seq, pdb in zip(peptide_sequences, pdb_files):
        # Extract ESM embeddings for the peptide sequence
        emb = get_esm_embeddings([seq], model_esm, alphabet, batch_converter, device, sequence_embedding=False)[0]

        # Parse PDB structure to get coordinates
        pep_coords, pocket_coords = extract_coords_from_structure(pdb)

        # Compute binary contact labels based on spatial proximity
        label = compute_contact_labels(pep_coords, pocket_coords)

        embeddings.append(emb)
        labels.append(label)

    # Prepare dataset and dataloader
    dataset = PeptidePocketDataset(peptide_sequences, labels, embeddings)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize model
    model = ResiduePocketClassifier()

    # Train and evaluate
    train_model(model, dataloader, epochs=10)
    evaluate_model(model, dataloader)


if __name__ == "__main__":
    main()
