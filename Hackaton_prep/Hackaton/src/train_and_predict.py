"""
train_and_predict.py

Pipeline Overview:
------------------
This script implements a full pipeline for predicting binding residues in peptides using residue-level embeddings.
The pipeline includes the following stages:

1. **Data Loading**: Peptide residue embeddings and labels are loaded from a CSV file.
   Each row represents a residue, and each column is an embedding dimension or label.

2. **Dataset Handling**: A custom Dataset class (PeptideDataset) wraps the data to work with PyTorch.

3. **Data Splitting**: The dataset is split into train/validation/test sets for model development and evaluation.

4. **Model Definition**: Uses a BiLSTM-based neural network (imported from `model.py`) to predict binary binding labels for each residue.

5. **Training**: The model is trained using Binary Cross-Entropy with logits on batches of peptides.

6. **Evaluation**: Accuracy, precision, recall, and F1 are computed on validation/test sets.

7. **Inference**: Final model performance is reported on the test set.

Each function in this script plays a role in preparing, training, and evaluating the model.

Typical Usage:
--------------
python train_and_predict.py --csv path/to/data.csv --epochs 20 --lr 0.001
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from model import ResidueBindingModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class PeptideDataset(Dataset):
    """
    Custom dataset class that loads peptide residue-level data from a CSV.
    The CSV should contain embedding vectors (float columns) and a 'label' column (0 or 1).

    Role in Pipeline:
    -----------------
    Used for feeding the residue embeddings and labels to the model during training/evaluation.
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.embeddings = df.iloc[:, :-1].values.astype(np.float32)
        self.labels = df["label"].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]), torch.tensor(self.labels[idx])


def collate_batch(batch):
    """
    Collates a batch of samples. Assumes each batch contains one full peptide (same length residues).

    Role in Pipeline:
    -----------------
    Ensures inputs are properly stacked for LSTM input. Works with batch_size=1 for peptides.
    """
    embeddings, labels = zip(*batch)
    embeddings = torch.stack(embeddings)
    labels = torch.stack(labels)
    return embeddings, labels


def train_model(train_loader, model, optimizer, criterion, device):
    """
    Trains the model on the training set for one epoch.

    Role in Pipeline:
    -----------------
    Executes forward pass, backpropagation, and optimization step for each peptide (batch of residues).
    """
    #TODO is this really the best way? look a bit strange
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.unsqueeze(0)  # batch_size=1
        y_batch = y_batch.unsqueeze(0)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(val_loader, model, criterion, device):
    """
    Evaluates the model on validation or test set and calculates loss + performance metrics.

    Role in Pipeline:
    -----------------
    Used after training to monitor generalization, and finally to report test performance.
    """
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.unsqueeze(0)
            y_batch = y_batch.unsqueeze(0)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_preds.extend(preds.flatten().tolist())
            all_labels.extend(y_batch.cpu().numpy().flatten().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    return val_loss / len(val_loader), acc, precision, recall, f1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to peptide CSV dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset and initialize splits
    dataset = PeptideDataset(args.csv)

    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)

    # Initialize model, optimizer, and loss function
    model = ResidueBindingModel(embedding_dim=dataset.embeddings.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_model(train_loader, model, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(val_loader, model, criterion, device)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}")

    # Final test evaluation
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(test_loader, model, criterion, device)
    print(f"Test Results | Loss: {test_loss:.4f} | Acc: {test_acc:.3f} | "
          f"Precision: {test_prec:.3f} | Recall: {test_rec:.3f} | F1: {test_f1:.3f}")
