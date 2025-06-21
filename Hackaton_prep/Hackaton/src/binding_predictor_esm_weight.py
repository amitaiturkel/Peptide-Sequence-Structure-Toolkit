import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import esm
import numpy as np
from typing import List
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import itertools
import csv
import collections

from sklearn.metrics import classification_report


# ---------- Load ESM Model ----------
def load_esm_model(embedding_size=640):
    ESM_MODELS_DICT = {
        320: esm.pretrained.esm2_t6_8M_UR50D,
        480: esm.pretrained.esm2_t12_35M_UR50D,
        640: esm.pretrained.esm2_t30_150M_UR50D,
        1280: esm.pretrained.esm2_t33_650M_UR50D,
        2560: esm.pretrained.esm2_t36_3B_UR50D,
        5120: esm.pretrained.esm2_t48_15B_UR50D
    }
    model, alphabet = ESM_MODELS_DICT[embedding_size]()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, alphabet, batch_converter, device


# ---------- Extract ESM Embeddings ----------
def get_esm_embeddings(sequences: List[str], model, alphabet, batch_converter,
                       device, layer=6):
    batch = [(f"pep{i}", seq) for i, seq in enumerate(sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer])
    token_reps = results["representations"][layer]

    embeddings = []
    for i, seq in enumerate(sequences):
        emb = token_reps[i, 1:len(seq) + 1].cpu().numpy()
        embeddings.append(emb)
    return embeddings


# ---------- Dataset Class ----------
class PepResidueDataset(Dataset):
    def __init__(self, embeddings: List[np.ndarray], labels: List[np.ndarray]):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx],
                            dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.float32)


# ---------- Collate Function ----------
def pad_collate(batch):
    seqs, labels = zip(*batch)
    lens = [len(x) for x in seqs]
    padded_seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    mask = torch.arange(padded_seqs.shape[1])[None, :] < torch.tensor(lens)[:,
                                                         None]
    return padded_seqs, padded_labels, mask


# ---------- Model ----------
class BindingPredictor(nn.Module):
    def __init__(self, emb_dim=640, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.classifier(out).squeeze(-1)
        return out


# ---------- Train ----------
def compute_pos_weight(labels):
    total_pos = sum(mask.sum() for mask in labels)
    total_count = sum(len(mask) for mask in labels)
    pos_weight = (total_count - total_pos) / total_pos
    return torch.tensor([pos_weight], dtype=torch.float32)


def train_model(model, dataloader, n_epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = compute_pos_weight(dataloader.dataset.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for x, y, mask in dataloader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            loss = (criterion(pred, y) * mask).sum() / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")


# ---------- Evaluate ----------
def topk_predictions(pred_probs, k=5):
    pred = np.zeros_like(pred_probs)
    topk_idx = np.argsort(pred_probs)[-k:]
    pred[topk_idx] = 1
    return pred


def evaluate_model(model, dataloader, top_k=5, save_dir="results"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    per_peptide_f1s = []
    per_peptide_matches = []

    with torch.no_grad():
        for x, y, mask in dataloader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            preds = model(x)
            for p, t, m in zip(preds, y, mask):
                probs = p[m].cpu().numpy()
                true = t[m].cpu().numpy().astype(int)
                pred_bin = topk_predictions(probs, k=top_k)

                # Log for global stats
                all_preds.extend(pred_bin)
                all_labels.extend(true)

                # Per peptide
                correct = (pred_bin == true).sum()
                per_peptide_matches.append({
                    "correct_predictions": int((pred_bin.astype(bool) & true.astype(bool)).sum()),
                    "total_true": int(true.sum()),
                    "peptide_length": len(true)
                })

                if np.sum(true) > 0:
                    precision = np.sum((pred_bin == 1) & (true == 1)) / np.sum(pred_bin == 1) if np.sum(pred_bin == 1) else 0
                    recall = np.sum((pred_bin == 1) & (true == 1)) / np.sum(true == 1)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    per_peptide_f1s.append(f1)

    print("Evaluation Report (Top-5 forced predictions per peptide):")
    print(classification_report(all_labels, all_preds, digits=4))

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary")

    mean_f1_per_peptide = np.mean(per_peptide_f1s) if per_peptide_f1s else 0

    print(f"Mean F1 (per-peptide): {mean_f1_per_peptide:.4f}")

    # --- Per-Peptide Histogram ---
    correct_counts = [entry["correct_predictions"] for entry in per_peptide_matches]
    plt.figure(figsize=(6,4))
    plt.hist(correct_counts, bins=[0,1,2,3,4,5,6], align='left', rwidth=0.8, color="skyblue", edgecolor="black")
    plt.xticks(range(6))
    plt.xlabel("Correctly Predicted Binding Residues (out of 5)")
    plt.ylabel("Number of Peptides")
    plt.title("Per-Peptide Prediction Accuracy")
    plt.tight_layout()
    hist_path = os.path.join(save_dir, "per_peptide_accuracy_histogram.png")
    plt.savefig(hist_path)
    print(f"Saved histogram to {hist_path}")

    # --- CSV Table ---
    df = pd.DataFrame(per_peptide_matches)
    csv_path = os.path.join(save_dir, "per_peptide_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved per-peptide analysis to {csv_path}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_f1_per_peptide": mean_f1_per_peptide,
        "support": len(all_labels)
    }


# ---------- Save ----------
def save_model(model, path="residue_classifier.pt"):
    torch.save(model.state_dict(), path)


# ---------- Main Pipeline ----------
def run_pipeline_from_pickle(data_path="peptide_data.pkl",
                             embedding_path="peptide_embeddings.pkl",
                             model_path="residue_classifier.pt",
                             embedding_size=640,
                             esm_layer=6,
                             test_size=0.2,
                             n_epochs=10,
                             batch_size=10):
    # Load raw data
    print("Loading peptide data...")
    with open(data_path, "rb") as f:
        seqs, labels = pickle.load(f)

    # Check if embedding already exists
    if os.path.exists(embedding_path):
        print("Loading precomputed ESM embeddings...")
        with open(embedding_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print("Computing ESM embeddings...")
        esm_model, alphabet, batch_converter, device = load_esm_model(
            embedding_size)
        embeddings = get_esm_embeddings(seqs, esm_model, alphabet,
                                        batch_converter, device,
                                        layer=esm_layer)
        with open(embedding_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Saved embeddings to {embedding_path}")

    # Split train/test
    train_X, test_X, train_y, test_y = train_test_split(embeddings, labels,
                                                        test_size=test_size,
                                                        random_state=42)

    train_dataset = PepResidueDataset(train_X, train_y)
    test_dataset = PepResidueDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                             collate_fn=pad_collate)

    # Train
    model = BindingPredictor(emb_dim=embedding_size)
    print("Training model...")
    train_model(model, train_loader, n_epochs)

    # Save
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    # Evaluate
    print("Evaluating on test set...")
    return evaluate_model(model, test_loader)


def main(data_path="peptide_data.pkl",
         embedding_path="peptide_embeddings.pkl",
         model_path="residue_classifier.pt",
         embedding_size=640,
         esm_layer=6,
         test_size=0.2,
         n_epochs=10,
         batch_size=8):
    # Run the full training and evaluation pipeline
    run_pipeline_from_pickle(
        data_path=data_path,
        embedding_path=embedding_path,
        model_path=model_path,
        embedding_size=embedding_size,
        esm_layer=esm_layer,
        test_size=test_size,
        n_epochs=n_epochs,
        batch_size=batch_size

    )





if __name__ == "__main__":
    os.makedirs(os.path.dirname("results/results.csv"), exist_ok=True)

    with open("results/results.csv", "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames= ["precision", "recall", "f1",
                                            "support", "mean_f1_per_peptide"])
        writer.writeheader()

        metrics = run_pipeline_from_pickle(
            data_path="data/peptide_data.pkl",
            embedding_path=f"saved_models/emb_640.pkl",
            model_path=f"saved_models/model_emb_best_weight.pt",
            embedding_size=640,
            esm_layer=6,
            test_size=0.2,
            n_epochs=10,
            batch_size=16
        )
        result_row = {**metrics}
        writer.writerow(result_row)
        print(f"Logged results: {result_row}")

