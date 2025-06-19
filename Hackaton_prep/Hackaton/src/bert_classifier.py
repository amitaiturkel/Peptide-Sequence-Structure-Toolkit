"""
the raw model I made (yedaaya). still need to compare it to the rest of the models
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import csv
import itertools
from sklearn.metrics import classification_report, precision_recall_fscore_support

import pickle
from Bio import PDB

# 1. Dataset
class AminoAcidDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len=None):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Tokenize עם או בלי הגבלת אורך
        encoded = self.tokenizer(
            seq,
            padding='max_length' if self.max_len else 'longest',
            truncation=bool(self.max_len),
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        seq_len = input_ids.size(0)

        # קיצוץ הלייבלים למקרה שהם ארוכים מדי
        label_trimmed = label[:seq_len]

        # יצירת טנסור לייבלים עם ריפוד
        label_tensor = torch.zeros(seq_len, dtype=torch.float)
        label_tensor[:len(label_trimmed)] = torch.tensor(label_trimmed, dtype=torch.float)

        return input_ids, attention_mask, label_tensor


# 2. Model
class NESClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("Rostlab/prot_bert")
        self.classifier = nn.Linear(1024, 1)  # 1024 = embedding size of ProtBERT

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # freeze BERT during debug/trial
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # shape: (batch, seq_len, 1024)
        logits = self.classifier(x).squeeze(-1)  # shape: (batch, seq_len)
        probs = torch.sigmoid(logits)
        return probs



# ---------- Collate Function ----------
def collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return input_ids, attention_masks, labels


# ---------- Training ----------
def train(model, dataloader, n_epochs=5, lr=1e-4, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCELoss(reduction='none')

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = (criterion(outputs, labels) * attention_mask).sum() / attention_mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")


# ---------- Evaluation ----------

# Modify evaluate to return metrics instead of just print
def evaluate(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            preds = (outputs >= 0.5).float()

            valid_preds = preds[attention_mask == 1].cpu().numpy()
            valid_labels = labels[attention_mask == 1].cpu().numpy()

            all_preds.extend(valid_preds)
            all_labels.extend(valid_labels)

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(all_labels)
    }


# Extract main training and evaluation to a function with params
def train_and_evaluate(data_path, batch_size, n_epochs, max_len, freeze_bert, lr):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found!")

    print("Loading peptide data...")
    with open(data_path, "rb") as f:
        sequences, labels = pickle.load(f)

    print(f"Loaded {len(sequences)} sequences.")

    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    dataset = AminoAcidDataset(sequences, labels, tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NESClassifier().to(device)

    # Optionally freeze BERT
    if freeze_bert:
        for param in model.bert.parameters():
            param.requires_grad = False

    print("Training model...")
    train(model, dataloader, n_epochs=n_epochs,lr = lr, device=device)

    print("Evaluating model...")
    metrics = evaluate(model, dataloader, device=device)

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    freeze_flag = "frozen" if freeze_bert else "unfrozen"
    model_name = f"protbert_ep{n_epochs}_bs{batch_size}_ml{max_len}_lr{lr}_{'frozen' if freeze_bert else 'unfrozen'}.pt"
    model_path = os.path.join("saved_models", model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    return metrics, model_path


def grid_search(param_grid, data_path="data/peptide_data.pkl", csv_path="results/bart_results.csv"):
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))

    # Create CSV file with headers
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys + ["precision", "recall", "f1", "support", "model_path"])
        writer.writeheader()

        for combo in combinations:
            params = dict(zip(keys, combo))
            print(f"\nRunning grid search with parameters: {params}")

            try:
                metrics, model_path = train_and_evaluate(
                    data_path=data_path,
                    batch_size=params["batch_size"],
                    n_epochs=params["n_epochs"],
                    max_len=params["max_len"],
                    freeze_bert=params["freeze_bert"]
                )
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
                continue

            # Log params + metrics + model path
            row = {**params, **metrics, "model_path": model_path}
            writer.writerow(row)
            print(f"Logged results: {row}")


if __name__ == "__main__":
    param_grid = {
        "batch_size": [4, 8,16],
        "n_epochs": [5, 10,20],
        "max_len": [256, 512],
        "freeze_bert": [True, False]
    }

    grid_search(param_grid)