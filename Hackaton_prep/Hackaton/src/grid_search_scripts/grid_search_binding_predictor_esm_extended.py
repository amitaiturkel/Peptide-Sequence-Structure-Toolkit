import torch
import torch.nn as nn
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
def get_esm_embeddings(sequences: List[str], model, alphabet, batch_converter, device, layer=6):
    batch = [(f"pep{i}", seq) for i, seq in enumerate(sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer])
    token_reps = results["representations"][layer]

    embeddings = []
    for i, seq in enumerate(sequences):
        emb = token_reps[i, 1:len(seq)+1].cpu().numpy()
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
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# ---------- Collate Function ----------
def pad_collate(batch):
    seqs, labels = zip(*batch)
    lens = [len(x) for x in seqs]
    padded_seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    mask = torch.arange(padded_seqs.shape[1])[None, :] < torch.tensor(lens)[:, None]
    return padded_seqs, padded_labels, mask


class BindingPredictor(nn.Module):
    def __init__(self,
                 emb_dim=640,
                 hidden_dim=128,
                 classifier_hidden_dim=128,
                 use_mlp=True,
                 use_dropout=True,
                 dropout_rate=0.5,
                 pos_weight =None):
        super().__init__()
        self.pos_weight = pos_weight

        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)

        classifier_layers = []

        if use_mlp:
            classifier_layers.append(nn.Linear(hidden_dim * 2, classifier_hidden_dim))
            classifier_layers.append(nn.ReLU())
            if use_dropout:
                classifier_layers.append(nn.Dropout(p=dropout_rate))
            classifier_layers.append(nn.Linear(classifier_hidden_dim, 1))
        else:
            classifier_layers.append(nn.Linear(hidden_dim * 2, 1))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.classifier(out).squeeze(-1)
        return out


def calculate_pos_weight(labels):
    all_labels = np.concatenate(labels)
    n_pos = (all_labels == 1).sum()
    n_neg = (all_labels == 0).sum()
    return n_neg / (n_pos + 1e-6)  # add epsilon to avoid division by zero


# ---------- Train ----------
def train_model(model, dataloader, n_epochs=10, lr=1e-3, loss_log_path = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight_tensor = torch.tensor([model.pos_weight or 1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_tensor)

    losses = []
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
            epoch_loss = total_loss
            losses.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
    if loss_log_path:
        os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)
        with open(loss_log_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["epoch", "loss"])
            writer.writeheader()
            writer.writerows(losses)

# ---------- Evaluate ----------
def evaluate_model(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y, mask in dataloader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            preds = model(x)
            for p, t, m in zip(preds, y, mask):
                all_preds.extend(p[m].cpu().numpy() >= 0.5)
                all_labels.extend(t[m].cpu().numpy())
    print("Evaluation Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(all_labels)
    }

# ---------- Save ----------
def save_model(model, path="residue_classifier.pt"):
    torch.save(model.state_dict(), path)

# ---------- Main Pipeline ----------
def run_pipeline_from_pickle(data_path="peptide_data.pkl",
                             embedding_path="peptide_embeddings.pkl",
                             model_path="residue_classifier.pt",
                             hidden_dim=128,
                             classifier_hidden_dim=128,
                             use_mlp=True,
                             use_dropout=True,
                             dropout_rate=0.5,
                             embedding_size=640,
                             esm_layer=6,
                             test_size=0.2,
                             n_epochs=10,
                             batch_size=10,
                             add_weights = False,
                             lr=1e-3,
                             suffix = ''):


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
        esm_model, alphabet, batch_converter, device = load_esm_model(embedding_size)
        embeddings = get_esm_embeddings(seqs, esm_model, alphabet, batch_converter, device, layer=esm_layer)
        with open(embedding_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Saved embeddings to {embedding_path}")

    # Split train/test
    train_X, test_X, train_y, test_y = train_test_split(embeddings, labels, test_size=test_size, random_state=42)

    train_dataset = PepResidueDataset(train_X, train_y)
    test_dataset = PepResidueDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=pad_collate)

    # Train
    pos_weight = None
    if add_weights:
        pos_weight = calculate_pos_weight(train_y)
    model = BindingPredictor(
        emb_dim=embedding_size,
        hidden_dim=hidden_dim,
        classifier_hidden_dim=classifier_hidden_dim,
        use_mlp=use_mlp,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        pos_weight = pos_weight
    )

    print("Training model...")


    train_model(model, train_loader, n_epochs=n_epochs, lr=lr,loss_log_path =f"results/reports/{suffix}.csv")


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
         n_epochs = 10,
         batch_size =8):


    # Run the full training and evaluation pipeline
    run_pipeline_from_pickle(
        data_path=data_path,
        embedding_path=embedding_path,
        model_path=model_path,
        embedding_size=embedding_size,
        esm_layer=esm_layer,
        test_size=test_size,
        n_epochs = n_epochs,
        batch_size =batch_size

    )

def format_params_for_filename(params: dict) -> str:
    parts = []
    for k, v in params.items():
        if isinstance(v, bool):
            v = int(v)  # True → 1, False → 0
        elif isinstance(v, float):
            v = f"{v:.3g}".replace('.', 'p')  # 0.001 → 0p001
        parts.append(f"{k}_{v}")
    return "_".join(parts)


def grid_search(param_grid, csv_path="results.csv"):
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))

    # Ensure directory exists
    output_dir = os.path.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Define result file columns
    result_keys = keys + ["precision", "recall", "f1", "support"]

    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result_keys)
        writer.writeheader()

        for combo in combinations:
            params = dict(zip(keys, combo))
            print(f"\nRunning experiment with: {params}")

            suffix = format_params_for_filename(params)
            model_path = f"saved_models/model_{suffix}.pt"
            embedding_path = f"saved_models/emb_{params['embedding_size']}.pkl"

            try:
                metrics = run_pipeline_from_pickle(
                    data_path="data/peptide_data.pkl",
                    embedding_path=embedding_path,
                    model_path=model_path,
                    embedding_size=params["embedding_size"],
                    esm_layer=params["esm_layer"],
                    test_size=params["test_size"],
                    n_epochs=params["n_epochs"],
                    lr=params["lr"],
                    batch_size=params["batch_size"],
                    hidden_dim=params["hidden_dim"],
                    classifier_hidden_dim=params["classifier_hidden_dim"],
                    use_mlp=params["use_mlp"],
                    use_dropout=params["use_dropout"],
                    dropout_rate=params["dropout_rate"],
                    add_weights = params["add_weights"],
                    suffix = suffix
                )
                result_row = {**params, **metrics}
                writer.writerow(result_row)
                print(f"Logged results: {result_row}")
            except Exception as e:
                print(f"Failed with params {params}: {e}")

if __name__ == "__main__":
    param_grid = {
        "embedding_size": [640, 1280],
        "esm_layer": [6,8 ,20],
        "n_epochs": [ 10,20,50],
        "batch_size": [8,16],
        "test_size": [0.2],
        "hidden_dim": [128, 256],
        "classifier_hidden_dim": [64, 128],
        "use_mlp": [True,False],
        "lr": [1e-3],
        "use_dropout": [True, False],
        "dropout_rate": [0.15,0.3, 0.5],
        "add_weights": [True, False],
    }
    grid_search(param_grid, csv_path="results/results_ext.csv")
