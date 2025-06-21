#!/usr/bin/env python3
import os, sys

import numpy as np
import torch
import tkinter as tk
from tkinter import scrolledtext, messagebox

# ─── allow import from src/ ──────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # …/Hackaton/src
sys.path.insert(0, THIS_DIR)

# ─── pull in your binding predictor + ESM helpers ──────────
from binding_predictor_esm_weight import (
    BindingPredictor,
    load_esm_model,
    get_esm_embeddings,
)

# ─── find the right .pt under saved_models ──────────────────
PROJECT_ROOT = os.path.dirname(THIS_DIR)                   # …/Hackaton
MODEL_PATH   = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "model_emb_best_weight.pt",
)

class SequencePredictorApp:
    def __init__(self, root):
        self.root = root
        root.title("NES–CRM1 Contact Predictor")

        # 1) load ESM backbone (once)
        self.esm_model, self.alphabet, self.batch_converter, self.device = \
            load_esm_model(embedding_size=640)

        # 2) instantiate & load your trained classifier
        self.classifier = self._load_classifier(MODEL_PATH)

        # ─── build the simple UI ────────────────────────────────
        tk.Label(root, text="Enter amino-acid sequence:").pack(padx=10, pady=(10, 0))
        self.seq_var = tk.StringVar()
        tk.Entry(root, textvariable=self.seq_var, width=60).pack(padx=10, pady=5)
        tk.Button(root, text="Predict", command=self.on_predict).pack(padx=10, pady=5)
        tk.Label(root, text="Per-residue contact probability:").pack(padx=10, pady=(10, 0))

        self.output = scrolledtext.ScrolledText(root, width=70, height=15, state="disabled")
        self.output.pack(padx=10, pady=5)

    def _load_classifier(self, path):
        model = BindingPredictor(emb_dim=640, hidden_dim=128)
        state = torch.load(path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def on_predict(self):
        seq = self.seq_var.get().strip().upper()
        if not seq:
            messagebox.showwarning("Input Error", "Please enter a sequence.")
            return

        try:
            probs = self._predict(seq)  # → np.ndarray shape (L,)
            text = "\n".join(
                f"Pos {i+1} ({aa}): {p:.3f}"
                for i, (aa, p) in enumerate(zip(seq, probs))
            )
            self._set_output(text)
        except Exception as e:
            self._set_output(f"Prediction error:\n{e}")
    def topk_predictions(self, pred_probs, k=5):
        pred = np.zeros_like(pred_probs)
        topk_idx = np.argsort(pred_probs)[-k:]
        pred[topk_idx] = 1
        return pred
    def _predict(self, seq: str):
        """
        Discrete prediction (0/1) per residue with adaptive thresholding.
        Ensures at least 3 residues are predicted as binding.
        """
        # 1) Get ESM embedding
        emb_list = get_esm_embeddings(
            [seq],
            self.esm_model,
            self.alphabet,
            self.batch_converter,
            self.device,
            layer=6,
        )
        emb = torch.tensor(emb_list[0], dtype=torch.float32).unsqueeze(0).to(self.device)

        # 2) Forward pass through model (LSTM + classifier)
        with torch.no_grad():
            logits = self.classifier(emb)
            probs = torch.sigmoid(logits)[0, :].cpu().numpy()

        # 3) Apply top-k thresholding
        pred_mask = self.topk_predictions(probs, k=5)

        return pred_mask  # shape (L,), values 0 or 1

    def _set_output(self, text: str):
        self.output.config(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    SequencePredictorApp(root)
    root.mainloop()
