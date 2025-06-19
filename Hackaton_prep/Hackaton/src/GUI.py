#!/usr/bin/env python3
import os, sys
import torch
import tkinter as tk
from tkinter import scrolledtext, messagebox

# ─── allow import from src/ ──────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # …/Hackaton/src
sys.path.insert(0, THIS_DIR)

# ─── pull in your binding predictor + ESM helpers ──────────
from binding_predictor_esm_extended import (
    BindingPredictor,
    load_esm_model,
    get_esm_embeddings,
)

# ─── find the right .pt under saved_models ──────────────────
PROJECT_ROOT = os.path.dirname(THIS_DIR)                   # …/Hackaton
MODEL_PATH   = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "model_emb640_ep50_bs8.pt",  # ← adjust if you pick a different file
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
        # must exactly match what you trained!
        model = BindingPredictor(
            emb_dim=640,
            hidden_dim=128,
            classifier_hidden_dim=64,  # unused when use_mlp=False
            use_mlp=False,             # ← this matches a single Linear(256→1)
            use_dropout=False,         # dropout layer isn’t used either
            dropout_rate=0.0,
            pos_weight=None,
        )

        # load the actual weights you saved
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
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
            self._set_output(f"❌ Prediction error:\n{e}")

    def _predict(self, seq: str):
        """
        Given an amino-acid sequence, returns a discrete 0/1 prediction
        for each residue (shape: L,).
        """
        # 1) Compute ESM embeddings for this one sequence
        emb_list = get_esm_embeddings(
            [seq],
            self.esm_model,
            self.alphabet,
            self.batch_converter,
            self.device,
            layer=6,  # change if you used a different layer
        )
        # emb_list[0] is a numpy array of shape (L, emb_dim)
        emb_tensor = torch.tensor(emb_list[0], dtype=torch.float32).unsqueeze(0)  # (1, L, emb_dim)

        # 2) Forward pass, sigmoid, then threshold at 0.5 → discrete prediction
        with torch.no_grad():
            logits = self.classifier(emb_tensor)  # tensor shape (1, L)
            probs = torch.sigmoid(logits)  # tensor shape (1, L)
            preds = (probs > 0.5)[0].cpu().numpy().astype(int)  # ndarray shape (L,)

        return preds

    def _set_output(self, text: str):
        self.output.config(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    SequencePredictorApp(root)
    root.mainloop()
