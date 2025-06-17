"""
model.py

Defines a simple residue-level binary classifier for peptide binding prediction.
"""

import torch
import torch.nn as nn


class ResidueBindingModel(nn.Module):
    def __init__(self, embedding_dim=1280, hidden_dim=256, num_layers=1, dropout=0.3):
        super().__init__()
        # BiLSTM to capture sequential context
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)  # binary output per residue

    def forward(self, x):
        """
        x: tensor (batch_size, seq_len, embedding_dim)
        returns: logits (batch_size, seq_len)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 2*hidden_dim)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out).squeeze(-1)
        return logits
