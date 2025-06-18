"""
the raw model I made (yedaaya). still need to compare it to the rest of the models
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np

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

# 3. Example usage
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
print("a")
sequences = ["MSEQLLEK", "MLKQENSTV", "MGDV"]
labels = [
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0]
]

dataset = AminoAcidDataset(sequences, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

model = NESClassifier()

# 4. Example forward pass
for input_ids, attention_mask, targets in dataloader:
    outputs = model(input_ids, attention_mask)
    print("Predicted probabilities:", outputs)
    print("Target labels:", targets)
    break
