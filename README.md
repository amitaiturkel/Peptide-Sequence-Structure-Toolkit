# 🧠 Peptide Binding Prediction – Hackathon Project

Welcome to the repository for our Hackathon preparation and final submission!

---

## 📁 Repository Structure

This repo contains:

- `ex4/` – **Exercise 4**, which served as preparation for the hackathon.  
  It includes experiments with protein language models (ESM), training classifiers on residue-level labels, and tuning hyperparameters.

- `hackathon/` – The **Hackathon project** itself.  
  This folder implements our model and experimental framework based on a research paper (detailed below). It includes:
  - Model architecture
  - ESM embedding pipeline
  - Training scripts
  - Grid search
  - Inference and evaluation tools

---

## 📄 Hackathon Paper

The Hackathon project is an implementation of the model and approach described in the following paper:

🔗 [NES–CRM1 Complex Binding Site Classification](https://claude.ai/public/artifacts/001c25d1-4531-4332-8bb8-46935eb7bbc9)

Our goal was to reproduce the idea and extend the pipeline to classify peptide residues as binding or non-binding, using a combination of ESM embeddings and a bidirectional LSTM-based classifier.

---


