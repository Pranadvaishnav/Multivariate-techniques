# Variational Autoencoder for Credit Card Fraud Detection

## 📌 Overview
This project implements a Variational Autoencoder (VAE) for anomaly detection on the Credit Card Fraud dataset.

The model is trained only on normal transactions and detects fraud based on weighted reconstruction error.

---

## 🧠 Methodology

1. Train VAE on normal samples only
2. Compute weighted reconstruction error
3. Select optimal threshold using F1-score
4. Evaluate using ROC-AUC and Precision-Recall

---

## ⚙️ Model Architecture

- Encoder:
  - PCA features (28 dims) → 128 → 64
  - Raw features (2 dims) → 16
- Latent dimension: 16
- Decoder:
  - Latent → 64 → Input dimension

Loss:
- Reconstruction Loss (weighted MSE)
- KL Divergence (β-VAE with β=0.1)

---

## 📊 Dataset

Hugging Face:
David-Egea/Creditcard-fraud-detection

- Highly imbalanced dataset
- Fraud ≈ 0.17%
