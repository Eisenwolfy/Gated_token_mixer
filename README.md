# TabMixer
> A gated token-mixing neural architecture for tabular data.

TabMixer is a deep learning architecture designed for tabular classification tasks. Instead of relying on self-attention, it combines token mixing, channel mixing, learnable numerical embeddings and adaptive feature gating to efficiently model interactions between numerical features.

The project also includes a comparison against:

- Scratch NumPy MLP
- Standard PyTorch MLP
- TabMixer (proposed architecture)

---

# Motivation

Tabular data remains one of the hardest domains for deep learning.

Classical neural networks often struggle to model feature interactions efficiently, while transformer-based architectures can become computationally expensive.

The goal of this project was to investigate whether a lightweight mixer-based architecture with adaptive feature gating can achieve competitive performance while remaining computationally efficient.

---

# Architecture

```
Input Features
      │
      ▼
Numerical Embedding
(sin/cos periodic embedding
 + learnable positional embedding)
      │
      ▼
─────────────────────────────
Mixer Block × N
─────────────────────────────

LayerNorm
      │
Token Mixer
      │
Feature Gate
      │
Highway Residual

      ↓

LayerNorm
      │
Channel Mixer
      │
Feature Gate
      │
Highway Residual

─────────────────

      ▼
  LayerNorm
      ▼
   Pooling
      ▼
Classification Head
```

---

# Main Components

## Numerical Embedding

Each numerical feature is converted into a high-dimensional representation using

- learnable frequencies
- learnable phases
- sinusoidal encoding
- learnable positional embeddings

Unlike standard linear embeddings, this representation allows the network to model periodic and non-linear relationships.

---

## Token Mixer

Instead of self-attention, TabMixer mixes information across features using lightweight MLP layers.

Advantages:

- fewer parameters
- lower computational cost
- efficient feature interaction

---

## Channel Mixer

Processes each feature representation independently in the embedding dimension.

---

## Feature Gate

One of the key contributions of this work.

Each feature receives an adaptive gate computed from

- local feature representation
- global feature context

```
Gate = σ(Local + Context)
```

The gate controls how much new information should replace the previous representation.

---

## Highway Residual Update

Instead of the standard residual connection

```
x + F(x)
```

TabMixer performs

```
x + gate · (new − old)
```

This allows the network to dynamically regulate feature updates.

During training, stochastic residual dropping is applied to improve regularization.

---

# Experiments

Datasets:

- Breast Cancer Wisconsin
- Adult Income
- California Housing

---

# Baselines

The proposed architecture is compared against

- Scratch NumPy MLP
- Standard PyTorch MLP

Future comparisons:

- XGBoost
- LightGBM
- CatBoost
- FT-Transformer
- TabNet

---

# Results

| Model | Accuracy |
|--------|----------|
| NumPy MLP | XX.XX% |
| PyTorch MLP | XX.XX% |
| TabMixer | XX.XX% |

---

# Technologies

- Python
- PyTorch
- NumPy
- scikit-learn
- matplotlib

---

# Repository Structure

```
.
├── models/
│   ├── tabmixer.py
│   ├── mlp_numpy.py
│   └── mlp_pytorch.py
│
├── experiments/
│
├── datasets/
│
├── plots/
│
├── train.py
│
└── README.md
```

---


#

Developed as an independent deep learning research project exploring efficient neural architectures for tabular data.
