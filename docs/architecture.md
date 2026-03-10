# MolFoundry Architecture Documentation

## System Overview

MolFoundry is a three-module pipeline for structure-aware de novo molecular generation:

```
Module 1: SE(3)-Equivariant Scorer → Module 2: Pocket-Conditioned Generator → Module 3: Multi-Objective Optimizer
```

---

## Module 1: SE(3)-Equivariant Pocket–Ligand Affinity Scorer

### Purpose
Predicts binding affinity from 3D protein–ligand geometry and provides pocket embeddings for downstream generation.

### Architecture

```
Input: Joint pocket–ligand graph G = (V_pocket ∪ V_ligand, E)
  │
  ├─ Node features (12-dim): element type, formal charge, hybridization,
  │   aromaticity, degree, num_H, ring membership, SASA, distance to centroid
  │
  ├─ Edge features: RBF-encoded inter-atomic distances (16 kernels)
  │
  ├─ Molecular fingerprint: 2048-bit ECFP4 → Linear(2048, 512)
  │
  ▼
┌─────────────────────────────────────┐
│  EGNN Layer × 6                     │
│  ┌───────────────────────────────┐  │
│  │ Message: m_ij = MLP(h_i, h_j, │  │
│  │          RBF(||x_i - x_j||))  │  │
│  │                               │  │
│  │ Update:  h_i' = h_i +        │  │
│  │          MLP(h_i, Σ m_ij)     │  │
│  │                               │  │
│  │ Coord:   x_i' = x_i +        │  │
│  │          Σ (x_i-x_j)φ(m_ij)  │  │
│  └───────────────────────────────┘  │
│  Hidden size: 128                   │
│  Dropout: 0.1                       │
└─────────────────────────────────────┘
  │
  ├─ Global Mean Pooling (all nodes) → h_global (128-dim)
  │
  ├─ Pocket-only Mean Pooling → e_pocket (128-dim)  ← cached for Module 2
  │
  ▼
┌──────────────────────┐
│ MLP Regression Head  │
│ Linear(128, 64)      │
│ ReLU + Dropout(0.1)  │
│ Linear(64, 1)        │    → predicted pKd (scalar)
└──────────────────────┘
```

### Training Details

| Parameter | Value |
|-----------|-------|
| Training data | PDBbind v2020 (18,412 complexes) |
| Labels | Experimental Ki/Kd/IC50 (converted to pKd) |
| Loss | Mean Squared Error (MSE) |
| Optimizer | AdamW (lr=2×10⁻⁴, weight_decay=1×10⁻⁵) |
| Scheduler | Cosine annealing over 80 epochs |
| Batch size | 32 |
| Validation | 5-fold stratified CV |
| Selection criterion | Best validation RMSE |

### Key Implementation Files

- `deep_learning/models/equivariant_gnn.py` — EGNN model definition
- `deep_learning/data/featurizers.py` — Graph construction and featurization
- `train.py` — Training loop with K-fold CV

---

## Module 2: Pocket-Conditioned Cross-Attention Generator

### Purpose
Generates molecules atom-by-atom, conditioned on the 3D pocket geometry via cross-attention.

### Architecture

```
Input: Pocket embedding e_pocket from Module 1
  │
  ▼
┌──────────────────────────────────────────┐
│  Transformer Encoder (Pocket Tokenizer)  │
│  Layers: 12                              │
│  Attention heads: 8                      │
│  Hidden size: 512                        │
│  Input: e_pocket → sequence of pocket    │
│         tokens {p_1, ..., p_K}           │
└──────────────────────────────────────────┘
  │
  │  Pocket tokens P = {p_1, ..., p_K}
  ▼
┌──────────────────────────────────────────┐
│  Cross-Attention Layers × 4              │
│                                          │
│  Q = ligand token l_t (current step)     │
│  K, V = pocket tokens P                  │
│                                          │
│  Attention(Q,K,V) = softmax(QK^T/√d)V   │
│                                          │
│  l_t' = l_t + CrossAttn(l_t, P)         │
│  l_t'' = l_t' + FFN(l_t')              │
└──────────────────────────────────────────┘
  │
  │  Pocket-aware ligand embeddings
  ▼
┌──────────────────────────────────────────┐
│  Autoregressive Graph Decoder            │
│                                          │
│  At each step t:                         │
│  1. Sample atom type: p(a_t | l_t'')     │
│  2. Sample bond type: p(b_t | l_t'', a_t)│
│  3. Sample attachment: p(pos | context)   │
│                                          │
│  Constraints enforced:                   │
│  - Valency rules                         │
│  - Ring closure validity                 │
│  - Maximum molecule size                 │
└──────────────────────────────────────────┘
  │
  │  2,000 candidate molecules per pocket
  ▼
┌──────────────────────────────────────────┐
│  Preliminary Affinity Filter             │
│  Threshold: predicted affinity < -6.0    │
│  kcal/mol + valency check                │
│  → Screened library for Module 3         │
└──────────────────────────────────────────┘
```

### Dimension Alignment

The EGNN scorer outputs 128-dim pocket embeddings, while the generator operates at 512-dim. A **linear projection layer** (`Linear(128, 512)`) bridges the two:

```
e_pocket (128-dim) → Linear(128, 512) → pocket_tokens (512-dim)
```

Similarly, 2048-bit ECFP4 fingerprints are projected:
```
ECFP4 (2048-dim) → Linear(2048, 512) → fingerprint_embedding (512-dim)
```

### Key Implementation Files

- `deep_learning/models/transformer.py` — Pocket–Ligand Cross-attention Transformer
- `deep_learning_pipeline.py` — End-to-end generation pipeline

---

## Module 3: Multi-Objective Optimizer

### Purpose
Selects optimal molecules from the screened library using Pareto-based ranking and medicinal chemistry constraints.

### Pipeline

```
Screened Library (from Module 2)
  │
  ▼
┌──────────────────────────────────────────┐
│  Feature Annotation                      │
│  For each molecule compute:              │
│  - Predicted affinity (from EGNN scorer)│
│  - QED (drug-likeness)                   │
│  - SA score (synthetic accessibility)    │
│  - MW, logP, HBD, HBA, TPSA             │
└──────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────┐
│  Soft Lipinski Filters                   │
│  250 ≤ MW ≤ 500                          │
│  1 ≤ logP ≤ 3                            │
│  0 ≤ HBD ≤ 5                             │
│  0 ≤ HBA ≤ 10                            │
└──────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────┐
│  Non-Dominated Sorting (Pareto Ranking)  │
│                                          │
│  3 maximization objectives:              │
│  1. Normalized affinity (ã)              │
│  2. Normalized QED (q̃)                  │
│  3. Normalized SA (s̃)                   │
│                                          │
│  Composite score:                        │
│  S = w_aff·ã + w_qed·q̃ + w_sa·s̃       │
│     + w_lip·Lipinski_penalty             │
│                                          │
│  Molecules ranked by Pareto front index, │
│  then by composite score within front    │
└──────────────────────────────────────────┘
  │
  ▼
  Top-ranked candidates for docking validation
```

### Weight Sensitivity

Default weights: `w_aff = 0.4, w_qed = 0.3, w_sa = 0.2, w_lip = 0.1`

Dirichlet sensitivity analysis (200 random weight vectors, α=1.0) shows coefficient of variation (CV) < 15% across all configurations.

### Key Implementation Files

- `deep_learning_pipeline.py` — Optimization loop
- `scripts/weight_sensitivity.py` — Weight sensitivity analysis

---

## Data Flow Summary

```
PDBbind (18,412 complexes)
    │
    ▼
[Module 1] EGNN Scorer ──────────────── 5-fold CV accuracy: 0.869±0.016
    │                                    PDBbind test accuracy: 0.865
    ├── pKd predictions
    └── pocket embeddings (128-dim)
            │
            ▼
[Module 2] Cross-Attention Generator ── 2,000 candidates per pocket
    │       (12L Encoder + 4L CrossAttn)  Affinity filter: < -6.0 kcal/mol
    │
    ▼
[Module 3] Pareto Optimizer ─────────── Top candidates ranked
    │       (Soft Lipinski + NSGA-II)
    │
    ▼
AutoDock Vina Independent Validation ── Best: -9.201 kcal/mol
    │
    ▼
MD Simulation (50 ns, 3 replicas) ──── RMSD < 1.5 Å, ΔG < -10 kcal/mol
```

