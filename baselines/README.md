# Baseline Models

This directory contains configuration, wrapper scripts, and raw outputs for all four baseline generative models evaluated in the paper.

## Model Overview

| Model | Architecture | Version | Reference |
|-------|-------------|---------|-----------|
| **BIMODAL** | RNN-based autoregressive (bidirectional LSTM) | v1.0 | Grisoni et al., *J. Chem. Inf. Model.* 2020 |
| **QADD** | RL-guided property optimization (DQN + SMILES) | v1.0 | Zhou et al., *ACS Omega* 2019 |
| **SMILES-Transformer** | Sequence-to-sequence attention (Encoder–Decoder) | v1.0 | Honda et al., *arXiv:1911.04738* 2019 |
| **Diffusion** | Denoising diffusion probabilistic model (ligand-only) | v1.0 | Ho et al., *NeurIPS* 2020; adapted for SMILES |

## Installation

Each baseline has its own dependencies. Install from the respective repositories:

```bash
# BIMODAL
git clone https://github.com/ETHmodlab/BIMODAL.git
cd BIMODAL && pip install -e .

# QADD
git clone https://github.com/yulun-rayn/QADD.git
cd QADD && pip install -e .

# SMILES-Transformer
git clone https://github.com/DSPsleern/smiles-transformer.git
cd smiles-transformer && pip install -e .

# Diffusion (ligand-only DDPM)
pip install diffusers  # or use custom implementation in deep_learning/models/
```

## Running Baselines

All baselines are configured to generate **10,000 molecules** with `seed=42` for fair comparison.

### BIMODAL

```bash
python baselines/run_bimodal.py \
  --model_path baselines/checkpoints/bimodal_pretrained.pt \
  --n_molecules 10000 \
  --temperature 1.0 \
  --seed 42 \
  --output baselines/outputs/bimodal_10k.smi
```

**Key hyperparameters:**
- Hidden size: 512, Layers: 3 (bidirectional LSTM)
- Vocabulary: SMILES character-level tokenizer
- Sampling temperature: 1.0
- Pre-trained on ChEMBL 29

### QADD

```bash
python baselines/run_qadd.py \
  --model_path baselines/checkpoints/qadd_pretrained.pt \
  --n_molecules 10000 \
  --property_target docking_score \
  --seed 42 \
  --output baselines/outputs/qadd_10k.smi
```

**Key hyperparameters:**
- DQN hidden: 256, Replay buffer: 10,000
- Reward: composite (QED + SA + Vina proxy)
- Exploration: ε-greedy (ε = 0.1)
- Pre-trained policy on ZINC 250K

### SMILES-Transformer

```bash
python baselines/run_smiles_transformer.py \
  --model_path baselines/checkpoints/smiles_transformer_pretrained.pt \
  --n_molecules 10000 \
  --max_length 128 \
  --seed 42 \
  --output baselines/outputs/smiles_transformer_10k.smi
```

**Key hyperparameters:**
- Encoder: 6 layers, 8 heads, d_model = 256
- Decoder: 6 layers, 8 heads, d_model = 256
- Vocabulary: SMILES token-level (regex-based)
- Max sequence length: 128
- Pre-trained on ZINC 250K, fine-tuned on PDBbind ligands

### Diffusion

```bash
python baselines/run_diffusion.py \
  --model_path baselines/checkpoints/diffusion_pretrained.pt \
  --n_molecules 10000 \
  --diffusion_steps 1000 \
  --seed 42 \
  --output baselines/outputs/diffusion_10k.smi
```

**Key hyperparameters:**
- UNet backbone: 4 ResBlocks, channels = [128, 256, 512]
- Noise schedule: linear (β_1 = 1e-4, β_T = 0.02, T = 1000)
- Ligand-only (no pocket conditioning)
- Trained on ZINC 250K SMILES latent space

## Post-processing

All generated SMILES undergo the same unified post-processing pipeline:

```bash
python Evaluation/scripts/postprocess.py \
  --input baselines/outputs/*.smi \
  --output baselines/outputs/processed/ \
  --sanitize \
  --remove_duplicates \
  --canonical
```

Steps:
1. RDKit sanitization and canonicalization
2. Valency check and invalid molecule removal
3. Duplicate removal (canonical SMILES comparison)

## Output Format

Each `.smi` file contains one SMILES string per line:

```
CC(=O)Nc1ccc(O)cc1
c1ccc2c(c1)cc1ccccc12
CC(C)NCC(O)c1ccc(O)c(O)c1
...
```

## Evaluation

After generation, all models are evaluated with the same metrics:

```bash
python Evaluation/scripts/evaluate_all.py \
  --input_dir baselines/outputs/ \
  --pocket_pdb data/1p65.pdb \
  --scorer_checkpoint logs/egnn_scorer/best_model.pth \
  --output results/table1_comparison.csv
```

Metrics computed:
- **Optimal Binding Energy**: Best AutoDock Vina score (kcal/mol)
- **QED**: Quantitative Estimate of Drug-likeness (Bickerton et al.)
- **Novelty**: Fraction of generated SMILES not in training set
- **Validity**: Fraction passing RDKit sanitization
- **Uniqueness**: Fraction of unique canonical SMILES
- **Internal Diversity**: Average pairwise Tanimoto distance (ECFP4)

## Checkpoints

Model checkpoints are available from [GitHub Releases](https://github.com/hjd20030114-blip/MolFoundry/releases/tag/v1.0-checkpoints):

| File | Size | MD5 |
|------|------|-----|
| `bimodal_pretrained.pt` | ~50 MB | *(to be added)* |
| `qadd_pretrained.pt` | ~30 MB | *(to be added)* |
| `smiles_transformer_pretrained.pt` | ~80 MB | *(to be added)* |
| `diffusion_pretrained.pt` | ~120 MB | *(to be added)* |
| `egnn_scorer_best.pth` | ~45 MB | *(to be added)* |

