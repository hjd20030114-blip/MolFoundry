# Reproducing MolFoundry Results

This guide provides step-by-step instructions to reproduce Table 1 (model comparison) and Figure 2 (overall performance) from the paper.

## Prerequisites

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/hjd20030114-blip/MolFoundry.git
cd MolFoundry

# Create conda environment (Python 3.10 recommended)
conda create -n molfoundry python=3.10 -y
conda activate molfoundry

# Install dependencies
pip install -r requirements.txt

# Verify critical packages
python -c "import torch; import e3nn; import torch_geometric; print('OK')"
```

### 2. Data Preparation

```bash
# Download PDBbind v2020 (see data/README.md for details)
# After downloading, place files in data/P-L/
python tools/verify_data.py          # Verify data integrity
python tools/data_statistics.py      # Print dataset statistics (~18,412 complexes after filtering)
```

### 3. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 3080 (10 GB) | NVIDIA A100 (40 GB) |
| RAM | 32 GB | 64 GB |
| Disk | 50 GB | 100 GB |
| Training Time | ~12 h (3080) | ~4 h (A100) |

---

## Reproducing Table 1: Model Comparison

### Step 1: Train the EGNN Affinity Scorer (Module 1)

```bash
python train.py \
  --config config/train_config.yaml \
  --model equivariant_gnn \
  --epochs 80 \
  --batch_size 32 \
  --lr 2e-4 \
  --weight_decay 1e-5 \
  --scheduler cosine \
  --seed 42 \
  --kfold 5 \
  --output_dir logs/egnn_scorer
```

Expected output: 5-fold CV accuracy ≈ 0.869 ± 0.016

### Step 2: Generate Molecules with MolFoundry (Module 2 + 3)

```bash
python deep_learning_pipeline.py \
  --mode generate \
  --scorer_checkpoint logs/egnn_scorer/best_model.pth \
  --pocket_pdb data/1p65.pdb \
  --n_candidates 10000 \
  --affinity_threshold -6.0 \
  --optimization pareto \
  --seed 42 \
  --output_dir results/molfoundry_generation
```

### Step 3: Run Baseline Models

```bash
# BIMODAL
python baselines/run_bimodal.py --n_molecules 10000 --seed 42 \
  --output baselines/outputs/bimodal_10k.smi

# QADD
python baselines/run_qadd.py --n_molecules 10000 --seed 42 \
  --output baselines/outputs/qadd_10k.smi

# SMILES-Transformer
python baselines/run_smiles_transformer.py --n_molecules 10000 --seed 42 \
  --output baselines/outputs/smiles_transformer_10k.smi

# Diffusion
python baselines/run_diffusion.py --n_molecules 10000 --seed 42 \
  --output baselines/outputs/diffusion_10k.smi
```

See `baselines/README.md` for model versions, configurations, and installation.

### Step 4: Unified Evaluation (Table 1 Metrics)

```bash
python Evaluation/scripts/evaluate_all.py \
  --molfoundry_smi results/molfoundry_generation/generated.smi \
  --bimodal_smi baselines/outputs/bimodal_10k.smi \
  --qadd_smi baselines/outputs/qadd_10k.smi \
  --transformer_smi baselines/outputs/smiles_transformer_10k.smi \
  --diffusion_smi baselines/outputs/diffusion_10k.smi \
  --pocket_pdb data/1p65.pdb \
  --scorer_checkpoint logs/egnn_scorer/best_model.pth \
  --output results/table1_comparison.csv
```

### Expected Results (Table 1)

| Model | Binding Energy ↓ | QED ↑ | Novelty ↑ | Validity ↑ | Uniq ↑ | IntDiv ↑ |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| **MolFoundry** | **−9.201** | **0.791** | **1.000** | 0.938 | 0.974 | 0.918 |
| BIMODAL | −8.892 | 0.679 | 0.811 | 0.988 | 0.986 | 0.908 |
| QADD | −8.565 | 0.473 | 0.821 | 1.000 | 1.000 | 0.921 |
| SMILES-Transformer | −6.892 | 0.424 | 0.827 | 0.889 | 0.952 | 0.906 |
| Diffusion | −6.421 | 0.430 | 0.781 | 0.857 | 0.976 | 0.898 |

---

## Reproducing Figure 2: Performance Visualizations

### Step 1: Generate All Sub-figures

```bash
# After completing Table 1 reproduction above:
cd /path/to/project
python scripts/fig2_subplots.py \
  --data_dir results/ \
  --output_dir results/figures/
```

This produces 5 independent sub-figures (300 DPI, PDF + PNG):

| File | Content |
|------|---------|
| `fig2a_validation_accuracy.png/pdf` | Validation accuracy violin plots |
| `fig2b_docking_affinity.png/pdf` | Docking score box plots |
| `fig2c_logp_distribution.png/pdf` | LogP density curves |
| `fig2d_hba_distribution.png/pdf` | H-bond acceptor distributions |
| `fig2e_tpsa_distribution.png/pdf` | TPSA distributions |

### Step 2: Statistical Analysis

```bash
python scripts/statistical_analysis.py \
  --results_csv results/table1_comparison.csv \
  --output results/statistics_report.txt
```

Expected: Wilcoxon signed-rank test (exact method, N = 5):
- MolFoundry vs. SMILES-Transformer: p = 0.125, Cohen's d = 1.23
- MolFoundry vs. BIMODAL: p = 0.063, Cohen's d = 5.31
- MolFoundry vs. QADD: p = 0.063, Cohen's d = 2.57
- MolFoundry vs. Diffusion: p = 0.063, Cohen's d = 2.28

---

## Reproducing Table 3: PDBBind Generalization

```bash
python scripts/pdbbind_generalization.py \
  --scorer_checkpoint logs/egnn_scorer/best_model.pth \
  --test_split results/pl_splits_seed42_test.json \
  --output_dir results/generalization/
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: e3nn` | `pip install e3nn>=0.5.0` |
| `torch_geometric` import error | Install matching PyG version for your CUDA: see [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) |
| AutoDock Vina not found | `pip install vina>=1.2.0` or install from [Vina releases](https://github.com/ccsb-scripps/AutoDock-Vina/releases) |
| CUDA OOM during training | Reduce `batch_size` to 16 or 8 |
| Docking results differ slightly | Expected due to Vina stochastic search; set `--exhaustiveness 32` for better reproducibility |

## Random Seeds

All experiments use `seed=42` unless otherwise noted. Due to GPU non-determinism in PyTorch, results may vary by ±0.5% across runs. The reported values represent the mean of 3 independent runs.

