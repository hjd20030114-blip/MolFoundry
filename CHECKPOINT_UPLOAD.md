# Model Checkpoint Upload Guide

## Checkpoints to Upload

Upload the following model checkpoints to [GitHub Releases](https://github.com/hjd20030114-blip/MolFoundry/releases) as `v1.0-checkpoints`:

| File | Description | Expected Size |
|------|-------------|:---:|
| `phase1_equivariant_gnn.pth` | EGNN affinity scorer (best fold) | ~45 MB |
| `egnn_scorer_5fold.tar.gz` | All 5-fold EGNN checkpoints | ~200 MB |
| `generator_transformer.pth` | Pocket-conditioned generator | ~120 MB |
| `bimodal_pretrained.pt` | BIMODAL baseline checkpoint | ~50 MB |
| `qadd_pretrained.pt` | QADD baseline checkpoint | ~30 MB |
| `smiles_transformer_pretrained.pt` | SMILES-Transformer checkpoint | ~80 MB |
| `diffusion_pretrained.pt` | Diffusion baseline checkpoint | ~120 MB |

## Upload Steps

### Option 1: GitHub Web UI

1. Go to **Releases** → **Draft a new release**
2. Tag: `v1.0-checkpoints`, Title: `Model Checkpoints v1.0`
3. Drag-and-drop checkpoint files into the upload area
4. Publish release

### Option 2: GitHub CLI

```bash
# Install gh CLI if needed: brew install gh

# Create release
gh release create v1.0-checkpoints \
  --title "Model Checkpoints v1.0" \
  --notes "Pre-trained model weights for MolFoundry and baseline models.

## Contents
- EGNN affinity scorer (5-fold CV, best validation RMSE)
- Pocket-conditioned cross-attention generator
- Baseline model checkpoints (BIMODAL, QADD, SMILES-Transformer, Diffusion)

## Usage
Download and place in \`logs/\` directory. See REPRODUCE.md for details."

# Upload files
gh release upload v1.0-checkpoints \
  deep_learning_results/phase1_equivariant_gnn.pth \
  logs/best_model.pth \
  baselines/checkpoints/*.pt
```

## Existing Checkpoints in Repository

```
HJD/deep_learning_results/phase1_equivariant_gnn.pth    ← EGNN scorer (80 epochs)
HJD/deep_learning_results_20epoch/phase1_equivariant_gnn.pth  ← EGNN scorer (20 epochs)
HJD/logs/best_model.pth                                  ← Best training checkpoint
HJD/logs/final_model.pth                                 ← Final epoch checkpoint
```

## Verification

After uploading, verify downloads work:

```bash
# Download and verify
wget https://github.com/hjd20030114-blip/MolFoundry/releases/download/v1.0-checkpoints/phase1_equivariant_gnn.pth
python -c "
import torch
ckpt = torch.load('phase1_equivariant_gnn.pth', map_location='cpu')
print(f'Keys: {list(ckpt.keys())}')
print(f'Loaded successfully')
"
```

