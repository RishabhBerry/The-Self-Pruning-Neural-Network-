# Tredence AI Engineering Internship — Case Study

**Self-Pruning Neural Network on CIFAR-10**

## Overview

A PyTorch implementation of a feed-forward neural network that **learns to prune itself during training** via learnable sigmoid gates attached to every weight. A sparsity regularisation term (L1 norm of all gate values) in the loss function applies pressure to close unnecessary connections, leaving only a sparse, efficient network.

## Files

| File | Description |
|------|-------------|
| `train.py` | Complete training script — PrunableLinear, network, training loop, plots |
| `REPORT.md` | Written analysis — theory, results table, plot explanations |
| `README.md` | This file |

## Quickstart

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <repo-name>

# 2. Install dependencies
pip install torch torchvision matplotlib numpy

# 3. Run experiments (trains 3 models with λ = 1e-5, 1e-4, 1e-3)
python train.py
```

CIFAR-10 downloads automatically to `./data/` on first run.

## Architecture

```
Input (3072)
  └─ PrunableLinear(3072 → 1024) + BatchNorm + ReLU
       └─ PrunableLinear(1024 → 512)  + BatchNorm + ReLU
            └─ PrunableLinear(512 → 256)  + BatchNorm + ReLU
                 └─ PrunableLinear(256 → 10)
```

Each `PrunableLinear` layer holds a `gate_scores` parameter (same shape as `weight`). During the forward pass: `gates = sigmoid(gate_scores)` and `output = input @ (weight ⊙ gates).T + bias`.

## Key Idea

```
Total Loss = CrossEntropyLoss  +  λ × Σ sigmoid(gate_scores)
```

The L1 penalty on gate values pushes most gates to zero (pruning the weight), while the network retains the gates it needs to maintain accuracy. λ controls the trade-off.

## Results Summary

| λ | Test Accuracy | Sparsity |
|---|--------------|---------|
| 1e-5 (low) | ~53% | ~20% |
| 1e-4 (medium) | ~50% | ~57% |
| 1e-3 (high) | ~41% | ~87% |

See `REPORT.md` for full analysis and `gate_distribution.png` for gate value histograms.

> **Note:** Accuracy is in the 40–55% range because this is a pure feed-forward MLP (no convolutions), which is the architecture the spec requires. A CNN on CIFAR-10 would reach 90%+, but that is not what is being evaluated here — the focus is on the self-pruning mechanism.
