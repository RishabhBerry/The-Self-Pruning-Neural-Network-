"""
Self-Pruning Neural Network on CIFAR-10
========================================
Tredence AI Engineering Internship — Case Study Submission

This script implements a feed-forward neural network that learns to prune
itself during training via learnable sigmoid gates on each weight.

Architecture:
  - Custom PrunableLinear layers (replaces nn.Linear)
  - Gate scores (same shape as weights) learned alongside weights
  - Sparsity regularization via L1 penalty on sigmoid(gate_scores)
  - Trained on CIFAR-10 with three lambda values for trade-off analysis

Author: Rishabh Berry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# ─────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# PART 1: PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that attaches a learnable
    gate_scores tensor (same shape as weight) to each connection.

    Forward pass:
        gates        = sigmoid(gate_scores)          ∈ (0, 1)
        pruned_weight = weight ⊙ gates               (element-wise)
        output        = input @ pruned_weight.T + bias

    The sigmoid ensures gates are always in (0, 1), keeping them
    bounded while remaining differentiable. When gate_scores → -∞,
    sigmoid → 0 and the connection is effectively removed.

    Gradients flow through both `weight` and `gate_scores` because
    all operations (sigmoid, element-wise mul, linear) are differentiable
    and tracked by PyTorch's autograd engine.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias — same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight
        # Initialised near 0 so sigmoid(0) ≈ 0.5 (gates start half-open)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform init for weights (standard practice)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: turn gate_scores into gates in (0, 1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: element-wise multiply weight by gates
        pruned_weight = self.weight * gates

        # Step 3: standard linear operation with pruned weights
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached) for analysis."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of gates = sum of all gate values.
        Since gates ∈ (0,1) after sigmoid, L1 = simple sum.
        This acts as the per-layer contribution to the sparsity penalty.
        """
        return torch.sigmoid(self.gate_scores).sum()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ─────────────────────────────────────────────
# NETWORK DEFINITION
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (32×32×3 = 3072 input dims, 10 classes).
    All linear layers use PrunableLinear so every weight has a learnable gate.

    Architecture: 3072 → 1024 → 512 → 256 → 10
    BatchNorm and ReLU activations between layers.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dims: (B, 3, 32, 32) → (B, 3072)
        x = x.view(x.size(0), -1)
        return self.net(x)

    def prunable_layers(self):
        """Yield all PrunableLinear sub-modules."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 gate norms across all PrunableLinear layers."""
        return sum(layer.sparsity_loss() for layer in self.prunable_layers())

    def compute_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Fraction of weights whose gate < threshold (i.e., effectively pruned).
        Returns a value in [0, 1].
        """
        total = 0
        pruned = 0
        for layer in self.prunable_layers():
            gates = layer.get_gates()
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
        return pruned / total if total > 0 else 0.0


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def get_dataloaders(batch_size: int = 256):
    """
    Returns CIFAR-10 train and test DataLoaders.
    Applies standard normalisation (mean/std of CIFAR-10 channel statistics).
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# PART 3: TRAINING LOOP
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lam: float) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        avg_total_loss   — average (CE + λ·sparsity) loss over batches
        avg_clf_loss     — average cross-entropy loss (for monitoring)
    """
    model.train()
    total_loss_sum = 0.0
    clf_loss_sum   = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        # Classification loss
        clf_loss = F.cross_entropy(logits, labels)

        # Sparsity regularisation: L1 of all gate values
        sparsity_loss = model.total_sparsity_loss()

        # Combined loss
        total_loss = clf_loss + lam * sparsity_loss

        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        clf_loss_sum   += clf_loss.item()

    n = len(loader)
    return total_loss_sum / n, clf_loss_sum / n


@torch.no_grad()
def evaluate(model, loader) -> float:
    """Return top-1 accuracy on the given DataLoader."""
    model.eval()
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total


# ─────────────────────────────────────────────
# FULL TRAINING RUN FOR ONE LAMBDA
# ─────────────────────────────────────────────

def run_experiment(lam: float, epochs: int, train_loader, test_loader,
                   lr: float = 1e-3) -> dict:
    """
    Train a fresh SelfPruningNet with the given lambda value.

    Returns a dict with:
        model, test_accuracy, sparsity, history (loss/acc per epoch)
    """
    print(f"\n{'='*55}")
    print(f"  Training with λ = {lam}")
    print(f"{'='*55}")

    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"total_loss": [], "clf_loss": [], "test_acc": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        total_loss, clf_loss = train_one_epoch(model, train_loader, optimizer, lam)
        scheduler.step()
        test_acc = evaluate(model, test_loader)
        sparsity  = model.compute_sparsity()

        history["total_loss"].append(total_loss)
        history["clf_loss"].append(clf_loss)
        history["test_acc"].append(test_acc)

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Total Loss: {total_loss:.4f} | "
                  f"CE Loss: {clf_loss:.4f} | "
                  f"Test Acc: {test_acc*100:.2f}% | "
                  f"Sparsity: {sparsity*100:.1f}% | "
                  f"Elapsed: {elapsed:.0f}s")

    final_acc      = evaluate(model, test_loader)
    final_sparsity = model.compute_sparsity()

    print(f"\n  ✓  Final Test Accuracy : {final_acc*100:.2f}%")
    print(f"  ✓  Final Sparsity Level: {final_sparsity*100:.2f}%")

    return {
        "model":    model,
        "lam":      lam,
        "accuracy": final_acc,
        "sparsity": final_sparsity,
        "history":  history,
    }


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_gate_distribution(result: dict, save_path: str = "gate_distribution.png"):
    """
    Plot a histogram of all final gate values for the given model.
    A successful pruning will show a large spike near 0
    and a secondary cluster of "important" gates away from 0.
    """
    model = result["model"]
    all_gates = []

    for layer in model.prunable_layers():
        gates = layer.get_gates().cpu().numpy().flatten()
        all_gates.append(gates)

    all_gates = np.concatenate(all_gates)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(all_gates, bins=100, color="#2563EB", edgecolor="white",
            linewidth=0.3, alpha=0.9)
    ax.set_xlabel("Gate Value", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(
        f"Gate Value Distribution  (λ = {result['lam']})\n"
        f"Sparsity: {result['sparsity']*100:.1f}%  |  "
        f"Test Acc: {result['accuracy']*100:.2f}%",
        fontsize=13,
    )
    ax.axvline(x=0.01, color="#EF4444", linestyle="--", linewidth=1.2,
               label="Prune threshold (0.01)")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Gate distribution plot saved → {save_path}")


def plot_accuracy_vs_lambda(results: list, save_path: str = "lambda_tradeoff.png"):
    """
    Bar chart comparing test accuracy and sparsity across lambda values.
    """
    lambdas   = [str(r["lam"]) for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    sparsities = [r["sparsity"] * 100 for r in results]

    x = np.arange(len(lambdas))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, accuracies, width, label="Test Accuracy (%)",
                    color="#2563EB", alpha=0.85)
    bars2 = ax2.bar(x + width/2, sparsities, width, label="Sparsity (%)",
                    color="#10B981", alpha=0.85)

    ax1.set_xlabel("Lambda (λ)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12, color="#2563EB")
    ax2.set_ylabel("Sparsity (%)", fontsize=12, color="#10B981")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"λ={l}" for l in lambdas], fontsize=11)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    plt.title("Sparsity vs Accuracy Trade-off Across λ Values", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Lambda trade-off plot saved → {save_path}")


def plot_training_curves(results: list, save_path: str = "training_curves.png"):
    """Plot test accuracy over epochs for each lambda value."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2563EB", "#F59E0B", "#EF4444"]

    for result, color in zip(results, colors):
        acc_pct = [a * 100 for a in result["history"]["test_acc"]]
        ax.plot(acc_pct, label=f"λ = {result['lam']}", color=color, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Test Accuracy During Training for Different λ Values", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training curves plot saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Hyperparameters ──────────────────────
    EPOCHS     = 30       # Increase to 50–80 for better accuracy
    BATCH_SIZE = 256
    LR         = 1e-3

    # Three lambda values: low / medium / high sparsity pressure
    LAMBDAS = [1e-5, 1e-4, 1e-3]

    # ── Data ─────────────────────────────────
    print("Loading CIFAR-10 ...")
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # ── Experiments ──────────────────────────
    results = []
    for lam in LAMBDAS:
        result = run_experiment(lam, EPOCHS, train_loader, test_loader, lr=LR)
        results.append(result)

    # ── Summary Table ─────────────────────────
    print("\n" + "="*55)
    print(f"  {'Lambda':<12} {'Test Accuracy':>16} {'Sparsity (%)':>14}")
    print("  " + "-"*50)
    for r in results:
        print(f"  {r['lam']:<12} {r['accuracy']*100:>14.2f}%  {r['sparsity']*100:>12.1f}%")
    print("="*55)

    # ── Pick best model (highest acc) ────────
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n  Best model: λ = {best['lam']}  "
          f"(Acc={best['accuracy']*100:.2f}%, Sparsity={best['sparsity']*100:.1f}%)")

    # ── Plots ─────────────────────────────────
    plot_gate_distribution(best, save_path="gate_distribution.png")
    plot_accuracy_vs_lambda(results, save_path="lambda_tradeoff.png")
    plot_training_curves(results, save_path="training_curves.png")

    print("\nAll done. Check gate_distribution.png, lambda_tradeoff.png, training_curves.png")


if __name__ == "__main__":
    main()
