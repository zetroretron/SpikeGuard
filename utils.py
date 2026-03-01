"""
SpikeGuard — Utilities
Seeding, device selection, visualization helpers, and common utilities.
"""
import os
import random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Wildlife class labels ───────────────────────────────────
# 10-class subset for camera-trap classification
WILDLIFE_CLASSES = [
    "tiger", "leopard", "elephant", "deer", "wild_boar",
    "bear", "monkey", "bird", "human", "blank",
]
NUM_CLASSES = len(WILDLIFE_CLASSES)


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: {torch.cuda.get_device_name(0)} (CUDA)")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU")
    return device


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def plot_spike_raster(spike_data, layer_names=None, title="Spike Raster Plot",
                      save_path=None):
    """
    Plot spike raster showing neural firing patterns across layers and timesteps.

    Args:
        spike_data: Dict mapping layer_name -> tensor of shape (T, N) where
                    T=timesteps, N=subset of neurons.
        layer_names: Optional list of layer names for y-axis.
        title: Plot title.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(len(spike_data), 1, figsize=(12, 2 * len(spike_data)),
                              sharex=True, facecolor="#0d1117")

    if len(spike_data) == 1:
        axes = [axes]

    colors = ["#2ed573", "#00d2ff", "#7b2ff7", "#ff4757", "#ffa502"]

    for idx, (name, spikes) in enumerate(spike_data.items()):
        ax = axes[idx]
        ax.set_facecolor("#0d1117")

        if isinstance(spikes, torch.Tensor):
            spikes = spikes.cpu().detach().numpy()

        T, N = spikes.shape
        for neuron in range(min(N, 50)):  # Show max 50 neurons
            times = np.where(spikes[:, neuron] > 0)[0]
            ax.scatter(times, [neuron] * len(times), s=2,
                       color=colors[idx % len(colors)], alpha=0.8)

        ax.set_ylabel(name, fontsize=8, color="#e0e0e0")
        ax.tick_params(colors="#888")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Timestep", color="#e0e0e0")
    fig.suptitle(title, color="#2ed573", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")

    return fig


def plot_energy_bar(spike_count, max_spikes=10.0, save_path=None):
    """
    Create a visual energy consumption bar based on spike count.

    Args:
        spike_count: Normalized spike count (1-10 scale).
        max_spikes: Maximum expected spike count for normalization.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 1.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    ratio = min(spike_count / max_spikes, 1.0)

    # Color gradient: green (low energy) -> yellow -> red (high energy)
    if ratio < 0.4:
        color = "#2ed573"
    elif ratio < 0.7:
        color = "#ffa502"
    else:
        color = "#ff4757"

    ax.barh(0, ratio, height=0.6, color=color, alpha=0.9, edgecolor="none")
    ax.barh(0, 1.0, height=0.6, color="#1a1a2e", alpha=0.5, edgecolor="#333")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "2.5", "5.0", "7.5", "10.0"], color="#888")
    ax.set_xlabel("Estimated Spike Energy (normalized)", color="#aaa", fontsize=9)
    ax.set_title(f"Energy Proxy: {spike_count:.1f} / {max_spikes:.0f}",
                 color="#e0e0e0", fontsize=11)
    ax.spines["bottom"].set_color("#333")
    for s in ["top", "left", "right"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")

    return fig


def plot_pareto_front(results_dict, save_path=None):
    """
    Plot accuracy vs spike count Pareto frontier.

    Args:
        results_dict: Dict mapping model_name -> {"accuracy": float, "spikes": float,
                      "accuracy_std": float, "spikes_std": float}
        save_path: Optional save path.
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    colors = {"MegaDetector": "#ff4757", "Vanilla ANN": "#ffa502",
              "Vanilla SNN": "#00d2ff", "Energy-only SNN": "#7b2ff7",
              "SpikeGuard": "#2ed573"}

    for name, data in results_dict.items():
        c = colors.get(name, "#888")
        ax.scatter(data["spikes"], data["accuracy"], s=120, color=c,
                   edgecolors="white", linewidths=0.5, zorder=5, label=name)
        if "accuracy_std" in data and "spikes_std" in data:
            ax.errorbar(data["spikes"], data["accuracy"],
                        xerr=data["spikes_std"], yerr=data["accuracy_std"],
                        fmt="none", ecolor=c, alpha=0.4, capsize=3)

    ax.set_xlabel("Avg Spike Count (normalized)", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("Accuracy under 30% Tampering", color="#e0e0e0", fontsize=11)
    ax.set_title("Pareto Front: Accuracy vs Energy (Spike Count)",
                 color="#2ed573", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#e0e0e0",
              fontsize=9)
    ax.tick_params(colors="#888")
    ax.grid(True, alpha=0.15, color="#444")
    for s in ax.spines.values():
        s.set_color("#333")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")

    return fig
