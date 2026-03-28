"""
Plotting utilities for experiment results.

Generates:
  - Learning curves (reward vs. steps)
  - Exploration metrics
  - Latent space distribution (variance over time)
  - LeWM training curves (pred_loss, sigreg_loss)
  - Stage comparison plots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def load_metrics(log_dir: str | Path) -> list[dict]:
    """Load metrics from a saved JSON file.

    Args:
        log_dir: Directory containing metrics.json.

    Returns:
        List of metric records.
    """
    path = Path(log_dir) / "metrics.json"
    with open(path) as f:
        return json.load(f)


def smooth(values: list[float], window: int = 10) -> np.ndarray:
    """Apply moving average smoothing.

    Args:
        values: Raw values.
        window: Smoothing window size.

    Returns:
        Smoothed array.
    """
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_learning_curves(
    metrics_by_stage: dict[str, list[dict]],
    save_path: str | Path,
    title: str = "Learning Curves",
    smooth_window: int = 10,
) -> None:
    """Plot extrinsic reward learning curves for each stage.

    Args:
        metrics_by_stage: Dict mapping stage name → metrics list.
        save_path:        Path to save the figure.
        title:            Plot title.
        smooth_window:    Smoothing window for curves.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plots] matplotlib not installed.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"stage1": "#e74c3c", "stage2": "#3498db", "stage3": "#2ecc71"}

    for stage_name, metrics in metrics_by_stage.items():
        steps = [m["step"] for m in metrics if "reward/extrinsic" in m]
        rewards = [m["reward/extrinsic"] for m in metrics if "reward/extrinsic" in m]
        if not steps:
            continue
        color = colors.get(stage_name, "#95a5a6")
        ax.plot(steps, rewards, alpha=0.3, color=color)
        if len(rewards) > smooth_window:
            s_rewards = smooth(rewards, smooth_window)
            s_steps = steps[smooth_window - 1:]
            ax.plot(s_steps, s_rewards, color=color, linewidth=2,
                    label=f"{stage_name} (smoothed)")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return (Extrinsic)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Saved: {save_path}")


def plot_intrinsic_rewards(
    metrics: list[dict],
    save_path: str | Path,
    smooth_window: int = 10,
) -> None:
    """Plot intrinsic reward over training.

    Args:
        metrics:       List of metric records.
        save_path:     Path to save figure.
        smooth_window: Smoothing window.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extrinsic vs. intrinsic
    steps = [m["step"] for m in metrics if "reward/extrinsic" in m]
    r_ext = [m.get("reward/extrinsic", 0) for m in metrics if "reward/extrinsic" in m]
    r_int = [m.get("reward/intrinsic", 0) for m in metrics if "reward/extrinsic" in m]

    axes[0].plot(steps, r_ext, alpha=0.4, color="blue", label="Extrinsic")
    axes[0].plot(steps, r_int, alpha=0.4, color="red", label="Intrinsic")
    if len(r_ext) > smooth_window:
        axes[0].plot(steps[smooth_window - 1:], smooth(r_ext, smooth_window),
                     color="blue", linewidth=2)
        axes[0].plot(steps[smooth_window - 1:], smooth(r_int, smooth_window),
                     color="red", linewidth=2)
    axes[0].set_title("Extrinsic vs. Intrinsic Rewards")
    axes[0].set_xlabel("Steps")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # LeWM losses
    steps_wm = [m["step"] for m in metrics if "lewm/pred_loss" in m]
    pred_l = [m["lewm/pred_loss"] for m in metrics if "lewm/pred_loss" in m]
    sig_l = [m["lewm/sigreg_loss"] for m in metrics if "lewm/sigreg_loss" in m]

    if steps_wm:
        ax2 = axes[1].twinx()
        axes[1].plot(steps_wm, pred_l, color="purple", label="Pred Loss")
        ax2.plot(steps_wm, sig_l, color="orange", linestyle="--", label="SIGReg Loss")
        axes[1].set_title("LeWM Training Losses")
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Prediction Loss", color="purple")
        ax2.set_ylabel("SIGReg Loss", color="orange")
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Saved: {save_path}")


def plot_latent_variance(
    metrics: list[dict],
    save_path: str | Path,
) -> None:
    """Plot latent embedding variance over training (collapse monitor).

    If this drops to near-zero, collapse is occurring.

    Args:
        metrics:   List of metric records.
        save_path: Save path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps = [m["step"] for m in metrics if "lewm/latent_variance" in m]
    var = [m["lewm/latent_variance"] for m in metrics if "lewm/latent_variance" in m]

    if not steps:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, var, color="teal", linewidth=2)
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="Collapse threshold")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Mean Latent Variance")
    ax.set_title("Latent Space Variance (Collapse Monitor)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Saved: {save_path}")


def plot_stage_comparison(
    results: dict[str, dict],
    save_path: str | Path,
) -> None:
    """Bar chart comparing final performance across stages.

    Args:
        results:   Dict mapping stage → {'mean': x, 'std': y, 'label': str}.
        save_path: Save path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    stages = list(results.keys())
    means = [results[s]["mean"] for s in stages]
    stds = [results[s].get("std", 0) for s in stages]
    labels = [results[s].get("label", s) for s in stages]
    colors = ["#e74c3c", "#3498db", "#2ecc71"][:len(stages)]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=5,
                  alpha=0.8, edgecolor="black")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Episode Return")
    ax.set_title("Stage Ablation: Performance Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Saved: {save_path}")
