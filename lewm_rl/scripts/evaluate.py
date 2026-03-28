#!/usr/bin/env python3
"""
evaluate.py — Evaluate trained models and generate comparison plots.

Usage:
    # Evaluate a single stage checkpoint:
    python scripts/evaluate.py --config configs/stage1.yaml --checkpoint results/checkpoints/stage1/best.pt

    # Compare all three stages and generate plots:
    python scripts/evaluate.py --compare \
        --stage1 results/checkpoints/stage1/best.pt \
        --stage2 results/checkpoints/stage2/best.pt \
        --stage3 results/checkpoints/stage3/best.pt \
        --config configs/stage1.yaml

    # Generate plots from existing log files:
    python scripts/evaluate.py --plot-only --log-dir results/logs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.training.factory import load_config, build_stage
from src.utils.plotting import (
    plot_learning_curves, plot_intrinsic_rewards,
    plot_latent_variance, plot_stage_comparison, load_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LeWM RL agents")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--checkpoint", type=str, help="Single checkpoint to evaluate")
    parser.add_argument("--stage", type=str, choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--n-episodes", type=int, default=20, help="Number of eval episodes")
    parser.add_argument("--compare", action="store_true", help="Compare all 3 stages")
    parser.add_argument("--stage1", type=str, help="Stage 1 checkpoint for comparison")
    parser.add_argument("--stage2", type=str, help="Stage 2 checkpoint")
    parser.add_argument("--stage3", type=str, help="Stage 3 checkpoint")
    parser.add_argument("--plot-only", action="store_true", help="Generate plots from existing logs")
    parser.add_argument("--log-dir", type=str, default="results/logs", help="Log directory for plot-only mode")
    parser.add_argument("--plot-dir", type=str, default="results/plots", help="Where to save plots")
    return parser.parse_args()


def evaluate_policy(trainer, n_episodes: int = 20) -> dict:
    """Run evaluation episodes and return statistics.

    Args:
        trainer:    Trainer instance with loaded model.
        n_episodes: Number of evaluation episodes.

    Returns:
        Dictionary with mean/std/min/max of episode returns.
    """
    trainer.actor_critic.eval()
    trainer.lewm.eval()
    device = trainer.device

    episode_returns = []

    for _ in range(n_episodes):
        obs_np, _ = trainer.env.reset()
        episode_return = 0.0
        done = False

        while not done:
            obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)

            with torch.no_grad():
                if trainer.stage == "stage1":
                    policy_input = obs.squeeze(0)
                else:
                    z = trainer.lewm.encode(obs).squeeze(0)
                    policy_input = z

                dist, _ = trainer.actor_critic(policy_input.unsqueeze(0))
                action = dist.sample().squeeze(0)

            action_np = action.item() if trainer.discrete else action.cpu().numpy()
            obs_np, reward, terminated, truncated, _ = trainer.env.step(action_np)
            done = terminated or truncated
            episode_return += reward

        episode_returns.append(episode_return)

    returns = np.array(episode_returns)
    return {
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "median": float(np.median(returns)),
    }


def generate_all_plots(log_dir: str, plot_dir: str) -> None:
    """Generate all comparison plots from saved log files.

    Args:
        log_dir:  Root directory containing per-run log subdirectories.
        plot_dir: Directory to save generated plots.
    """
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir)

    # Discover stage logs
    metrics_by_stage = {}
    for stage in ["stage1", "stage2", "stage3"]:
        stage_dirs = list(log_path.glob(f"{stage}_*"))
        if stage_dirs:
            # Use most recent
            stage_dir = sorted(stage_dirs)[-1]
            try:
                metrics = load_metrics(stage_dir)
                metrics_by_stage[stage] = metrics
                print(f"Loaded {len(metrics)} records for {stage} from {stage_dir}")
            except FileNotFoundError:
                print(f"No metrics found for {stage}")

    if not metrics_by_stage:
        print("No metrics found. Run training first.")
        return

    # Learning curves
    plot_learning_curves(
        metrics_by_stage,
        save_path=plot_path / "learning_curves.png",
        title="Reward Learning Curves: Stage Comparison",
    )

    # Per-stage detailed plots
    for stage_name, metrics in metrics_by_stage.items():
        plot_intrinsic_rewards(
            metrics,
            save_path=plot_path / f"{stage_name}_rewards.png",
        )
        plot_latent_variance(
            metrics,
            save_path=plot_path / f"{stage_name}_latent_variance.png",
        )

    # Stage comparison bar chart (using final 10% of training data)
    comparison_data = {}
    for stage_name, metrics in metrics_by_stage.items():
        rewards = [m.get("reward/extrinsic", 0) for m in metrics if "reward/extrinsic" in m]
        if rewards:
            final_rewards = rewards[int(0.9 * len(rewards)):]  # last 10%
            comparison_data[stage_name] = {
                "mean": np.mean(final_rewards),
                "std": np.std(final_rewards),
                "label": {
                    "stage1": "Stage 1\n(Intrinsic Only)",
                    "stage2": "Stage 2\n(Representation Only)",
                    "stage3": "Stage 3\n(Full Model)",
                }.get(stage_name, stage_name),
            }

    if comparison_data:
        plot_stage_comparison(
            comparison_data,
            save_path=plot_path / "stage_comparison.png",
        )

    print(f"\nAll plots saved to {plot_path}/")


def main() -> None:
    args = parse_args()

    # Plot-only mode
    if args.plot_only:
        generate_all_plots(args.log_dir, args.plot_dir)
        return

    # Need config for evaluation
    if not args.config:
        print("Error: --config required for evaluation mode")
        return

    config = load_config(args.config)

    # Single checkpoint evaluation
    if args.checkpoint and args.stage:
        print(f"Evaluating {args.stage} checkpoint: {args.checkpoint}")
        trainer = build_stage(config, stage=args.stage)
        trainer.load_checkpoint(args.checkpoint)
        results = evaluate_policy(trainer, n_episodes=args.n_episodes)
        print(f"\n{args.stage} Results ({args.n_episodes} episodes):")
        for k, v in results.items():
            print(f"  {k:10s}: {v:.3f}")
        return

    # Multi-stage comparison
    if args.compare:
        checkpoints = {
            "stage1": args.stage1,
            "stage2": args.stage2,
            "stage3": args.stage3,
        }
        checkpoints = {k: v for k, v in checkpoints.items() if v is not None}

        comparison = {}
        for stage_name, ckpt_path in checkpoints.items():
            print(f"\nEvaluating {stage_name}...")
            trainer = build_stage(config, stage=stage_name)
            trainer.load_checkpoint(ckpt_path)
            results = evaluate_policy(trainer, n_episodes=args.n_episodes)
            comparison[stage_name] = results
            comparison[stage_name]["label"] = {
                "stage1": "Stage 1\n(Intrinsic Only)",
                "stage2": "Stage 2\n(Representation Only)",
                "stage3": "Stage 3\n(Full Model)",
            }[stage_name]
            print(f"  Mean return: {results['mean']:.3f} ± {results['std']:.3f}")

        # Generate comparison plot
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
        plot_stage_comparison(comparison, save_path=f"{args.plot_dir}/eval_comparison.png")

        print("\nSummary:")
        for stage_name, res in comparison.items():
            print(f"  {stage_name}: {res['mean']:.3f} ± {res['std']:.3f}")

        generate_all_plots(args.log_dir, args.plot_dir)


if __name__ == "__main__":
    main()
