#!/usr/bin/env python3
"""
run_experiments.py — Run all 3 stages sequentially and generate final comparison plots.

Usage:
    python experiments/run_experiments.py
    python experiments/run_experiments.py --steps 200000  # quick test run
    python experiments/run_experiments.py --stages 1 3    # only stages 1 and 3
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.factory import load_config, build_stage
from src.utils.plotting import (
    plot_learning_curves, plot_stage_comparison, load_metrics
)


def run_stage(stage: str, steps_override: int | None = None) -> dict:
    """Run a single stage experiment."""
    config = load_config(f"configs/{stage}.yaml")
    if steps_override:
        config.setdefault("training", {})["total_steps"] = steps_override
    print(f"\n{'─'*60}")
    print(f"  Starting {stage.upper()}")
    print(f"{'─'*60}")
    t0 = time.time()
    trainer = build_stage(config, stage=stage)
    results = trainer.train()
    results["wall_time"] = time.time() - t0
    print(f"  {stage} done in {results['wall_time']:.0f}s | "
          f"best_return={results['best_return']:.2f}")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None,
                   help="Override total_steps for all stages")
    p.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3],
                   choices=[1, 2, 3], help="Which stages to run")
    return p.parse_args()


def main():
    args = parse_args()
    all_results = {}

    for s in sorted(args.stages):
        stage = f"stage{s}"
        try:
            res = run_stage(stage, args.steps)
            all_results[stage] = res
        except Exception as e:
            print(f"[ERROR] {stage} failed: {e}")
            continue

    # Generate comparison plots from logs
    print("\nGenerating plots...")
    log_dir  = Path("results/logs")
    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics per stage
    metrics_by_stage = {}
    for stage in [f"stage{s}" for s in sorted(args.stages)]:
        stage_dirs = sorted(log_dir.glob(f"{stage}_*"))
        if stage_dirs:
            try:
                metrics_by_stage[stage] = load_metrics(stage_dirs[-1])
            except Exception:
                pass

    if metrics_by_stage:
        plot_learning_curves(
            metrics_by_stage,
            save_path=plot_dir / "learning_curves_comparison.png",
            title="LeWM RL: Stage Ablation Learning Curves",
        )

    # Summary
    print(f"\n{'='*60}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for stage, res in all_results.items():
        label = {
            "stage1": "Stage 1 (Intrinsic Reward Only)",
            "stage2": "Stage 2 (Representation Only)",
            "stage3": "Stage 3 (Full Model)",
        }.get(stage, stage)
        print(f"  {label}")
        print(f"    Best Return: {res.get('best_return', 'N/A'):.2f}")
        print(f"    Episodes:    {res.get('total_episodes', 'N/A')}")
        print(f"    Wall Time:   {res.get('wall_time', 0):.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
