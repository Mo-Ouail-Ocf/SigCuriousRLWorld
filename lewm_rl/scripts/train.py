#!/usr/bin/env python3
"""
train.py — Main training script for the LeWM Curiosity-Driven RL Framework.

Usage:
    python scripts/train.py --config configs/stage1.yaml
    python scripts/train.py --config configs/stage2.yaml --seed 123
    python scripts/train.py --config configs/stage3.yaml --device cuda --steps 1000000

Arguments:
    --config   Path to YAML config file (required)
    --stage    Override stage from config (optional)
    --seed     Override random seed (optional)
    --device   Override device: "cpu", "cuda", "auto" (optional)
    --steps    Override total_steps (optional)
    --env      Override environment ID (optional)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.factory import load_config, build_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LeWM curiosity-driven RL agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--stage", type=str, choices=["stage1", "stage2", "stage3"],
                        help="Override stage from config")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--steps", type=int, help="Override total_steps")
    parser.add_argument("--env", type=str, help="Override environment ID")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Apply CLI overrides
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device
    if args.steps is not None:
        config.setdefault("training", {})["total_steps"] = args.steps
    if args.env is not None:
        config.setdefault("env", {})["id"] = args.env

    stage = args.stage or config.get("stage", "stage1")
    print(f"\n{'='*60}")
    print(f"  LeWM Curiosity-Driven RL Framework")
    print(f"  Stage: {stage}")
    print(f"  Env:   {config.get('env', {}).get('id', 'unknown')}")
    print(f"  Steps: {config.get('training', {}).get('total_steps', 500000):,}")
    print(f"{'='*60}\n")

    # Build trainer
    trainer = build_stage(config, stage=stage)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    results = trainer.train()

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Return:    {results['best_return']:.2f}")
    print(f"  Total Steps:    {results['total_steps']:,}")
    print(f"  Total Episodes: {results['total_episodes']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
