"""
Logging utilities.

Supports WandB and TensorBoard for experiment tracking.
Falls back gracefully if neither is available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class Logger:
    """Unified logger supporting WandB and TensorBoard.

    Args:
        run_name:   Experiment name.
        log_dir:    Directory for local logs and TensorBoard.
        use_wandb:  Whether to log to WandB.
        config:     Experiment config dict to log.
        project:    WandB project name.
    """

    def __init__(
        self,
        run_name: str,
        log_dir: str = "results/logs",
        use_wandb: bool = False,
        config: dict | None = None,
        project: str = "lewm-rl",
    ) -> None:
        self.run_name = run_name
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0
        self._metrics: list[dict] = []

        # Try WandB
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb.init(
                    project=project,
                    name=run_name,
                    config=config or {},
                    dir=str(self.log_dir),
                )
            except ImportError:
                print("[Logger] wandb not installed. Skipping W&B logging.")
            except Exception as e:
                print(f"[Logger] W&B init failed: {e}")

        # Try TensorBoard
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))
        except ImportError:
            pass  # TensorBoard not available

        # Save config
        if config:
            with open(self.log_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2, default=str)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dictionary of metrics.

        Args:
            metrics: Metrics to log (name → scalar value).
            step:    Global step (defaults to internal counter).
        """
        if step is None:
            step = self._step

        # Always save locally
        record = {"step": step, **metrics}
        self._metrics.append(record)

        # WandB
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

        # TensorBoard
        if self.tb_writer is not None:
            for k, v in metrics.items():
                try:
                    self.tb_writer.add_scalar(k, float(v), step)
                except (TypeError, ValueError):
                    pass

    def step(self) -> None:
        """Increment internal step counter."""
        self._step += 1

    def save_metrics(self) -> None:
        """Flush all metrics to a JSON file."""
        with open(self.log_dir / "metrics.json", "w") as f:
            json.dump(self._metrics, f, indent=2, default=str)

    def close(self) -> None:
        """Close all logging backends."""
        self.save_metrics()
        if self.wandb is not None:
            self.wandb.finish()
        if self.tb_writer is not None:
            self.tb_writer.close()
