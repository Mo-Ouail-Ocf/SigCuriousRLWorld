"""
Intrinsic Reward Module.

Computes stable curiosity-driven intrinsic rewards using LeWM's prediction error
in latent space:

    r_int = ||ẑ_{t+1} - z_{t+1}||²

Unlike ICM and RND, this reward is stable because:
  - z_{t+1} lies on a structured Gaussian manifold (SIGReg guarantee)
  - High reward → genuine predictive uncertainty (not pixel noise)
  - The reward scale remains meaningful throughout training
"""

from __future__ import annotations

import torch
import torch.nn as nn
from collections import deque


class IntrinsicRewardModule(nn.Module):
    """Computes and normalizes LeWM-based intrinsic rewards.

    Intrinsic reward: r_int = ||ẑ_{t+1} - z_{t+1}||²

    Applies running normalization using Welford's online algorithm to keep
    rewards on a consistent scale throughout training.

    Args:
        lambda_int:      Coefficient for intrinsic reward (default 0.01).
        normalize:       Whether to normalize rewards using running statistics.
        buffer_size:     Size of running statistics buffer.
        epsilon:         Numerical stability constant.
    """

    def __init__(
        self,
        lambda_int: float = 0.01,
        normalize: bool = True,
        buffer_size: int = 1000,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lambda_int = lambda_int
        self.normalize = normalize
        self.epsilon = epsilon

        # Running statistics (Welford's algorithm)
        self.register_buffer("_count", torch.tensor(0, dtype=torch.float64))
        self.register_buffer("_mean", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("_M2", torch.tensor(0.0, dtype=torch.float64))

        # History buffer for logging
        self._reward_history: deque[float] = deque(maxlen=buffer_size)

    def update_stats(self, rewards: torch.Tensor) -> None:
        """Update running mean/variance using Welford's online algorithm.

        Args:
            rewards: (N,) reward tensor.
        """
        rewards_f64 = rewards.detach().to(self._mean.device).double()
        for r in rewards_f64.flatten():
            self._count += 1
            delta = r - self._mean
            self._mean += delta / self._count
            delta2 = r - self._mean
            self._M2 += delta * delta2

    @property
    def running_mean(self) -> float:
        return self._mean.item()

    @property
    def running_std(self) -> float:
        if self._count < 2:
            return 1.0
        variance = self._M2 / (self._count - 1)
        return (variance.sqrt() + self.epsilon).item()

    def compute(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute raw intrinsic reward.

        Args:
            z_pred:   (B, D) predicted next embedding ẑ_{t+1}.
            z_target: (B, D) actual next embedding z_{t+1}.

        Returns:
            (B,) raw intrinsic reward values.
        """
        return (z_pred - z_target).pow(2).mean(dim=-1)  # (B,)

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        update_stats: bool = True,
    ) -> torch.Tensor:
        """Compute normalized intrinsic reward.

        Args:
            z_pred:       (B, D) predicted next embedding.
            z_target:     (B, D) actual next embedding.
            update_stats: Whether to update running statistics.

        Returns:
            (B,) scaled intrinsic reward λ_int * r_int_normalized.
        """
        with torch.no_grad():
            r_raw = self.compute(z_pred, z_target)  # (B,)

        if update_stats:
            self.update_stats(r_raw)

        if self.normalize:
            r_norm = (r_raw - self.running_mean) / (self.running_std + self.epsilon)
        else:
            r_norm = r_raw

        # Log to history
        self._reward_history.extend(r_raw.detach().cpu().tolist())

        return self.lambda_int * r_norm

    def total_reward(
        self,
        r_env: torch.Tensor,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute total reward = r_env + λ * r_int.

        Args:
            r_env:    (B,) extrinsic (environment) reward.
            z_pred:   (B, D) predicted next embedding.
            z_target: (B, D) actual next embedding.

        Returns:
            Tuple of (total_reward, intrinsic_reward), both (B,).
        """
        r_int = self.forward(z_pred, z_target)
        r_total = r_env + r_int
        return r_total, r_int

    def get_stats(self) -> dict[str, float]:
        """Get reward statistics for logging.

        Returns:
            Dictionary with mean, std, min, max of recent rewards.
        """
        if not self._reward_history:
            return {}
        history = list(self._reward_history)
        t = torch.tensor(history)
        return {
            "intrinsic_reward/mean": t.mean().item(),
            "intrinsic_reward/std": t.std().item(),
            "intrinsic_reward/min": t.min().item(),
            "intrinsic_reward/max": t.max().item(),
            "intrinsic_reward/running_mean": self.running_mean,
            "intrinsic_reward/running_std": self.running_std,
        }