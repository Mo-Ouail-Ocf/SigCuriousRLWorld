"""
Replay Buffer for LeWM training.

Stores (obs, action, reward, next_obs, done) transitions.
LeWM is trained on sub-trajectories sampled from this buffer.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Circular replay buffer for trajectory data.

    Stores individual transitions. For LeWM training, we sample
    sub-trajectories by sampling a start index and taking the next T steps.

    Args:
        capacity:      Maximum number of transitions to store.
        obs_shape:     Shape of a single observation (C, H, W).
        action_dim:    Dimensionality of the action.
        device:        Device for returned tensors.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_dim: int,
        device: str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate numpy arrays for efficiency
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer.

        Args:
            obs:      Current observation (C, H, W).
            action:   Action taken.
            reward:   Reward received.
            next_obs: Next observation.
            done:     Episode terminal flag.
        """
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action if np.ndim(action) > 0 else [action]
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary of tensors on self.device.
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]).to(self.device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(self.device),
            "actions": torch.from_numpy(self.actions[idx]).to(self.device),
            "rewards": torch.from_numpy(self.rewards[idx]).to(self.device),
            "dones": torch.from_numpy(self.dones[idx]).to(self.device),
        }

    def sample_trajectories(
        self, batch_size: int, traj_len: int
    ) -> dict[str, torch.Tensor]:
        """Sample sub-trajectories for LeWM training.

        Samples `batch_size` start indices and returns the next
        `traj_len` consecutive transitions.

        Note: Simple implementation that ignores episode boundaries.
        For production use, track episode boundaries for proper masking.

        Args:
            batch_size: Number of trajectories to sample.
            traj_len:   Length of each sub-trajectory.

        Returns:
            Dictionary with (B, T, ...) tensors on self.device.
        """
        # Ensure we have enough data
        assert self.size >= traj_len, f"Not enough data: {self.size} < {traj_len}"

        # Sample start indices (avoid wrap-around by leaving `traj_len` gap before ptr)
        valid_end = (self.ptr - traj_len) % self.capacity
        if valid_end > self.ptr:
            valid_starts = np.arange(self.ptr, valid_end)
        else:
            valid_starts = np.concatenate([
                np.arange(0, valid_end),
                np.arange(self.ptr, self.capacity),
            ])

        if len(valid_starts) < batch_size:
            # Fallback: random sampling with wrap check
            valid_starts = np.arange(self.size - traj_len)

        start_indices = np.random.choice(valid_starts, size=batch_size, replace=True)

        # Gather trajectory data
        obs_batch = np.zeros((batch_size, traj_len, *self.obs_shape), dtype=np.uint8)
        act_batch = np.zeros((batch_size, traj_len, self.action_dim), dtype=np.float32)

        for i, start in enumerate(start_indices):
            indices = (start + np.arange(traj_len)) % self.capacity
            obs_batch[i] = self.obs[indices]
            act_batch[i] = self.actions[indices]

        return {
            "obs": torch.from_numpy(obs_batch).to(self.device),
            "actions": torch.from_numpy(act_batch).to(self.device),
        }

    def __len__(self) -> int:
        return self.size
