"""
Training Orchestrator — coordinates all components for each experimental stage.

Implements the 3-stage ablation strategy:
  Stage 1: LeWM as Intrinsic Reward ONLY (detached, separate CNN policy)
  Stage 2: LeWM as Representation ONLY (no intrinsic reward, latent policy)
  Stage 3: Full Model — latent representation + intrinsic reward (joint)

Each stage shares the same training loop structure but differs in:
  - What the policy receives as input (pixels vs. z_t)
  - Whether intrinsic rewards are active
  - How gradients flow between LeWM and the RL agent
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.lewm.world_model import LeWorldModel
from src.agents.ppo import PPO, CNNActorCritic, LatentActorCritic
from src.rewards.intrinsic_reward import IntrinsicRewardModule
from src.utils.replay_buffer import ReplayBuffer
from src.utils.logger import Logger


class RolloutBuffer:
    """On-policy rollout buffer for PPO.

    Accumulates T steps of experience from a single environment.
    """

    def __init__(
        self,
        T: int,
        obs_shape: tuple,
        action_dim: int,
        device: str,
        discrete: bool = False,
    ) -> None:
        self.T = T
        self.device = device
        self.discrete = discrete
        self.ptr = 0
        self.obs = torch.zeros(T, *obs_shape)
        self.latents = None  # allocated lazily if needed
        if discrete:
            self.actions = torch.zeros(T, dtype=torch.long)
        else:
            self.actions = torch.zeros(T, action_dim)
        self.log_probs = torch.zeros(T)
        self.values = torch.zeros(T)
        self.rewards = torch.zeros(T)
        self.dones = torch.zeros(T)

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
        latent: torch.Tensor | None = None,
    ) -> None:
        t = self.ptr
        self.obs[t] = obs
        if self.discrete:
            self.actions[t] = action.reshape(-1)[0].long()
        else:
            if action.dim() == 0:
                action = action.unsqueeze(0)
            self.actions[t] = action
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = float(done)
        if latent is not None:
            if self.latents is None:
                self.latents = torch.zeros(self.T, *latent.shape)
            self.latents[t] = latent
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.T

    def get(self, next_obs_or_latent: torch.Tensor) -> dict:
        """Return complete rollout data for PPO update."""
        data = {
            "obs": (self.latents if self.latents is not None else self.obs).to(self.device),
            "actions": self.actions.to(self.device),
            "log_probs": self.log_probs.to(self.device),
            "values": self.values.to(self.device),
            "rewards": self.rewards.to(self.device),
            "dones": self.dones.to(self.device),
            "next_obs": next_obs_or_latent.to(self.device),
        }
        self.ptr = 0  # reset
        if self.latents is not None:
            self.latents = torch.zeros_like(self.latents)
        return data


class Trainer:
    """Training orchestrator for the LeWM curiosity-driven RL framework.

    Args:
        stage:           "stage1", "stage2", or "stage3"
        env:             Gymnasium environment instance
        lewm:            LeWorldModel instance
        actor_critic:    PPO Actor-Critic module
        ppo:             PPO trainer
        intrinsic_reward: IntrinsicRewardModule (None for Stage 2)
        replay_buffer:   ReplayBuffer for LeWM training
        logger:          Logger instance
        config:          Full training config dictionary
        device:          Torch device
    """

    def __init__(
        self,
        stage: str,
        env,
        lewm: LeWorldModel,
        actor_critic: nn.Module,
        ppo: PPO,
        intrinsic_reward: IntrinsicRewardModule | None,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        config: dict,
        device: torch.device,
    ) -> None:
        self.stage = stage
        self.env = env
        self.lewm = lewm.to(device)
        self.actor_critic = actor_critic.to(device)
        self.ppo = ppo
        self.intrinsic_reward = intrinsic_reward
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.config = config
        self.device = device

        # LeWM optimizer (separate from PPO optimizer)
        self.lewm_optimizer = torch.optim.Adam(
            lewm.parameters(),
            lr=config.get("lewm_lr", 3e-4),
            weight_decay=config.get("lewm_weight_decay", 1e-5),
        )

        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_return = float("-inf")

        # Config shortcuts
        self.total_steps = int(config.get("total_steps", 1_000_000))
        self.rollout_steps = config.get("rollout_steps", 2048)
        self.lewm_batch_size = config.get("lewm_batch_size", 32)
        self.lewm_traj_len = config.get("lewm_traj_len", 8)
        self.lewm_update_freq = config.get("lewm_update_freq", 1)
        self.lewm_warmup_steps = config.get("lewm_warmup_steps", 5000)
        self.checkpoint_freq = config.get("checkpoint_freq", 50_000)
        self.eval_freq = config.get("eval_freq", 10_000)
        self.save_dir = Path(config.get("save_dir", "results/checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Determine action dim
        action_space = env.action_space
        if hasattr(action_space, "n"):
            self.action_dim = action_space.n
            self.discrete = True
        else:
            self.action_dim = action_space.shape[0]
            self.discrete = False

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """Run the full training loop.

        Returns:
            Final training metrics summary.
        """
        obs_np, _ = self.env.reset()
        obs = torch.from_numpy(obs_np).float()

        rollout_buf = RolloutBuffer(
            T=self.rollout_steps,
            obs_shape=obs.shape,
            action_dim=self.action_dim,
            device=str(self.device),
            discrete=self.discrete,
        )

        # LeWM warmup: collect random data before RL starts
        if self.lewm_warmup_steps > 0:
            print(f"[{self.stage}] LeWM warmup: collecting {self.lewm_warmup_steps} steps...")
            self._warmup(self.lewm_warmup_steps)

        episode_reward_ext = 0.0
        episode_reward_int = 0.0
        episode_len = 0
        episode_start = time.time()

        print(f"[{self.stage}] Starting training for {self.total_steps} steps...")

        while self.global_step < self.total_steps:
            # -------- Collect rollout --------
            self.actor_critic.eval()
            self.lewm.eval()
            with torch.no_grad():
                obs_t = obs.unsqueeze(0).to(self.device)

                # Get latent (always encoded, for both reward computation and policy)
                z_t = self.lewm.encode(obs_t).squeeze(0)  # (D,)

                # Policy input depends on stage
                if self.stage == "stage1":
                    policy_input = obs_t.squeeze(0)  # raw pixels for Stage 1
                else:
                    policy_input = z_t  # latent for Stages 2 & 3

                dist, value = self.actor_critic(policy_input.unsqueeze(0))
                action = dist.sample().squeeze(0)
                log_prob = dist.log_prob(action)
                if log_prob.dim() > 0:
                    log_prob = log_prob.sum()

            # Environment step
            if self.discrete:
                action_np = action.item()
            else:
                action_np = action.detach().cpu().numpy()

            next_obs_np, r_env, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            next_obs = torch.from_numpy(next_obs_np).float()

            # -------- Compute intrinsic reward --------
            r_int = 0.0
            if self.intrinsic_reward is not None and self.stage != "stage2":
                with torch.no_grad():
                    next_obs_t = next_obs.unsqueeze(0).to(self.device)
                    action_for_reward = action.unsqueeze(0).to(self.device)
                    if self.discrete:
                        action_for_reward = action_for_reward.unsqueeze(-1)
                    action_for_reward = action_for_reward.float()

                    r_raw = self.lewm.compute_intrinsic_reward(
                        obs_t=obs_t,
                        action_t=action_for_reward,
                        obs_t1=next_obs_t,
                    )

                    if self.intrinsic_reward.normalize:
                        self.intrinsic_reward.update_stats(r_raw)
                        r_int_tensor = (
                            (r_raw - self.intrinsic_reward.running_mean)
                            / (self.intrinsic_reward.running_std + self.intrinsic_reward.epsilon)
                        )
                    else:
                        r_int_tensor = r_raw

                    self.intrinsic_reward._reward_history.extend(r_raw.detach().cpu().tolist())
                    r_int_tensor = self.intrinsic_reward.lambda_int * r_int_tensor
                    r_int = r_int_tensor.item()

            r_total = float(r_env) + r_int

            # Store in rollout buffer
            rollout_buf.add(
                obs=obs,
                action=action.cpu(),
                log_prob=log_prob.cpu(),
                value=value.squeeze().cpu(),
                reward=r_total,
                done=done,
            )

            # Store in replay buffer for LeWM training
            act_np = np.array([action_np] if self.discrete else action_np, dtype=np.float32)
            self.replay_buffer.add(obs_np, act_np, float(r_env), next_obs_np, done)

            # Update tracking
            obs_np = next_obs_np
            obs = next_obs
            episode_reward_ext += float(r_env)
            episode_reward_int += r_int
            episode_len += 1
            self.global_step += 1

            # -------- Episode end --------
            if done:
                episode_time = time.time() - episode_start
                self.episode_count += 1
                metrics = {
                    "reward/extrinsic": episode_reward_ext,
                    "reward/intrinsic": episode_reward_int,
                    "reward/total": episode_reward_ext + episode_reward_int,
                    "episode/length": episode_len,
                    "episode/fps": episode_len / max(episode_time, 1e-6),
                }
                self.logger.log(metrics, step=self.global_step)

                if episode_reward_ext > self.best_return:
                    self.best_return = episode_reward_ext
                    self._save_checkpoint("best")

                # Reset
                obs_np, _ = self.env.reset()
                obs = torch.from_numpy(obs_np).float()
                episode_reward_ext = 0.0
                episode_reward_int = 0.0
                episode_len = 0
                episode_start = time.time()

            # -------- PPO update --------
            if rollout_buf.is_full():
                rollout = rollout_buf.get(next_obs.cpu())
                ppo_metrics = self.ppo.update(rollout)

                ppo_log = {f"ppo/{k}": v for k, v in ppo_metrics.items()}
                self.logger.log(ppo_log, step=self.global_step)

            # -------- LeWM update --------
            if (self.global_step % self.lewm_update_freq == 0
                    and len(self.replay_buffer) >= self.lewm_traj_len * self.lewm_batch_size):
                lewm_metrics = self._update_lewm()
                self.logger.log(lewm_metrics, step=self.global_step)

            # -------- Checkpoint --------
            if self.global_step % self.checkpoint_freq == 0:
                self._save_checkpoint(f"step_{self.global_step}")

            self.logger.step()

        # Final checkpoint
        self._save_checkpoint("final")
        self.logger.save_metrics()

        return {
            "best_return": self.best_return,
            "total_steps": self.global_step,
            "total_episodes": self.episode_count,
        }

    # ------------------------------------------------------------------
    # LeWM training
    # ------------------------------------------------------------------

    def _update_lewm(self) -> dict[str, float]:
        """Sample a batch of trajectories and update LeWM.

        Returns:
            Dictionary of LeWM training metrics.
        """
        self.lewm.train()
        self.lewm_optimizer.zero_grad()

        batch = self.replay_buffer.sample_trajectories(
            batch_size=self.lewm_batch_size,
            traj_len=self.lewm_traj_len,
        )
        obs = batch["obs"].to(self.device).float()     # (B, T, C, H, W)
        actions = batch["actions"].to(self.device)      # (B, T, action_dim)

        # Stage 1: detach LeWM from RL — still update LeWM on its own loss
        result = self.lewm(obs, actions)

        result["loss"].backward()
        nn.utils.clip_grad_norm_(self.lewm.parameters(), 1.0)
        self.lewm_optimizer.step()
        self.lewm.eval()

        # Compute latent variance for collapse monitoring
        with torch.no_grad():
            latent_var = self.lewm.get_latent_variance(result["embeddings"])

        return {
            "lewm/loss": result["loss"].item(),
            "lewm/pred_loss": result["pred_loss"].item(),
            "lewm/sigreg_loss": result["sigreg_loss"].item(),
            "lewm/latent_variance": latent_var.item(),
        }

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self, n_steps: int) -> None:
        """Collect random experience to warm up LeWM before RL starts."""
        obs_np, _ = self.env.reset()
        for _ in range(n_steps):
            action = self.env.action_space.sample()
            act_np = np.array([action] if self.discrete else action, dtype=np.float32)
            next_obs_np, r, term, trunc, _ = self.env.step(action)
            done = term or trunc
            self.replay_buffer.add(obs_np, act_np, r, next_obs_np, done)
            obs_np = next_obs_np
            if done:
                obs_np, _ = self.env.reset()

            # Train LeWM during warmup
            if (len(self.replay_buffer) >= self.lewm_traj_len * self.lewm_batch_size
                    and len(self.replay_buffer) % 100 == 0):
                self._update_lewm()

        print(f"[{self.stage}] Warmup complete. Replay buffer size: {len(self.replay_buffer)}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> None:
        """Save model checkpoints.

        Args:
            tag: Checkpoint identifier (e.g., "best", "final", "step_50000").
        """
        stage_dir = self.save_dir / self.stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "lewm": self.lewm.state_dict(),
            "actor_critic": self.actor_critic.state_dict(),
            "lewm_optimizer": self.lewm_optimizer.state_dict(),
            "ppo_optimizer": self.ppo.optimizer.state_dict(),
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "best_return": self.best_return,
        }, stage_dir / f"{tag}.pt")

    def load_checkpoint(self, path: str) -> None:
        """Load a saved checkpoint.

        Args:
            path: Path to checkpoint .pt file.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.lewm.load_state_dict(ckpt["lewm"])
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        self.lewm_optimizer.load_state_dict(ckpt["lewm_optimizer"])
        self.ppo.optimizer.load_state_dict(ckpt["ppo_optimizer"])
        self.global_step = ckpt["global_step"]
        self.episode_count = ckpt["episode_count"]
        self.best_return = ckpt["best_return"]
        print(f"[{self.stage}] Loaded checkpoint from {path} (step {self.global_step})")
