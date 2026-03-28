"""
PPO Agent — Proximal Policy Optimization.

Supports two operating modes:
  Stage 1: CNNActorCritic  — raw pixel observations → CNN → policy head
  Stage 2/3: LatentActorCritic — LeWM latent z_t → MLP → policy head
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Callable, Iterable


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _build_cnn(in_channels: int, image_size: int) -> tuple[nn.Module, int]:
    """Build a CNN backbone adaptive to image size.

    Returns (cnn_module, output_flat_dim).
    """
    if image_size >= 64:
        layers = [
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),           nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),           nn.ReLU(),
            nn.Flatten(),
        ]
    elif image_size >= 32:
        layers = [
            nn.Conv2d(in_channels, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),           nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
        ]
    else:
        layers = [
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           nn.ReLU(),
            nn.Flatten(),
        ]

    cnn = nn.Sequential(*layers)

    with torch.no_grad():
        out_dim = cnn(torch.zeros(1, in_channels, image_size, image_size)).shape[1]

    return cnn, out_dim


def _normalize_pixels(obs: torch.Tensor) -> torch.Tensor:
    """Accept uint8 tensors or float tensors already in either [0, 1] or [0, 255]."""
    x = obs.float()
    if obs.dtype == torch.uint8 or x.max() > 1.0:
        x = x / 255.0
    return x


# ---------------------------------------------------------------------------
# Policy Networks
# ---------------------------------------------------------------------------

class CNNActorCritic(nn.Module):
    """Actor-Critic with CNN backbone for raw pixel observations (Stage 1).

    Args:
        obs_shape:   (C, H, W) observation shape.
        action_dim:  Number of actions (discrete) or action dimension (continuous).
        continuous:  Whether the action space is continuous.
        hidden_dim:  MLP hidden layer size after CNN.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        continuous: bool = False,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        C, H, W = obs_shape
        self.continuous = continuous

        self.cnn, cnn_out = _build_cnn(C, H)

        self.fc = nn.Sequential(nn.Linear(cnn_out, hidden_dim), nn.ReLU())

        if continuous:
            self.actor_mean    = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        self.critic = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        head = self.actor_mean if self.continuous else self.actor
        nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple:
        """
        Args:
            obs: (B, C, H, W) uint8 or float observations.
        Returns:
            (distribution, value_tensor)
        """
        x = _normalize_pixels(obs)
        feat  = self.fc(self.cnn(x))
        value = self.critic(feat).squeeze(-1)

        if self.continuous:
            mean = self.actor_mean(feat)
            std  = self.actor_log_std.exp().expand_as(mean)
            return Normal(mean, std), value
        else:
            return Categorical(logits=self.actor(feat)), value


class LatentActorCritic(nn.Module):
    """Actor-Critic that operates on LeWM latent embeddings (Stages 2 & 3).

    Args:
        latent_dim:  Dimension of LeWM latent space D.
        action_dim:  Action space dimension.
        continuous:  Whether the action space is continuous.
        hidden_dim:  MLP hidden layer size.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        continuous: bool = False,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.continuous = continuous

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
        )

        if continuous:
            self.actor_mean    = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        self.critic = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        head = self.actor_mean if self.continuous else self.actor
        nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, z: torch.Tensor) -> tuple:
        """
        Args:
            z: (B, latent_dim) latent embeddings.
        Returns:
            (distribution, value_tensor)
        """
        feat  = self.trunk(z.float())
        value = self.critic(feat).squeeze(-1)

        if self.continuous:
            mean = self.actor_mean(feat)
            std  = self.actor_log_std.exp().expand_as(mean)
            return Normal(mean, std), value
        else:
            return Categorical(logits=self.actor(feat)), value


# ---------------------------------------------------------------------------
# PPO Update
# ---------------------------------------------------------------------------

class PPO:
    """Proximal Policy Optimization with GAE, value clipping, and entropy bonus.

    Args:
        actor_critic:   Actor-Critic module.
        lr:             Learning rate.
        n_epochs:       Number of PPO epochs per rollout.
        batch_size:     Mini-batch size.
        clip_eps:       Clipping epsilon ε.
        gamma:          Discount factor γ.
        gae_lambda:     GAE λ.
        vf_coef:        Value function loss coefficient.
        ent_coef:       Entropy bonus coefficient.
        max_grad_norm:  Gradient clipping norm.
    """

    def __init__(
        self,
        actor_critic: nn.Module,
        lr: float = 3e-4,
        n_epochs: int = 4,
        batch_size: int = 64,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        obs_to_policy_input: Callable[[torch.Tensor], torch.Tensor] | None = None,
        shared_modules: Iterable[nn.Module] | None = None,
    ) -> None:
        self.actor_critic = actor_critic
        self.obs_to_policy_input = obs_to_policy_input
        self.shared_modules = list(shared_modules or [])

        params = list(actor_critic.parameters())
        seen = {id(p) for p in params}
        for module in self.shared_modules:
            for param in module.parameters():
                if id(param) not in seen:
                    params.append(param)
                    seen.add(id(param))

        self._trainable_params = params
        self.optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
        self.n_epochs       = n_epochs
        self.batch_size     = batch_size
        self.clip_eps       = clip_eps
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.vf_coef        = vf_coef
        self.ent_coef       = ent_coef
        self.max_grad_norm  = max_grad_norm

    def _prepare_policy_input(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_to_policy_input is None:
            return obs
        return self.obs_to_policy_input(obs)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation.

        Args:
            rewards:    (T,) rewards.
            values:     (T,) value estimates.
            dones:      (T,) done flags (1=done).
            next_value: () bootstrap value for the state after the last step.

        Returns:
            advantages (T,), returns (T,)
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=rewards.device)

        for t in reversed(range(T)):
            nxt_v    = next_value if t == T - 1 else values[t + 1]
            nxt_done = torch.tensor(0.0, device=rewards.device) if t == T - 1 else dones[t + 1]
            delta    = rewards[t] + self.gamma * nxt_v * (1 - nxt_done) - values[t]
            gae      = delta + self.gamma * self.gae_lambda * (1 - nxt_done) * gae
            advantages[t] = gae

        return advantages, advantages + values

    def update(self, rollout: dict) -> dict[str, float]:
        """Perform PPO update on a rollout.

        Args:
            rollout: dict with keys:
              obs        (T, *obs_shape)   — observations or latents
              actions    (T, ...)          — actions taken
              log_probs  (T,)              — log π(a|s) at collection time
              values     (T,)              — V(s) at collection time
              rewards    (T,)              — rewards received
              dones      (T,)              — done flags
              next_obs   (*obs_shape)      — next observation for bootstrapping

        Returns:
            Dict of scalar training metrics.
        """
        obs          = rollout["obs"]
        actions      = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        old_values   = rollout["values"]
        rewards      = rollout["rewards"]
        dones        = rollout["dones"]

        self.actor_critic.eval()
        for module in self.shared_modules:
            module.eval()

        with torch.no_grad():
            next_obs_input = self._prepare_policy_input(rollout["next_obs"].unsqueeze(0))
            _, next_v = self.actor_critic(next_obs_input)
            next_v    = next_v.squeeze()

        self.actor_critic.train()
        for module in self.shared_modules:
            module.train()

        advantages, returns = self.compute_gae(rewards, old_values, dones, next_v)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = obs.shape[0]
        totals = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0)
        n_updates = 0

        for _ in range(self.n_epochs):
            for idx in torch.randperm(T).split(self.batch_size):
                policy_input = self._prepare_policy_input(obs[idx])
                dist, values = self.actor_critic(policy_input)

                log_probs = dist.log_prob(actions[idx])
                if log_probs.dim() > 1:
                    log_probs = log_probs.sum(-1)
                entropy = dist.entropy()
                if entropy.dim() > 1:
                    entropy = entropy.sum(-1)
                entropy = entropy.mean()

                ratio  = (log_probs - old_log_probs[idx]).exp()
                adv_b  = advantages[idx]
                surr1  = ratio * adv_b
                surr2  = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped   = old_values[idx] + (values - old_values[idx]).clamp(
                    -self.clip_eps, self.clip_eps)
                value_loss  = torch.max(
                    F.mse_loss(values, returns[idx]),
                    F.mse_loss(v_clipped, returns[idx]),
                )

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._trainable_params, self.max_grad_norm)
                self.optimizer.step()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"]  += value_loss.item()
                totals["entropy"]     += entropy.item()
                totals["approx_kl"]   += (old_log_probs[idx] - log_probs).mean().item()
                n_updates += 1

        return {k: v / max(n_updates, 1) for k, v in totals.items()}
