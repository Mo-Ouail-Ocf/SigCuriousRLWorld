"""
Factory: builds all components for a given stage from a config dict.

Usage:
    config = load_config("configs/stage1.yaml")
    trainer = build_stage(config)
    trainer.train()
"""

from __future__ import annotations

import yaml
from pathlib import Path

import torch

from src.models.lewm.world_model import build_lewm
from src.agents.ppo import PPO, CNNActorCritic, LatentActorCritic
from src.rewards.intrinsic_reward import IntrinsicRewardModule
from src.utils.replay_buffer import ReplayBuffer
from src.utils.logger import Logger
from src.envs.wrappers import make_pixel_env
from src.training.trainer import Trainer


def load_config(path: str) -> dict:
    """Load a YAML config file.

    Args:
        path: Path to .yaml config file.

    Returns:
        Config dictionary.
    """
    config_path = Path(path)
    if not config_path.is_absolute() and not config_path.exists():
        repo_root_candidate = Path(__file__).resolve().parents[2] / path
        if repo_root_candidate.exists():
            config_path = repo_root_candidate

    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_device(device_name: str | None) -> torch.device:
    """Resolve config device strings, including the common `auto` setting."""
    if device_name in (None, "auto"):
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def build_stage(config: dict, stage: str | None = None) -> Trainer:
    """Build a complete Trainer for the given stage.

    Args:
        config: Full configuration dictionary.
        stage:  Override stage (otherwise read from config["stage"]).

    Returns:
        Configured Trainer instance ready to call .train().
    """
    stage = stage or config.get("stage", "stage1")
    device = _resolve_device(config.get("device"))
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # ---- Environment ----
    env_cfg = config.get("env", {})
    env_id = env_cfg.get("id", "CartPole-v1")
    image_size = env_cfg.get("image_size", 64)
    env = make_pixel_env(
        env_id=env_id,
        image_size=image_size,
        grayscale=env_cfg.get("grayscale", False),
        frame_stack=env_cfg.get("frame_stack", 1),
        seed=seed,
    )

    obs_shape = env.observation_space.shape  # (C, H, W)
    in_channels = obs_shape[0]

    # ---- Action space ----
    if hasattr(env.action_space, "n"):
        action_dim = env.action_space.n
        continuous = False
        act_dim_lewm = 1
    else:
        action_dim = env.action_space.shape[0]
        continuous = True
        act_dim_lewm = action_dim

    # ---- LeWM ----
    lewm_cfg = config.get("lewm", {})
    lewm_cfg["action_dim"] = act_dim_lewm
    lewm_cfg["in_channels"] = in_channels
    lewm_cfg["image_size"] = image_size
    lewm = build_lewm(lewm_cfg)

    # ---- Actor-Critic ----
    agent_cfg = config.get("agent", {})
    latent_dim = lewm_cfg.get("latent_dim", 256)

    if stage == "stage1":
        actor_critic = CNNActorCritic(
            obs_shape=obs_shape,
            action_dim=action_dim,
            continuous=continuous,
            hidden_dim=agent_cfg.get("hidden_dim", 256),
        )
    else:
        actor_critic = LatentActorCritic(
            latent_dim=latent_dim,
            action_dim=action_dim,
            continuous=continuous,
            hidden_dim=agent_cfg.get("hidden_dim", 256),
        )

    # ---- PPO ----
    ppo = PPO(
        actor_critic=actor_critic,
        lr=agent_cfg.get("lr", 3e-4),
        n_epochs=agent_cfg.get("n_epochs", 4),
        batch_size=agent_cfg.get("batch_size", 64),
        clip_eps=agent_cfg.get("clip_eps", 0.2),
        gamma=agent_cfg.get("gamma", 0.99),
        gae_lambda=agent_cfg.get("gae_lambda", 0.95),
        vf_coef=agent_cfg.get("vf_coef", 0.5),
        ent_coef=agent_cfg.get("ent_coef", 0.01),
        max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
        obs_to_policy_input=lewm.encode if stage != "stage1" else None,
        shared_modules=[lewm.encoder] if stage != "stage1" else None,
    )

    # ---- Intrinsic Reward ----
    intrinsic_reward = None
    if stage != "stage2":
        ir_cfg = config.get("intrinsic_reward", {})
        intrinsic_reward = IntrinsicRewardModule(
            lambda_int=ir_cfg.get("lambda_int", 0.01),
            normalize=ir_cfg.get("normalize", True),
        )

    # ---- Replay Buffer ----
    training_cfg = config.get("training", {})
    replay_buffer = ReplayBuffer(
        capacity=training_cfg.get("buffer_capacity", 100_000),
        obs_shape=obs_shape,
        action_dim=act_dim_lewm,
        device=str(device),
    )

    # ---- Logger ----
    run_name = f"{stage}_{env_id.replace('/', '_')}_{seed}"
    logger = Logger(
        run_name=run_name,
        log_dir=training_cfg.get("log_dir", "results/logs"),
        use_wandb=training_cfg.get("use_wandb", False),
        config=config,
        project=training_cfg.get("wandb_project", "lewm-rl"),
    )

    # ---- Trainer ----
    trainer = Trainer(
        stage=stage,
        env=env,
        lewm=lewm,
        actor_critic=actor_critic,
        ppo=ppo,
        intrinsic_reward=intrinsic_reward,
        replay_buffer=replay_buffer,
        logger=logger,
        config=training_cfg,
        device=device,
    )

    return trainer
