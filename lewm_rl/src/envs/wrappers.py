"""
Environment Wrappers for pixel-based RL.

Provides standardized Gymnasium-compatible wrappers that:
  - Return pixel observations as (C, H, W) float tensors
  - Support frame stacking for temporal context
  - Normalize observations
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from collections import deque


class ResizeObservation(gym.ObservationWrapper):
    """Resize pixel observations to a fixed size.

    Args:
        env:    Base environment.
        size:   Target size (height, width) or scalar for square.
    """

    def __init__(self, env: gym.Env, size: int | tuple[int, int] = 64) -> None:
        super().__init__(env)
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        h, w = size
        c = env.observation_space.shape[-1] if len(env.observation_space.shape) == 3 else 1
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        try:
            import cv2
            return cv2.resize(obs, self.size[::-1], interpolation=cv2.INTER_AREA)
        except ImportError:
            from PIL import Image
            img = Image.fromarray(obs)
            img = img.resize(self.size[::-1])
            return np.array(img)


class ChannelFirst(gym.ObservationWrapper):
    """Convert (H, W, C) observations to (C, H, W) format."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old_shape = env.observation_space.shape
        if len(old_shape) == 3:
            h, w, c = old_shape
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(c, h, w), dtype=np.uint8
            )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3:
            return np.transpose(obs, (2, 0, 1))
        return obs


class GrayscaleObservation(gym.ObservationWrapper):
    """Convert RGB observations to grayscale."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if len(env.observation_space.shape) == 3:
            h, w = env.observation_space.shape[:2]
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(1, h, w), dtype=np.uint8
            )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3 and obs.shape[0] == 3:
            # (C, H, W) → grayscale
            gray = (0.299 * obs[0] + 0.587 * obs[1] + 0.114 * obs[2]).astype(np.uint8)
            return gray[np.newaxis]
        return obs


class FrameStack(gym.Wrapper):
    """Stack N consecutive frames as a single observation.

    Args:
        env:     Base environment.
        n_stack: Number of frames to stack.
    """

    def __init__(self, env: gym.Env, n_stack: int = 4) -> None:
        super().__init__(env)
        self.n_stack = n_stack
        self._frames: deque[np.ndarray] = deque(maxlen=n_stack)

        old_shape = env.observation_space.shape
        if len(old_shape) == 3:
            c, h, w = old_shape
            new_shape = (c * n_stack, h, w)
        else:
            new_shape = (old_shape[0] * n_stack, *old_shape[1:])

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)


def make_pixel_env(
    env_id: str,
    image_size: int = 64,
    grayscale: bool = False,
    frame_stack: int = 1,
    seed: int | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a pixel-based environment with standard wrappers.

    Args:
        env_id:      Gymnasium environment ID.
        image_size:  Target observation size (square).
        grayscale:   Whether to convert to grayscale.
        frame_stack: Number of frames to stack (1 = no stacking).
        seed:        Random seed.
        render_mode: Render mode ('rgb_array' for pixel envs).

    Returns:
        Wrapped Gymnasium environment.
    """
    if env_id.startswith("MiniGrid-"):
        env = make_minigrid_env(env_id=env_id, image_size=image_size, seed=seed)

        if grayscale:
            env = GrayscaleObservation(env)

        if frame_stack > 1:
            env = FrameStack(env, frame_stack)

        return env

    env = gym.make(env_id, render_mode=render_mode or "rgb_array")

    if seed is not None:
        env.reset(seed=seed)

    # Ensure we get pixel observations
    obs_shape = env.observation_space.shape
    if len(obs_shape) < 3:
        # Not a pixel env — use render as observation
        env = gym.wrappers.PixelObservationWrapper(env)

    env = ResizeObservation(env, image_size)
    env = ChannelFirst(env)

    if grayscale:
        env = GrayscaleObservation(env)

    if frame_stack > 1:
        env = FrameStack(env, frame_stack)

    return env


def make_minigrid_env(
    env_id: str = "MiniGrid-Empty-8x8-v0",
    image_size: int = 64,
    seed: int | None = None,
) -> gym.Env:
    """Create a MiniGrid environment (sparse reward, good for curiosity testing).

    Args:
        env_id:     MiniGrid environment ID.
        image_size: Target image size.
        seed:       Random seed.

    Returns:
        Configured environment.
    """
    try:
        import minigrid  # noqa: F401
        from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
    except ImportError:
        raise ImportError("Install minigrid: pip install minigrid")

    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)

    # Convert MiniGrid's dict observation into full RGB pixel observations.
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = ResizeObservation(env, image_size)
    env = ChannelFirst(env)
    return env
