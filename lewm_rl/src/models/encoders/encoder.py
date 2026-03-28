"""
LeWM Encoder — ViT-based or CNN-based observation encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from src.models.lewm.modules import MLPProjector


def _normalize_pixels(obs: torch.Tensor) -> torch.Tensor:
    """Accept uint8 tensors or float tensors already in either [0, 1] or [0, 255]."""
    x = obs.float()
    if obs.dtype == torch.uint8 or x.max() > 1.0:
        x = x / 255.0
    return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder (ViT-Tiny, matches paper architecture)."""

    def __init__(self, latent_dim: int = 192, image_size: int = 64, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from transformers import ViTConfig, ViTModel
        except ImportError:
            raise ImportError("pip install transformers")

        config = ViTConfig(
            hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
            intermediate_size=768, patch_size=14 if image_size >= 56 else 8,
            image_size=image_size,
        )
        self.vit = ViTModel(config)
        self.projector = MLPProjector(192, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = _normalize_pixels(obs)
        output = self.vit(pixel_values=obs, interpolate_pos_encoding=True)
        cls_token = output.last_hidden_state[:, 0]
        return self.projector(cls_token)


class CNNEncoder(nn.Module):
    """Lightweight CNN encoder for fast experimentation.

    Uses adaptive kernel sizes to support different image resolutions.
    For 64x64: 4x4 kernels with stride 2 (Dreamer-style).
    For 32x32: 3x3 kernels with stride 2 + padding (avoids size underflow).

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB).
        latent_dim:  Output embedding dimension D.
        image_size:  Input image spatial size (height = width assumed).
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 256, image_size: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        if image_size >= 64:
            cnn_layers = [
                nn.Conv2d(in_channels, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),           nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2),          nn.ReLU(),
                nn.Conv2d(128, 256, 4, stride=2),         nn.ReLU(),
            ]
        else:
            # Small images: use 3x3 kernels with padding to avoid underflow
            cnn_layers = [
                nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),            nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),           nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),          nn.ReLU(),
            ]

        self.cnn = nn.Sequential(*cnn_layers, nn.Flatten())

        # Compute CNN output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            cnn_out = self.cnn(dummy).shape[1]

        self.projector = MLPProjector(cnn_out, latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (N, C, H, W) pixel observations.
        Returns:
            (N, latent_dim) latent embeddings.
        """
        obs = _normalize_pixels(obs)
        return self.projector(self.cnn(obs))


class TemporalEncoder(nn.Module):
    """Wraps a frame encoder to handle temporal sequences (B, T, C, H, W)."""

    def __init__(self, frame_encoder: nn.Module) -> None:
        super().__init__()
        self.frame_encoder = frame_encoder
        self.latent_dim = frame_encoder.latent_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, T, C, H, W)
        Returns:
            (B, T, D)
        """
        B, T = obs.shape[:2]
        obs_flat = rearrange(obs, "b t c h w -> (b t) c h w")
        emb_flat = self.frame_encoder(obs_flat)
        return rearrange(emb_flat, "(b t) d -> b t d", b=B, t=T)


def build_encoder(encoder_type: str, latent_dim: int, image_size: int, in_channels: int = 3) -> TemporalEncoder:
    """Factory for building encoders."""
    if encoder_type == "vit":
        frame_enc = ViTEncoder(latent_dim=latent_dim, image_size=image_size)
    elif encoder_type == "cnn":
        frame_enc = CNNEncoder(in_channels=in_channels, latent_dim=latent_dim, image_size=image_size)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")
    return TemporalEncoder(frame_enc)
