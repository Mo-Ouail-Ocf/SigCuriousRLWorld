"""
LeWorldModel (LeWM) — Main world model class.

Combines encoder + predictor + SIGReg for stable end-to-end JEPA training.

Training objective:
  L_LeWM = L_pred + λ * SIGReg(Z)

where:
  L_pred  = MSE(ẑ_{t+1}, z_{t+1})   — prediction loss
  SIGReg  = Gaussian distribution matching regularizer

Reference: "LeWorldModel: Stable End-to-End JEPA from Pixels"
           arXiv:2603.19312v1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.lewm.modules import SIGReg, ARPredictor, ActionEmbedder
from src.models.encoders.encoder import TemporalEncoder, build_encoder


class LeWorldModel(nn.Module):
    """LeWorldModel: Stable JEPA-based World Model.

    Architecture:
      Encoder:   TemporalEncoder (ViT or CNN backbone + MLP+BN projector)
      Predictor: ARPredictor (Transformer with AdaLN action conditioning)
      Loss:      Prediction MSE + SIGReg anti-collapse regularizer

    Args:
        encoder:        TemporalEncoder instance.
        predictor:      ARPredictor instance.
        action_embedder: ActionEmbedder instance.
        sigreg:         SIGReg regularizer instance.
        lambda_reg:     Weight for SIGReg term (λ in paper, default 0.1).
        history_size:   Number of past frames used at inference rollout time.
        max_seq_len:    Maximum sequence length seen during training (for pos. embeddings).
    """

    def __init__(
        self,
        encoder: TemporalEncoder,
        predictor: ARPredictor,
        action_embedder: ActionEmbedder,
        sigreg: SIGReg,
        lambda_reg: float = 0.1,
        history_size: int = 3,
        max_seq_len: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_embedder = action_embedder
        self.sigreg = sigreg
        self.lambda_reg = lambda_reg
        self.history_size = history_size
        self.max_seq_len = max_seq_len
        self.latent_dim = encoder.latent_dim

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent embeddings.

        Args:
            obs: (B, T, C, H, W) sequence  OR  (B, C, H, W) single frame.

        Returns:
            (B, T, D)  or  (B, D) respectively.
        """
        if obs.dim() == 4:
            obs_seq = obs.unsqueeze(1)                    # (B, 1, C, H, W)
            return self.encoder(obs_seq).squeeze(1)       # (B, D)
        return self.encoder(obs)                          # (B, T, D)

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for computing the LeWM training loss.

        Args:
            obs:     (B, T, C, H, W) sequence of observations.
            actions: (B, T, action_dim) corresponding actions.

        Returns:
            Dictionary:
              loss        — total LeWM loss (pred + λ*SIGReg)
              pred_loss   — prediction MSE
              sigreg_loss — anti-collapse regularizer value
              embeddings  — (B, T, D) encoder outputs z_{1:T}
              predictions — (B, T-1, D) predictor outputs ẑ_{2:T}
        """
        # Encode all frames: (B, T, D)
        emb = self.encoder(obs)

        # Encode actions: (B, T, emb_dim)
        act_emb = self.action_embedder(actions)

        # Autoregressive prediction: (B, T, D)
        # Positional embeddings inside ARPredictor handle sequences up to max_seq_len.
        preds = self.predictor(emb, act_emb)

        # ── Prediction loss ─────────────────────────────────────────
        # ẑ_{t+1} = preds[:, :-1]   vs   z_{t+1} = emb[:, 1:]
        # Detach target so gradients flow back through prediction path
        # but NOT doubly back through the target encoder path
        # (standard teacher-forcing; both paths do get gradients from
        #  SIGReg below — so enc receives two gradient paths total)
        pred_loss = F.mse_loss(preds[:, :-1], emb[:, 1:].detach())

        # ── SIGReg ──────────────────────────────────────────────────
        # Apply step-wise: (T, B, D) — one Gaussian test per timestep
        sigreg_loss = self.sigreg(emb.permute(1, 0, 2))

        total_loss = pred_loss + self.lambda_reg * sigreg_loss

        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "embeddings": emb,
            "predictions": preds[:, :-1],
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_intrinsic_reward(
        self,
        obs_t: torch.Tensor,
        action_t: torch.Tensor,
        obs_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample intrinsic reward: r_int = ||ẑ_{t+1} − z_{t+1}||²

        Args:
            obs_t:    (B, C, H, W) current observations.
            action_t: (B, action_dim) current actions.
            obs_t1:   (B, C, H, W) next observations.

        Returns:
            (B,) intrinsic reward per sample.
        """
        z_t  = self.encode(obs_t)    # (B, D)
        z_t1 = self.encode(obs_t1)   # (B, D)

        # Single-step prediction
        z_t_seq  = z_t.unsqueeze(1)                         # (B, 1, D)
        act_seq  = action_t.unsqueeze(1).float()            # (B, 1, action_dim)
        act_emb  = self.action_embedder(act_seq)            # (B, 1, D)
        z_hat_t1 = self.predictor(z_t_seq, act_emb)[:, 0]  # (B, D)

        return F.mse_loss(z_hat_t1, z_t1, reduction="none").mean(dim=-1)  # (B,)

    @torch.no_grad()
    def rollout(
        self,
        z_init: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive latent rollout for planning / evaluation.

        Args:
            z_init:  (B, T_ctx, D) initial latent history (context).
            actions: (B, T_roll, action_dim) future action sequence.

        Returns:
            (B, T_roll, D) predicted future embeddings.
        """
        history  = z_init.clone()
        predicted = []
        T_roll = actions.shape[1]

        for t in range(T_roll):
            hs = min(self.history_size, history.size(1))
            emb_ctx = history[:, -hs:]                              # (B, hs, D)
            act_ctx = actions[:, t : t + 1].float()                # (B, 1, action_dim)
            # Repeat single action to match history length for conditioning
            act_ctx_full = act_ctx.expand(-1, hs, -1)              # (B, hs, action_dim)
            act_emb      = self.action_embedder(act_ctx_full)      # (B, hs, D)

            z_pred = self.predictor(emb_ctx, act_emb)[:, -1:]      # (B, 1, D)
            predicted.append(z_pred)
            history = torch.cat([history, z_pred], dim=1)

        return torch.cat(predicted, dim=1)                          # (B, T_roll, D)

    def get_latent_variance(self, emb: torch.Tensor) -> torch.Tensor:
        """Mean variance of embedding dimensions (collapse monitor).

        Args:
            emb: (B, T, D) or (B, D).

        Returns:
            Scalar mean variance (should stay > ~0.01 during healthy training).
        """
        if emb.dim() == 3:
            emb = emb.reshape(-1, emb.shape[-1])
        return emb.var(dim=0).mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_lewm(config: dict) -> LeWorldModel:
    """Build a LeWorldModel from a config dictionary.

    Args:
        config: Keys used:
          encoder_type, latent_dim, image_size, in_channels,
          action_dim, history_size, max_seq_len, lambda_reg,
          predictor_depth, predictor_heads, predictor_hidden_dim,
          predictor_mlp_dim, predictor_dropout,
          sigreg_num_proj, sigreg_knots.

    Returns:
        Configured LeWorldModel.
    """
    latent_dim   = config.get("latent_dim", 256)
    action_dim   = config.get("action_dim", 4)
    history_size = config.get("history_size", 3)
    # max_seq_len must cover the full trajectory length used in training
    max_seq_len  = config.get("max_seq_len", config.get("lewm_traj_len", 8))

    encoder = build_encoder(
        encoder_type=config.get("encoder_type", "cnn"),
        latent_dim=latent_dim,
        image_size=config.get("image_size", 64),
        in_channels=config.get("in_channels", 3),
    )

    action_embedder = ActionEmbedder(
        action_dim=action_dim,
        smooth_dim=min(64, latent_dim),
        emb_dim=latent_dim,
    )

    predictor = ARPredictor(
        num_frames=max_seq_len,                           # ← covers full training length
        depth=config.get("predictor_depth", 6),
        heads=config.get("predictor_heads", 8),
        mlp_dim=config.get("predictor_mlp_dim", latent_dim * 4),
        input_dim=latent_dim,
        hidden_dim=config.get("predictor_hidden_dim", latent_dim * 2),
        dropout=config.get("predictor_dropout", 0.1),
    )

    sigreg = SIGReg(
        knots=config.get("sigreg_knots", 17),
        num_proj=config.get("sigreg_num_proj", 1024),
    )

    return LeWorldModel(
        encoder=encoder,
        predictor=predictor,
        action_embedder=action_embedder,
        sigreg=sigreg,
        lambda_reg=config.get("lambda_reg", 0.1),
        history_size=history_size,
        max_seq_len=max_seq_len,
    )
