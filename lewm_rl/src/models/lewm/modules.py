"""
Core building blocks for LeWorldModel.

Ported and adapted from the official le-wm repository:
  https://github.com/lucas-maes/le-wm/blob/main/module.py

Key changes from the original:
  - SIGReg is device-agnostic (no hardcoded "cuda")
  - Added type hints and docstrings
  - Minor code style improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# SIGReg — Sketched Isotropic Gaussian Regularizer
# ---------------------------------------------------------------------------

class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer.

    Prevents representation collapse by encouraging embeddings to follow
    an isotropic Gaussian distribution N(0, I).

    Algorithm:
      1. Project embeddings onto M random unit-norm directions (Cramer-Wold).
      2. For each projection, compute the Epps-Pulley test statistic comparing
         the empirical characteristic function to N(0,1)'s characteristic function.
      3. Average over projections.

    By the Cramer-Wold theorem, SIGReg(Z) → 0 ⟺ p_Z → N(0, I).

    Args:
        knots:    Number of quadrature nodes for integral approximation.
        num_proj: Number of random projection directions M.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj

        # Quadrature nodes on [0, 3]
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)

        # Trapezoidal weights
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt

        # Characteristic function of N(0,1): phi_0(t) = exp(-t²/2)
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)           # (knots,)
        self.register_buffer("phi", window)    # (knots,) — Gaussian CF values
        self.register_buffer("weights", weights * window)  # (knots,) — weighted quadrature

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            proj: Tensor of shape (T, B, D) — embeddings at T timesteps,
                  batch size B, embedding dimension D.

        Returns:
            Scalar SIGReg loss value.
        """
        device = proj.device

        # Sample M random unit-norm projection directions: (D, M)
        A = torch.randn(proj.size(-1), self.num_proj, device=device)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True))  # normalize each column

        # Project embeddings: (T, B, M)
        projected = proj @ A

        # Compute empirical characteristic function vs. Gaussian CF
        # x_t shape: (T, B, M, knots)
        x_t = projected.unsqueeze(-1) * self.t  # broadcast over knots

        # Real part: mean_n cos(t * h_n)   →  (T, M, knots)
        cos_mean = x_t.cos().mean(dim=1)
        # Imaginary part: mean_n sin(t * h_n)  →  (T, M, knots)
        sin_mean = x_t.sin().mean(dim=1)

        # Epps-Pulley statistic per projection per timestep
        # err = (cos_mean - phi_0)² + sin_mean²
        err = (cos_mean - self.phi).square() + sin_mean.square()  # (T, M, knots)

        # Integrate with quadrature weights and scale by batch size
        statistic = (err @ self.weights) * proj.size(1)  # (T, M)

        return statistic.mean()  # average over timesteps and projections


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head scaled dot-product attention with causal masking support.

    Args:
        dim:      Input/output dimension.
        heads:    Number of attention heads.
        dim_head: Dimension per head.
        dropout:  Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            x:      (B, T, D) input tensor.
            causal: Whether to apply causal masking.

        Returns:
            (B, T, D) output tensor.
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation.

    Args:
        dim:       Input dimension.
        hidden_dim: Hidden dimension (typically 4x dim).
        dropout:   Dropout rate.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Blocks
# ---------------------------------------------------------------------------

class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero action conditioning.

    Uses Adaptive Layer Normalization (AdaLN-zero) where the modulation
    parameters (shift, scale, gate) are predicted from a conditioning signal c.
    Parameters are initialized to zero so conditioning has no effect at the
    start of training — it grows gradually as training progresses.

    Args:
        dim:      Model dimension.
        heads:    Number of attention heads.
        dim_head: Dimension per head.
        mlp_dim:  FFN hidden dimension.
        dropout:  Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        # Zero-init: no effect at start of training
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) hidden states.
            c: (B, T, D) conditioning signal (action embedding).

        Returns:
            (B, T, D) updated hidden states.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block without conditioning.

    Args:
        dim:      Model dimension.
        heads:    Number of attention heads.
        dim_head: Dimension per head.
        mlp_dim:  FFN hidden dimension.
        dropout:  Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """Full Transformer stack with optional AdaLN conditioning.

    Args:
        input_dim:   Input dimension.
        hidden_dim:  Internal model dimension.
        output_dim:  Output dimension.
        depth:       Number of transformer blocks.
        heads:       Number of attention heads.
        dim_head:    Dimension per head.
        mlp_dim:     FFN hidden dimension.
        dropout:     Dropout rate.
        conditional: Whether to use ConditionalBlock (AdaLN) or standard Block.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        conditional: bool = False,
    ) -> None:
        super().__init__()
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim) if (conditional and input_dim != hidden_dim) else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        )
        self.norm = nn.LayerNorm(hidden_dim)
        block_class = ConditionalBlock if conditional else Block
        self.layers = nn.ModuleList([
            block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.conditional = conditional

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) input tensor.
            c: (B, T, input_dim) conditioning tensor (only for conditional transformer).

        Returns:
            (B, T, output_dim) output tensor.
        """
        x = self.input_proj(x)
        if self.conditional and c is not None:
            c = self.cond_proj(c)
            for block in self.layers:
                x = block(x, c)
        else:
            for block in self.layers:
                x = block(x)
        x = self.norm(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Action Embedder
# ---------------------------------------------------------------------------

class ActionEmbedder(nn.Module):
    """Embeds action sequences into a fixed-dimensional representation.

    Uses a Conv1d + 2-layer MLP architecture.

    Args:
        action_dim:  Dimensionality of the raw action.
        smooth_dim:  Intermediate projection dimension.
        emb_dim:     Output embedding dimension.
        mlp_scale:   Hidden dimension multiplier for MLP.
    """

    def __init__(
        self,
        action_dim: int,
        smooth_dim: int = 64,
        emb_dim: int = 192,
        mlp_scale: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(action_dim, smooth_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(smooth_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, action_dim) action sequence.

        Returns:
            (B, T, emb_dim) action embeddings.
        """
        x = x.float()
        x = x.permute(0, 2, 1)      # (B, action_dim, T)
        x = self.patch_embed(x)     # (B, smooth_dim, T)
        x = x.permute(0, 2, 1)      # (B, T, smooth_dim)
        return self.embed(x)         # (B, T, emb_dim)


# ---------------------------------------------------------------------------
# MLP Projector
# ---------------------------------------------------------------------------

class MLPProjector(nn.Module):
    """MLP projector with BatchNorm (critical for SIGReg).

    The BatchNorm in the projector is essential because:
    - The ViT encoder's last layer uses LayerNorm, which normalizes per-sample
    - SIGReg needs cross-sample statistics, which requires BN

    Args:
        input_dim:  Input dimension (e.g., ViT hidden size).
        output_dim: Output embedding dimension D.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, input_dim) where N = B*T.

        Returns:
            (N, output_dim) projected embeddings.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Autoregressive Predictor
# ---------------------------------------------------------------------------

class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step latent embedding prediction.

    Architecture:
      - Learned positional embeddings
      - Transformer with ConditionalBlock (AdaLN-zero) for action conditioning
      - Causal attention mask (can only attend to past)

    Args:
        num_frames:  Maximum sequence length (history size).
        depth:       Number of transformer blocks.
        heads:       Number of attention heads.
        mlp_dim:     FFN hidden dimension.
        input_dim:   Embedding dimension D.
        hidden_dim:  Internal transformer dimension.
        output_dim:  Output prediction dimension (default: same as input_dim).
        dim_head:    Dimension per head.
        dropout:     Dropout rate (default 0.1, see paper ablation).
        emb_dropout: Dropout on positional embeddings.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 768,
        input_dim: int = 192,
        hidden_dim: int = 384,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or input_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            conditional=True,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) latent embedding history.
            c: (B, T, D) action embedding conditioning.

        Returns:
            (B, T, D) predicted next-step embeddings.
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.emb_dropout(x)
        return self.transformer(x, c)
