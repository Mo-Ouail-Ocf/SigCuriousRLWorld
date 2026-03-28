# Official Repo Analysis: lucas-maes/le-wm

**URL:** https://github.com/lucas-maes/le-wm  
**License:** MIT  
**Language:** Python 100%

---

## 1. Repository Structure

```
le-wm/
├── assets/            # GIFs/images for README
├── config/            # Hydra YAML configs (train & eval)
│   ├── train/
│   │   ├── lewm.yaml  # Main training config (wandb, model, optimizer)
│   │   └── data/      # Per-environment data configs
│   └── eval/          # Planning evaluation configs
├── jepa.py            # JEPA model class (inference + planning)
├── module.py          # All PyTorch building blocks
├── train.py           # Training entry point (Hydra-based)
├── eval.py            # Planning evaluation entry point
├── utils.py           # Helper utilities
└── README.md
```

---

## 2. Key Modules

### `module.py` — Core Building Blocks

| Class | Description |
|---|---|
| `SIGReg` | The anti-collapse regularizer. Implements Epps-Pulley test with random projections. Buffers: `t` (knots), `phi` (Gaussian CF), `weights` (quadrature). GPU-only (hardcodes `device="cuda"`). |
| `Attention` | Scaled dot-product attention with causal masking. Uses `F.scaled_dot_product_attention`. |
| `ConditionalBlock` | Transformer block with AdaLN-zero conditioning (for predictor action conditioning). |
| `Block` | Standard Transformer block (no conditioning). |
| `Transformer` | Full transformer stack, supports both `Block` and `ConditionalBlock`. |
| `ARPredictor` | Autoregressive predictor. Wraps `Transformer` with positional embeddings, uses `ConditionalBlock`. |
| `Embedder` | Action encoder: Conv1d + 2-layer MLP with SiLU. |
| `MLP` | Simple MLP with optional BatchNorm (used as projector). |
| `FeedForward` | FFN sublayer with GELU and dropout. |

**Critical implementation note:** `SIGReg.forward` hardcodes `device="cuda"` for random projection sampling:
```python
A = torch.randn(proj.size(-1), self.num_proj, device="cuda")
```
We will make this device-agnostic in our codebase.

### `jepa.py` — JEPA World Model

The main model class wrapping encoder, predictor, action encoder, and projectors.

Key methods:
- `encode(info)` — encodes pixel observations into latent embeddings using ViT (CLS token)
- `predict(emb, act_emb)` — one-step prediction
- `rollout(info, action_sequence, history_size=3)` — autoregressive multi-step rollout for planning
- `criterion(info_dict)` — computes MSE between predicted and goal embedding (for MPC)
- `get_cost(info_dict, action_candidates)` — full planning cost computation

The JEPA class itself **does not compute the training loss** — that is handled externally by the `stable-pretraining` framework.

### `train.py` / `eval.py`
- Both use Hydra for config management
- Training delegates to `stable-pretraining` library (not in this repo)
- Evaluation delegates to `stable-worldmodel` library

---

## 3. Dependencies

The repo uses two external libraries maintained by the same team:

| Library | Purpose | Installation |
|---|---|---|
| `stable-pretraining` | Training loop, WandB logging, optimizer setup | `uv pip install stable-worldmodel[train,env]` |
| `stable-worldmodel` | Environment management, planning, evaluation | Same package |

These are **closed-source** frameworks that wrap the core model. We cannot directly reuse them for our RL framework. We must **re-implement the training loop ourselves**.

Other key dependencies:
- `torch` + `torchvision`
- `einops` (for tensor reshaping)
- `transformers` (HuggingFace — for ViT-Tiny encoder)
- `hydra-core` (config management)
- `wandb` (experiment tracking)
- `gymnasium` (RL environments)
- `h5py` (HDF5 dataset format)

---

## 4. What We MUST Reuse vs. Redesign

### Reuse (port directly):
- `SIGReg` implementation from `module.py` (fix device issue)
- `ARPredictor` / `Transformer` / `ConditionalBlock` from `module.py`
- `Embedder` (action encoder) from `module.py`
- `MLP` projector from `module.py`
- `JEPA.encode()` and `JEPA.predict()` logic from `jepa.py`
- Core loss: `pred_loss + λ * sigreg_loss`

### Redesign / Build Fresh:
- **Training loop**: Our own PyTorch training loop (not `stable-pretraining`)
- **RL agent**: PPO implementation with CNN or LeWM encoder
- **Intrinsic reward module**: Compute `||ẑ_{t+1} - z_{t+1}||²` and normalize
- **Environment wrappers**: Gymnasium wrappers for pixel observations
- **Config system**: YAML-based (similar to the repo, but without Hydra complexity — use OmegaConf or PyYAML)
- **3-stage experimental ablation**: Not in the original repo at all
- **Logging**: WandB or TensorBoard, with explicit metric tracking

---

## 5. Key Architectural Decisions to Preserve

1. **BatchNorm in projector** — critical for SIGReg to work. The projector MLP must have BN, not LN.
2. **AdaLN-zero initialization** — predictor action conditioning must be initialized to zeros.
3. **Step-wise SIGReg** — apply SIGReg at each time step independently (not across time):
   ```python
   sigreg_loss = mean(SIGReg(emb.transpose(0, 1)))  # (T, B, D)
   ```
4. **Predictor dropout = 0.1** — significant impact on performance (see paper ablation).
5. **CLS token** as the frame representation (not mean of patch tokens).
6. **History size = 3** for predictor context window (except TwoRoom which uses 1).

---

## 6. Our Adaptations for RL Framework

The original LeWM is **offline + goal-conditioned** (used for MPC planning). We adapt it for **online RL**:

1. **LeWM trained alongside RL agent** (or pre-trained and frozen, depending on stage)
2. **Prediction error as intrinsic reward** instead of planning cost
3. **LeWM encoder provides state representation** to RL policy
4. **Online data collection** via environment interaction
5. **Replay buffer** to train LeWM from collected experience
