# LeWM Curiosity-Driven RL Framework

**A research-grade framework integrating [LeWorldModel](https://arxiv.org/pdf/2603.19312v1) into online RL as a stable curiosity-driven intrinsic reward generator and state representation encoder.**

---

## Overview

Standard curiosity methods (ICM, RND) suffer from reward collapse and instability. This framework replaces them with **LeWorldModel (LeWM)** — a Joint Embedding Predictive Architecture (JEPA) that learns structured latent representations with provable anti-collapse guarantees via the SIGReg regularizer.

**Core innovation:** LeWM produces a stable, non-collapsing intrinsic reward:

```
r_int = || ẑ_{t+1} - z_{t+1} ||²
```

Because the latent space is regularized to match an isotropic Gaussian (SIGReg), this reward reflects genuine state-level uncertainty rather than pixel noise — unlike ICM/RND.

---

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │          LeWorldModel (LeWM)         │
                          │                                      │
  obs_t (C,H,W) ─────────►  [Encoder: CNN/ViT + BN projector]  │
                          │         z_t ∈ R^D                   │
                          │              │                       │
  action_t ──────────────►  [Predictor: Transformer + AdaLN]   │
                          │         ẑ_{t+1} ∈ R^D              │
                          │                                      │
  L_LeWM = L_pred + λ·SIGReg(Z)                                │
  └─ L_pred  = ||ẑ_{t+1} - z_{t+1}||²                          │
  └─ SIGReg  = Epps-Pulley test on random projections           │
                          └─────────────────────────────────────┘
                                        │
                             r_int = ||ẑ_{t+1} - z_{t+1}||²
                                        │
                          ┌─────────────▼────────────────────────┐
                          │           PPO Agent                   │
                          │   Stage 1: raw pixels → CNN → policy  │
                          │   Stage 2: z_t → linear → policy      │
                          │   Stage 3: z_t + r_int → policy       │
                          └───────────────────────────────────────┘
```

---

## 3-Stage Ablation

| Stage | Policy Input | Intrinsic Reward | Goal |
|-------|-------------|-----------------|------|
| **Stage 1** | Raw pixels (CNN) | ✅ `λ · ‖ẑ−z‖²` (detached) | Prove stable curiosity |
| **Stage 2** | Latent `z_t` | ❌ None | Prove better representations |
| **Stage 3** | Latent `z_t` | ✅ `λ · ‖ẑ−z‖²` (joint) | Show synergy |

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo>
cd lewm_rl

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install WandB for experiment tracking
pip install wandb
```

---

## Running Experiments

### Stage 1 — Intrinsic Reward Only

```bash
python scripts/train.py --config configs/stage1.yaml
```

### Stage 2 — Representation Only

```bash
python scripts/train.py --config configs/stage2.yaml
```

### Stage 3 — Full Model (Synergy)

```bash
python scripts/train.py --config configs/stage3.yaml
```

### CLI overrides

```bash
# Change environment, steps, seed, device:
python scripts/train.py --config configs/stage1.yaml \
    --env "MiniGrid-FourRooms-v0" \
    --steps 1000000 \
    --seed 0 \
    --device cuda

# Resume from checkpoint:
python scripts/train.py --config configs/stage3.yaml \
    --resume results/checkpoints/stage3/step_100000.pt
```

---

## Evaluation & Plots

### Evaluate a single checkpoint

```bash
python scripts/evaluate.py \
    --config configs/stage1.yaml \
    --checkpoint results/checkpoints/stage1/best.pt \
    --stage stage1 \
    --n-episodes 20
```

### Compare all 3 stages

```bash
python scripts/evaluate.py --compare \
    --config configs/stage1.yaml \
    --stage1 results/checkpoints/stage1/best.pt \
    --stage2 results/checkpoints/stage2/best.pt \
    --stage3 results/checkpoints/stage3/best.pt
```

### Generate all plots from training logs

```bash
python scripts/evaluate.py --plot-only --log-dir results/logs --plot-dir results/plots
```

Plots saved to `results/plots/`:
- `learning_curves.png` — reward vs. steps for all stages
- `stageN_rewards.png` — extrinsic + intrinsic + LeWM losses
- `stageN_latent_variance.png` — collapse monitor
- `stage_comparison.png` — final performance bar chart

---

## Configuration

All hyperparameters live in `configs/stageN.yaml`. Key options:

```yaml
lewm:
  lambda_reg: 0.1        # SIGReg weight (the one effective hyperparameter)
  latent_dim: 256        # Embedding dimension D
  encoder_type: cnn      # "cnn" or "vit"
  predictor_dropout: 0.1 # Important! See paper ablation

intrinsic_reward:
  lambda_int: 0.01       # r_total = r_env + lambda_int * r_int

training:
  total_steps: 500000
  lewm_warmup_steps: 5000  # Random exploration before RL starts
```

---

## Project Structure

```
lewm_rl/
├── configs/
│   ├── stage1.yaml          # Stage 1: Intrinsic reward only
│   ├── stage2.yaml          # Stage 2: Representation only
│   └── stage3.yaml          # Stage 3: Full model
│
├── src/
│   ├── models/
│   │   ├── lewm/
│   │   │   ├── modules.py   # SIGReg, ARPredictor, Attention, AdaLN blocks
│   │   │   └── world_model.py  # LeWorldModel: forward, intrinsic reward, rollout
│   │   └── encoders/
│   │       └── encoder.py   # ViTEncoder, CNNEncoder, TemporalEncoder
│   ├── agents/
│   │   └── ppo.py           # PPO: CNNActorCritic, LatentActorCritic, PPO update
│   ├── rewards/
│   │   └── intrinsic_reward.py  # IntrinsicRewardModule with running normalization
│   ├── training/
│   │   ├── trainer.py       # Trainer: main training loop, checkpoint, warmup
│   │   └── factory.py       # build_stage(): constructs all components from config
│   ├── envs/
│   │   └── wrappers.py      # Gymnasium wrappers: resize, channel-first, frame-stack
│   └── utils/
│       ├── replay_buffer.py # Circular replay buffer with trajectory sampling
│       ├── logger.py        # Logger: WandB + TensorBoard + JSON
│       └── plotting.py      # All plotting functions
│
├── scripts/
│   ├── train.py             # Training entry point
│   └── evaluate.py          # Evaluation + plot generation
│
├── docs/
│   ├── paper_summary.md     # Architecture, math, key findings
│   ├── repo_analysis.md     # Official repo analysis
│   ├── system_design.md     # Component design, data flow, gradient analysis
│   └── math.md              # Full mathematical derivations
│
├── results/
│   ├── plots/               # Generated figures
│   ├── logs/                # Training metrics (JSON + TensorBoard)
│   └── checkpoints/         # Model checkpoints
│
├── experiments/
│   ├── stage1/              # Stage 1 experiment results
│   ├── stage2/              # Stage 2 experiment results
│   └── stage3/              # Stage 3 experiment results
│
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

### Why SIGReg prevents collapse
The standard JEPA prediction loss alone has a trivial solution: map all inputs to the same constant vector. SIGReg forces the latent distribution to match an isotropic Gaussian N(0,I) — which requires the encoder to produce *diverse* representations. The guarantee is proven via the Cramer-Wold theorem.

### Why the projector uses BatchNorm (not LayerNorm)
The ViT backbone ends with LayerNorm, which normalizes per-sample. SIGReg needs cross-sample statistics. The MLP+BN projector provides the right interface where SIGReg can act effectively.

### Why AdaLN-zero for action conditioning
Initializing AdaLN parameters to zero means action conditioning starts at zero influence. The predictor first learns to predict future states from context alone, then gradually incorporates actions — this stabilizes early training.

### Why the intrinsic reward is detached in Stage 1
In Stage 1, LeWM is trained independently. If RL loss gradients flowed into LeWM, the reward signal would be shaped to maximize RL objectives rather than genuinely model world dynamics — defeating the purpose.

---

## Mathematical Summary

| Symbol | Meaning |
|--------|---------|
| `z_t = enc_θ(o_t)` | Latent embedding of observation |
| `ẑ_{t+1} = pred_φ(z_t, a_t)` | Predicted next embedding |
| `L_pred = ‖ẑ_{t+1} − z_{t+1}‖²` | Prediction loss |
| `SIGReg(Z) = (1/M) Σ T(Zu_m)` | Anti-collapse regularizer |
| `T(h) = ∫ w(t)|φ_N(t;h) − e^{-t²/2}|² dt` | Epps-Pulley test statistic |
| `L_LeWM = L_pred + λ·SIGReg(Z)` | Full training loss |
| `r_int = ‖ẑ_{t+1} − z_{t+1}‖²` | Intrinsic reward |

See `docs/math.md` for full derivations.

---

## References

- **LeWorldModel Paper:** Maes et al., "Stable End-to-End JEPA from Pixels", arXiv:2603.19312v1 (2026)
- **SIGReg:** Balestriero & LeCun, "LeJEPA: Provable and Scalable SSL", arXiv:2511.08544 (2025)
- **JEPA:** LeCun, "A Path towards Autonomous Machine Intelligence", OpenReview (2022)
- **PPO:** Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347 (2017)
- **ICM:** Pathak et al., "Curiosity-driven Exploration by Self-Supervised Prediction", ICML (2017)
- **RND:** Burda et al., "Exploration by Random Network Distillation", ICLR (2019)
