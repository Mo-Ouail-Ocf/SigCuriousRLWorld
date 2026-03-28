# System Design: LeWM Curiosity-Driven RL Framework

---

## 1. Overview

This framework integrates LeWorldModel (LeWM) into an online RL loop as:

- a curiosity-driven intrinsic reward generator
- a state representation encoder for PPO

It supports three experimental stages:

- **Stage 1:** intrinsic reward only
- **Stage 2:** representation only
- **Stage 3:** full joint model

---

## 2. Components

### 2.1 Environment

- **Interface:** Gymnasium environments with pixel observations
- **Wrapper path:**
  - MiniGrid: `RGBImgObsWrapper` -> `ImgObsWrapper` -> resize -> channel-first
  - Non-pixel envs: `PixelObservationWrapper` -> resize -> channel-first
- **Supported envs:** MiniGrid, Atari-style pixel envs, CartPole-style baselines via pixel wrapping
- **Frame stacking:** optional

### 2.2 RL Agent

- **Algorithm:** PPO
- **Stage 1 actor:** CNN policy over raw pixels
- **Stage 2/3 actor:** latent policy over `z_t`
- **Heads:** policy + value

### 2.3 LeWorldModel Module

- **Encoder:** CNN or ViT encoder with MLP + BatchNorm projector
- **Predictor:** autoregressive Transformer with AdaLN action conditioning
- **Loss:** prediction loss + SIGReg
- **Training mode:**
  - Stage 1: trained separately from PPO
  - Stage 2/3: shared with PPO through encoder gradients

### 2.4 Intrinsic Reward Module

- **Signal:** latent prediction error
- **Formula:** `r_int = ||z_hat_{t+1} - z_{t+1}||^2`
- **Normalization:** running mean/std
- **Usage:** active in Stages 1 and 3, off in Stage 2

### 2.5 Buffers

- **Rollout buffer:** on-policy PPO storage
- **Replay buffer:** LeWM training storage
- **Replay contents:** `(obs_t, action_t, reward_t, obs_{t+1}, done_t)`
- **Note:** latents are recomputed on demand during PPO updates in Stages 2 and 3 so RL gradients can reach the encoder

### 2.6 Training Orchestrator

- collects environment interaction
- computes intrinsic reward
- updates PPO
- updates LeWM
- logs metrics and checkpoints

---

## 3. Data Flow

### Stage 3 Pipeline

1. Environment returns pixel observation `obs_t`
2. LeWM encoder produces latent `z_t`
3. PPO policy selects action from `z_t`
4. Environment steps to `obs_{t+1}`
5. LeWM predicts `z_hat_{t+1}` from `(z_t, a_t)`
6. Intrinsic reward is computed from prediction error
7. PPO is updated from rollout data
8. LeWM is updated from replay-buffer trajectories

---

## 4. Gradient Flow

### Stage 1: Intrinsic Reward Only

- PPO policy sees raw pixels
- LeWM is trained on its own loss only
- Intrinsic reward is detached from LeWM optimization

Gradient flow:

- `L_RL` -> PPO only
- `L_LeWM` -> LeWM encoder + predictor
- no RL gradients enter LeWM

### Stage 2: Representation Only

- PPO policy sees latent `z_t`
- PPO update re-encodes stored rollout observations during the update step
- LeWM encoder receives gradients from PPO and from LeWM loss

Gradient flow:

- `L_RL` -> policy head -> LeWM encoder
- `L_LeWM` -> LeWM encoder + predictor

### Stage 3: Full Model

- PPO policy sees latent `z_t`
- intrinsic reward uses LeWM prediction error
- PPO update re-encodes stored rollout observations during the update step
- LeWM encoder is shared between RL and world-model learning

Gradient flow:

- `L_RL` -> policy head -> LeWM encoder
- `L_LeWM` -> LeWM encoder + predictor
- intrinsic reward gradients are stopped before reward shaping can distort LeWM

---

## 5. Training Schedule

### Phase 0: Optional LeWM Warmup

- collect random trajectories
- store them in replay buffer
- pretrain LeWM before PPO starts

### Phase 1: Online Training

- collect rollout with current policy
- compute PPO update every `rollout_steps`
- compute LeWM update from replay-buffer trajectories
- log rewards, losses, and collapse-monitor metrics

---

## 6. Per-Stage Trainability

| Component | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| LeWM encoder | trained by LeWM only | trained by LeWM + PPO | trained by LeWM + PPO |
| LeWM predictor | trained | trained | trained |
| PPO CNN backbone | trained | not used | not used |
| PPO latent head | not used | trained | trained |
| Intrinsic reward | on | off | on |

---

## 7. Reward Computation

```python
with torch.no_grad():
    z_t = encoder(obs_t)
    z_hat_t1 = predictor(z_t, a_t)
    z_t1 = encoder(obs_t1)

r_int_raw = mse(z_hat_t1, z_t1)
r_int = normalize(r_int_raw)

if stage in {"stage1", "stage3"}:
    r_total = r_env + lambda_rl * r_int
else:
    r_total = r_env
```

Implementation notes:

- rollout collection uses `torch.no_grad()` for action sampling
- Stage 2/3 PPO updates run the encoder again on stored raw observations so RL gradients reach LeWM
- final-value bootstrap encoding uses eval mode to avoid BatchNorm issues at batch size 1

---

## 8. Logging

Typical metrics:

- `reward/extrinsic`
- `reward/intrinsic`
- `reward/total`
- `lewm/loss`
- `lewm/pred_loss`
- `lewm/sigreg_loss`
- `lewm/latent_variance`
- `ppo/policy_loss`
- `ppo/value_loss`
- `ppo/entropy`

---

## 9. Validation Status

Current implementation has been smoke-tested on image-based MiniGrid (`MiniGrid-Empty-8x8-v0`) through the pixel wrapper stack, and unit-tested with `35` passing tests.
