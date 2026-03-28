# LeWorldModel (LeWM) — Paper Summary

**Paper:** "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"  
**Authors:** Lucas Maes*, Quentin Le Lidec*, Damien Scieur, Yann LeCun, Randall Balestriero  
**Venue:** arXiv:2603.19312v1, March 2026

---

## 1. What Problem Does LeWM Solve?

Learning world models end-to-end from raw pixels is attractive because it is fully general (no hand-crafted features). However, standard JEPA training with a prediction loss alone suffers from **representation collapse**: the encoder learns to map all inputs to the same constant vector, trivially minimizing the prediction loss without learning anything useful.

**Prior solutions and their flaws:**

| Method | Anti-collapse Strategy | Drawbacks |
|---|---|---|
| I-JEPA / V-JEPA | EMA + stop-gradient | No principled objective; theory unclear |
| DINO-WM | Frozen DINOv2 encoder | Not end-to-end; bounded by pretraining |
| PLDM | VICReg-style (7-term loss) | Unstable training, 6 hyperparameters, O(n⁶) search |
| ICM | Prediction error as curiosity | Reward collapse / noisy TV problem |
| RND | Novelty via random network | Reward non-stationary, unstable |

LeWM addresses all of these with a **2-term objective** and only **1 effective hyperparameter**.

---

## 2. Architecture

```
Observations o₁:T (B, T, C, H, W)  →  [Encoder]  →  Embeddings z₁:T (B, T, D)
                                                            ↓
                          Actions a₁:T  →  [Predictor]  →  ẑ₂:T+1 (B, T, D)
```

### Encoder
- **Architecture:** ViT-Tiny (patch size 14, 12 layers, 3 heads, hidden dim 192, ~5M params)
- Processes flattened `(B*T, C, H, W)` → CLS token → MLP projection with **BatchNorm**
- **Why BatchNorm?** The final ViT layer uses LayerNorm, which prevents SIGReg from being optimized effectively. The projection with BN is the interface where SIGReg acts.
- Output: `z_t ∈ R^D` per frame (D=192 default)

### Predictor
- **Architecture:** Transformer (ViT-Small: 6 layers, 16 heads, 10% dropout, ~10M params)
- Conditions on actions via **Adaptive Layer Normalization (AdaLN)** at each block
- AdaLN parameters initialized to zero → gradual action influence during training
- Takes history of N embeddings, outputs next-step prediction auto-regressively with causal masking
- Followed by a projector (MLP + BN) identical to the encoder projector

### Action Encoder
- Small Embedder MLP: Conv1d + 2-layer MLP → action embedding

### Total Parameters: ~15M, trainable on a single GPU in hours

---

## 3. Training Pipeline

**Offline, reward-free setting.** LeWM is trained purely from `(observations, actions)` trajectory data — no rewards, no task labels.

```python
# Algorithm 1 (simplified)
emb = encoder(obs)               # (B, T, D)
next_emb = predictor(emb, acts)  # (B, T, D)
pred_loss = MSE(emb[:, 1:], next_emb[:, :-1])
sigreg_loss = SIGReg(emb.transpose(0, 1))   # (T, B, D)
loss = pred_loss + λ * sigreg_loss
```

Key properties:
- **No stop-gradient** — full backprop through all components
- **No EMA** — a single network, not a teacher-student pair  
- **No reconstruction loss** — purely predictive in latent space
- Teacher-forcing: encoder provides the ground-truth `z_{t+1}` as regression target

---

## 4. Loss Functions

### 4.1 Prediction Loss (L_pred)

$$\mathcal{L}_{\text{pred}} = \|\hat{z}_{t+1} - z_{t+1}\|_2^2$$

where $\hat{z}_{t+1} = \text{pred}_\phi(z_t, a_t)$ and $z_{t+1} = \text{enc}_\theta(o_{t+1})$.

The encoder is trained jointly, receiving gradients from both the prediction loss (making representations predictable) and SIGReg (making them diverse).

### 4.2 SIGReg — Sketched Isotropic Gaussian Regularizer

SIGReg enforces that the marginal distribution of embeddings matches an **isotropic Gaussian** $\mathcal{N}(0, I)$.

**Why Gaussian?** It is the maximum-entropy distribution under a variance constraint — it occupies all dimensions without collapsing.

**The challenge:** Testing normality in high dimensions (D=192+) is hard — classical tests fail at high d.

**Solution:** Cramer-Wold theorem + Epps-Pulley test statistic.

**Step 1: Random projections (Cramer-Wold)**

$$h^{(m)} = Z u^{(m)}, \quad u^{(m)} \in \mathcal{S}^{D-1}$$

Project the embedding batch $Z \in \mathbb{R}^{N \times B \times D}$ onto M random unit-norm directions.

By Cramer-Wold: if ALL 1D marginals match, the joint distribution matches.

**Step 2: Epps-Pulley test statistic per projection**

$$T^{(m)} = \int_{-\infty}^{\infty} w(t) \left| \phi_N(t; h^{(m)}) - \phi_0(t) \right|^2 dt$$

where:
- $\phi_N(t; h) = \frac{1}{N}\sum_{n=1}^N e^{ith_n}$ is the **empirical characteristic function (ECF)**
- $\phi_0(t) = e^{-t^2/2}$ is the characteristic function of $\mathcal{N}(0,1)$
- $w(t) = e^{-t^2/2\lambda^2}$ is a weighting function
- Integral computed via trapezoidal quadrature on $[0, 4]$ with 17 knots

This measures how far the 1D projection distribution is from a standard Gaussian in frequency space.

**Step 3: Aggregate**

$$\text{SIGReg}(Z) = \frac{1}{M} \sum_{m=1}^M T^{(m)}$$

Applied **step-wise**: for each time-step $t$, compute SIGReg on the batch $Z_t \in \mathbb{R}^{B \times D}$.

### 4.3 Combined Objective

$$\mathcal{L}_{\text{LeWM}} = \mathcal{L}_{\text{pred}} + \lambda \cdot \text{SIGReg}(Z)$$

Only **one effective hyperparameter**: $\lambda$ (default 0.1, works well for $\lambda \in [0.01, 0.2]$).

---

## 5. Why Standard Curiosity Methods Fail

### ICM (Intrinsic Curiosity Module)
- Uses prediction error in feature space as curiosity reward
- **Noisy TV problem**: the reward is high for unpredictable noise (TV screens, random textures), causing agents to fixate on these rather than explore meaningfully
- Feature collapse: if the encoder collapses, the prediction error becomes meaningless
- Uses next-state prediction + inverse dynamics — the inverse dynamics objective can be gamed

### RND (Random Network Distillation)
- Uses the prediction error of a trained network vs. a fixed random target network
- **Non-stationarity**: as the trained network improves, rewards decrease even for genuinely novel states, leading to unstable training signals
- The reward scale decays monotonically, making long-term exploration difficult
- No structural guarantee that the representation captures meaningful structure

### Why LeWM Avoids These Problems
1. **Stable latent space**: SIGReg ensures $p(z) \approx \mathcal{N}(0,I)$, so the embedding space doesn't collapse or over-concentrate
2. **Meaningful prediction error**: Because the latent space is structured (Gaussian, full-rank), $\|\hat{z}_{t+1} - z_{t+1}\|^2$ reflects genuine state uncertainty rather than pixel noise
3. **Anti-collapse guarantee**: The Cramer-Wold argument provides a provable guarantee: SIGReg → 0 ⟺ $p_Z \to \mathcal{N}(0,I)$
4. **Stable training dynamics**: Only 2 loss terms, smooth and monotonic convergence (see paper Fig. 18)

---

## 6. Intrinsic Reward Formulation (Our Extension)

In the original paper, LeWM is used for **planning** (not RL with intrinsic rewards). Our framework **adapts** LeWM as an intrinsic reward generator for online RL in sparse-reward environments.

The intrinsic reward at step $t$ is the **prediction error in latent space**:

$$r^{\text{int}}_t = \|\hat{z}_{t+1} - z_{t+1}\|_2^2$$

The total reward:

$$r^{\text{total}}_t = r^{\text{env}}_t + \lambda_{\text{RL}} \cdot r^{\text{int}}_t$$

This is stable because:
- $z_{t+1}$ is a structured embedding (Gaussian), not raw pixels
- The reward signal measures genuine predictive uncertainty, not pixel noise
- LeWM's SIGReg ensures no collapse → reward stays informative throughout training

---

## 7. Key Findings from the Paper

- LeWM achieves **96% success on Push-T** (vs 78% PLDM, 92% DINO-WM)
- **48× faster planning** than DINO-WM (0.98s vs 47s per plan)
- Latent space encodes **physical structure** (position, velocity) linearly accessible via probes
- **Temporal path straightening** emerges without explicit regularization
- Violation-of-expectation: higher surprise on physically impossible events
- Only 1 hyperparameter (λ) vs 6 for PLDM → efficient bisection search O(log n)
