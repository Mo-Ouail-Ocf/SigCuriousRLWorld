# Mathematical Formulations

---

## 1. JEPA Framework

**Joint Embedding Predictive Architecture** operates entirely in latent space:

$$z_t = \text{enc}_\theta(o_t) \in \mathbb{R}^D$$
$$\hat{z}_{t+1} = \text{pred}_\phi(z_t, a_t) \in \mathbb{R}^D$$

**Intuition:** Instead of predicting future pixels $o_{t+1}$ (expensive, pixel-level), we predict the *representation* of the future frame $z_{t+1}$. This forces the encoder to learn compact, predictable structure.

**Why this matters:** The representation must capture what is *predictable* given the action — typically the physical state (positions, velocities) rather than irrelevant details (lighting, texture).

---

## 2. Prediction Loss

$$\mathcal{L}_{\text{pred}} = \frac{1}{BT} \sum_{b=1}^B \sum_{t=1}^{T-1} \|\hat{z}_{t+1}^{(b)} - z_{t+1}^{(b)}\|_2^2$$

**Teacher-forcing regime:** The encoder provides the "ground truth" target $z_{t+1}$ at every step, not the predictor's own output. This avoids compounding errors during training.

**Gradient flow:**
- Through $\hat{z}_{t+1}$: gradients flow to $\text{pred}_\phi$ and $z_t \to \text{enc}_\theta$
- Through $z_{t+1}$: gradients flow to $\text{enc}_\theta$ (making it *predictable*)

**Collapse risk:** Minimizing $\mathcal{L}_{\text{pred}}$ alone has a trivial solution: $\text{enc}_\theta(o) = 0\ \forall o$, giving $\mathcal{L}_{\text{pred}} = 0$.

---

## 3. SIGReg — Full Derivation

### 3.1 Goal

Enforce: $p_Z \approx \mathcal{N}(0, I_D)$

The isotropic Gaussian is the **maximum-entropy distribution** on $\mathbb{R}^D$ with bounded covariance. Enforcing it prevents collapse (constant vectors would have zero variance) and encourages full-rank, diverse representations.

### 3.2 Why High-Dimensional Normality Testing is Hard

Classical tests (Shapiro-Wilk, Kolmogorov-Smirnov) are designed for 1D and do not generalize to $D > 5$. Direct multivariate tests scale as $O(D^2)$ or $O(D^3)$.

### 3.3 Cramer-Wold Theorem

> A distribution $P$ on $\mathbb{R}^D$ is uniquely determined by all its 1D marginals $\{P_{u^\top Z} : u \in \mathbb{R}^D\}$.

**Corollary:** $P_Z = \mathcal{N}(0, I) \iff$ all projections $u^\top Z \sim \mathcal{N}(0, 1)$.

This reduces the D-dimensional problem to many 1D problems.

### 3.4 Epps-Pulley Test Statistic

For a 1D sample $h = \{h_1, \ldots, h_N\}$, the **empirical characteristic function (ECF)** is:

$$\phi_N(t; h) = \frac{1}{N} \sum_{n=1}^N e^{i t h_n}$$

The characteristic function of $\mathcal{N}(0, 1)$ is: $\phi_0(t) = e^{-t^2/2}$

The Epps-Pulley statistic measures the L2 distance between ECF and target CF, weighted by a Gaussian window:

$$T(h) = \int_{-\infty}^{\infty} w(t) \left|\phi_N(t; h) - \phi_0(t)\right|^2 dt$$

where $w(t) = e^{-t^2 / 2\lambda^2}$.

Expanding the squared norm:

$$T(h) = \int w(t) \left[(\text{Re}[\phi_N - \phi_0])^2 + (\text{Im}[\phi_N - \phi_0])^2\right] dt$$

$$= \int w(t) \left[\left(\frac{1}{N}\sum_n \cos(th_n) - e^{-t^2/2}\right)^2 + \left(\frac{1}{N}\sum_n \sin(th_n)\right)^2\right] dt$$

Practical computation: trapezoidal quadrature on $[0.2, 4]$ with $T=17$ nodes.

**Why ECF?** Characteristic functions uniquely characterize distributions (like PDFs/CDFs), are smooth even for discrete data, and can be computed in closed form for the Gaussian target.

### 3.5 Full SIGReg Formula

$$\text{SIGReg}(Z) = \frac{1}{M} \sum_{m=1}^M T(h^{(m)})$$

where:
- $Z \in \mathbb{R}^{N \times B \times D}$ (timesteps × batch × embedding dim)
- $u^{(m)} \sim \text{Uniform}(\mathcal{S}^{D-1})$ (random unit vectors)
- $h^{(m)} = Z u^{(m)} \in \mathbb{R}^{N \times B}$ (projected embeddings)
- $M = 1024$ random projections (robust to this choice)

**Convergence guarantee:**

$$\text{SIGReg}(Z) \to 0 \iff p_Z \to \mathcal{N}(0, I_D)$$

### 3.6 Applied Step-Wise

Applied at each timestep independently:

$$\mathcal{L}_{\text{SIGReg}} = \frac{1}{T} \sum_{t=1}^T \text{SIGReg}(Z_t)$$

where $Z_t \in \mathbb{R}^{B \times D}$ is the batch of embeddings at time $t$.

In code:
```python
sigreg_loss = SIGReg(emb.permute(1, 0, 2))  # (T, B, D)
```

---

## 4. Complete LeWM Training Objective

$$\mathcal{L}_{\text{LeWM}} = \underbrace{\|\hat{z}_{t+1} - z_{t+1}\|_2^2}_{\text{prediction loss}} + \lambda \cdot \underbrace{\text{SIGReg}(Z)}_{\text{anti-collapse}}$$

**Hyperparameter:** $\lambda = 0.1$ (default). Works for $\lambda \in [0.01, 0.2]$.

**Training dynamics:**
- SIGReg loss drops quickly in early training (representation quickly spreads to fill the Gaussian)
- Prediction loss decreases steadily throughout
- Both losses converge monotonically (unlike PLDM's 7-term objective)

---

## 5. Intrinsic Reward (Our Extension)

### 5.1 Raw Prediction Error

$$r^{\text{int}}_t = \|\hat{z}_{t+1} - z_{t+1}\|_2^2$$

**Interpretation:** High reward when the world model's prediction is wrong → agent is in a novel or surprising state.

**Why stable?** SIGReg ensures $z_{t+1}$ lies on a structured manifold (Gaussian). Unlike pixel-space ICM, the reward reflects *semantic* surprise (state-level), not pixel-level noise.

### 5.2 Normalized Reward (used in practice)

To keep the intrinsic reward on a consistent scale:

$$r^{\text{int,norm}}_t = \frac{r^{\text{int}}_t - \mu_r}{\sigma_r + \epsilon}$$

where $\mu_r, \sigma_r$ are running mean and std of the intrinsic reward buffer.

### 5.3 Total Reward

$$r^{\text{total}}_t = r^{\text{env}}_t + \lambda_{\text{RL}} \cdot r^{\text{int,norm}}_t$$

where $\lambda_{\text{RL}}$ is the intrinsic reward coefficient (tunable per stage).

### 5.4 Stage-Specific Formulations

**Stage 1** (intrinsic reward only):
$$r^{\text{total}}_t = r^{\text{env}}_t + \lambda \cdot \|\hat{z}_{t+1} - z_{t+1}\|_2^2$$
where LeWM is **detached** (gradients do not flow from reward back to LeWM).

**Stage 2** (latent representation only):
$$\text{input to policy} = z_t = \text{enc}_\theta(o_t), \quad r = r^{\text{env}}_t$$

**Stage 3** (full model):
$$\text{input} = z_t, \quad r = r^{\text{env}}_t + \lambda \cdot \|\hat{z}_{t+1} - z_{t+1}\|_2^2$$
Here LeWM is trained end-to-end jointly with the RL agent.

---

## 6. Planning (Latent MPC)

At test time, the world model is used for **Model Predictive Control**:

$$a^*_{1:H} = \arg\min_{a_{1:H}} \|{\hat{z}_H - z_g}\|_2^2$$

where $\hat{z}_H$ is the latent at the end of an $H$-step rollout and $z_g = \text{enc}_\theta(o_g)$ is the goal embedding.

Solved with **Cross-Entropy Method (CEM)**: iteratively sample action sequences from a Gaussian, evaluate cost via world model rollout, select top-K (elites), update sampling distribution.

---

## 7. Temporal Path Straightening (Emergent Property)

Define latent velocity: $v_t = z_{t+1} - z_t$

Straightness measure:
$$S_{\text{straight}} = \frac{1}{B(T-2)} \sum_{b,t} \frac{\langle v_t^{(b)}, v_{t+1}^{(b)} \rangle}{\|v_t^{(b)}\| \|v_{t+1}^{(b)}\|}$$

LeWM achieves higher temporal straightness than PLDM as an **emergent phenomenon** — the Gaussian constraint indirectly encourages smooth latent trajectories.
