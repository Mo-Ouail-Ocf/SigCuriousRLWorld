"""
tests/test_modules.py — Formal test suite for the LeWM RL framework.

Run with:
    pytest tests/test_modules.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lewm_config():
    return dict(
        encoder_type="cnn", latent_dim=32, image_size=32, in_channels=3,
        action_dim=4, history_size=3, max_seq_len=8, lambda_reg=0.1,
        sigreg_num_proj=64, sigreg_knots=17, predictor_depth=2,
        predictor_heads=4, predictor_hidden_dim=64, predictor_mlp_dim=128,
        predictor_dropout=0.1,
    )

@pytest.fixture
def lewm(lewm_config):
    from src.models.lewm.world_model import build_lewm
    return build_lewm(lewm_config)

@pytest.fixture
def obs_seq():
    return torch.randint(0, 255, (2, 5, 3, 32, 32), dtype=torch.uint8)

@pytest.fixture
def act_seq():
    return torch.randn(2, 5, 4)

# ---------------------------------------------------------------------------
# SIGReg Tests
# ---------------------------------------------------------------------------

class TestSIGReg:
    def test_anti_collapse(self):
        """Collapsed representations must score higher than Gaussian ones."""
        from src.models.lewm.modules import SIGReg
        sigreg = SIGReg(knots=17, num_proj=64)
        loss_collapse = sigreg(torch.zeros(4, 8, 32))
        loss_gaussian = sigreg(torch.randn(4, 8, 32))
        assert loss_collapse > loss_gaussian, (
            f"Collapse should have higher loss: {loss_collapse:.3f} vs {loss_gaussian:.3f}")

    def test_output_scalar(self):
        from src.models.lewm.modules import SIGReg
        sigreg = SIGReg(knots=17, num_proj=64)
        out = sigreg(torch.randn(3, 8, 16))
        assert out.shape == (), f"Expected scalar, got {out.shape}"

    def test_device_agnostic(self):
        """SIGReg should work on CPU (no hardcoded 'cuda')."""
        from src.models.lewm.modules import SIGReg
        sigreg = SIGReg(knots=17, num_proj=32)
        Z = torch.randn(2, 4, 16)  # CPU tensor
        loss = sigreg(Z)
        assert loss.device.type == "cpu"

    def test_differentiable(self):
        """SIGReg must be differentiable w.r.t. its input."""
        from src.models.lewm.modules import SIGReg
        sigreg = SIGReg(knots=17, num_proj=32)
        Z = torch.randn(2, 4, 16, requires_grad=True)
        loss = sigreg(Z)
        loss.backward()
        assert Z.grad is not None

# ---------------------------------------------------------------------------
# Encoder Tests
# ---------------------------------------------------------------------------

class TestEncoders:
    def test_cnn_32(self):
        from src.models.encoders.encoder import CNNEncoder
        enc = CNNEncoder(in_channels=3, latent_dim=32, image_size=32)
        z = enc(torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8))
        assert z.shape == (4, 32)

    def test_cnn_64(self):
        from src.models.encoders.encoder import CNNEncoder
        enc = CNNEncoder(in_channels=3, latent_dim=64, image_size=64)
        z = enc(torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8))
        assert z.shape == (2, 64)

    def test_temporal_encoder(self):
        from src.models.encoders.encoder import CNNEncoder, TemporalEncoder
        enc = TemporalEncoder(CNNEncoder(3, 32, 32))
        z = enc(torch.randint(0, 255, (2, 5, 3, 32, 32), dtype=torch.uint8))
        assert z.shape == (2, 5, 32)

    def test_cnn_uint8_normalization(self):
        """Encoder should handle uint8 input and normalize to [0,1]."""
        from src.models.encoders.encoder import CNNEncoder
        enc = CNNEncoder(in_channels=3, latent_dim=16, image_size=32)
        obs_255 = torch.full((2, 3, 32, 32), 255, dtype=torch.uint8)
        obs_0   = torch.zeros((2, 3, 32, 32), dtype=torch.uint8)
        z_255 = enc(obs_255)
        z_0   = enc(obs_0)
        # Different inputs should give different embeddings
        assert not torch.allclose(z_255, z_0)

# ---------------------------------------------------------------------------
# LeWorldModel Tests
# ---------------------------------------------------------------------------

class TestLeWorldModel:
    def test_forward_shapes(self, lewm, obs_seq, act_seq):
        res = lewm(obs_seq, act_seq)
        assert res["embeddings"].shape  == (2, 5, 32)
        assert res["predictions"].shape == (2, 4, 32)

    def test_loss_is_scalar(self, lewm, obs_seq, act_seq):
        res = lewm(obs_seq, act_seq)
        assert res["loss"].shape == ()
        assert res["pred_loss"].shape == ()
        assert res["sigreg_loss"].shape == ()

    def test_loss_positive(self, lewm, obs_seq, act_seq):
        res = lewm(obs_seq, act_seq)
        assert res["pred_loss"].item() >= 0
        assert res["sigreg_loss"].item() >= 0

    def test_full_gradient_flow(self, lewm, obs_seq, act_seq):
        """Every trainable parameter must receive a gradient."""
        res = lewm(obs_seq, act_seq)
        res["loss"].backward()
        missing = [n for n, p in lewm.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert len(missing) == 0, f"No gradient for: {missing}"

    def test_single_frame_encode(self, lewm):
        obs = torch.randint(0, 255, (3, 3, 32, 32), dtype=torch.uint8)
        z = lewm.encode(obs)
        assert z.shape == (3, 32)

    def test_sequence_encode(self, lewm, obs_seq):
        z = lewm.encode(obs_seq)
        assert z.shape == (2, 5, 32)

    def test_intrinsic_reward_shape_nonneg(self, lewm):
        o_t  = torch.randint(0, 255, (3, 3, 32, 32), dtype=torch.uint8)
        a_t  = torch.randn(3, 4)
        o_t1 = torch.randint(0, 255, (3, 3, 32, 32), dtype=torch.uint8)
        r = lewm.compute_intrinsic_reward(o_t, a_t, o_t1)
        assert r.shape == (3,)
        assert (r >= 0).all()

    def test_rollout_shape(self, lewm, obs_seq, act_seq):
        with torch.no_grad():
            z0  = lewm.encode(obs_seq[:, :3])      # (2, 3, 32)
            fut = lewm.rollout(z0, act_seq[:, :4]) # (2, 4, 32)
        assert fut.shape == (2, 4, 32)

    def test_latent_variance_positive(self, lewm, obs_seq, act_seq):
        res = lewm(obs_seq, act_seq)
        var = lewm.get_latent_variance(res["embeddings"])
        assert var.item() > 0, "Latent variance should be positive (no collapse)"

# ---------------------------------------------------------------------------
# Intrinsic Reward Tests
# ---------------------------------------------------------------------------

class TestIntrinsicReward:
    def test_output_shape(self):
        from src.rewards.intrinsic_reward import IntrinsicRewardModule
        ir = IntrinsicRewardModule(lambda_int=0.01)
        r = ir(torch.randn(4, 32), torch.randn(4, 32))
        assert r.shape == (4,)

    def test_total_reward_shape(self):
        from src.rewards.intrinsic_reward import IntrinsicRewardModule
        ir = IntrinsicRewardModule(lambda_int=0.01)
        r_tot, r_int = ir.total_reward(torch.zeros(4), torch.randn(4, 32), torch.randn(4, 32))
        assert r_tot.shape == (4,) and r_int.shape == (4,)

    def test_normalization_running_stats(self):
        """Running mean/std should update over time."""
        from src.rewards.intrinsic_reward import IntrinsicRewardModule
        ir = IntrinsicRewardModule(lambda_int=0.01, normalize=True)
        assert ir.running_mean == 0.0
        for _ in range(10):
            ir(torch.randn(16, 32), torch.randn(16, 32))
        assert ir._count.item() == 160  # 10 * 16 samples
        assert ir.running_mean != 0.0

    def test_stats_keys(self):
        from src.rewards.intrinsic_reward import IntrinsicRewardModule
        ir = IntrinsicRewardModule()
        ir(torch.randn(4, 16), torch.randn(4, 16))
        s = ir.get_stats()
        for key in ["intrinsic_reward/mean", "intrinsic_reward/std",
                    "intrinsic_reward/min", "intrinsic_reward/max"]:
            assert key in s, f"Missing stat key: {key}"

# ---------------------------------------------------------------------------
# PPO Tests
# ---------------------------------------------------------------------------

class TestPPO:
    def test_cnn_actor_discrete_32(self):
        from src.agents.ppo import CNNActorCritic
        ac = CNNActorCritic(obs_shape=(3, 32, 32), action_dim=4)
        d, v = ac(torch.randint(0, 255, (3, 3, 32, 32), dtype=torch.uint8))
        a  = d.sample()
        lp = d.log_prob(a)
        assert v.shape == (3,) and lp.shape == (3,)

    def test_cnn_actor_discrete_64(self):
        from src.agents.ppo import CNNActorCritic
        ac = CNNActorCritic(obs_shape=(3, 64, 64), action_dim=6)
        d, v = ac(torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8))
        assert v.shape == (2,)

    def test_latent_actor_discrete(self):
        from src.agents.ppo import LatentActorCritic
        ac = LatentActorCritic(latent_dim=32, action_dim=4)
        d, v = ac(torch.randn(3, 32))
        assert v.shape == (3,)

    def test_latent_actor_continuous(self):
        from src.agents.ppo import LatentActorCritic
        ac = LatentActorCritic(latent_dim=32, action_dim=2, continuous=True)
        d, v = ac(torch.randn(3, 32))
        a  = d.sample()
        lp = d.log_prob(a).sum(-1)
        assert lp.shape == (3,)

    def test_ppo_update_returns_metrics(self):
        from src.agents.ppo import LatentActorCritic, PPO
        ac  = LatentActorCritic(latent_dim=32, action_dim=4)
        ppo = PPO(ac, lr=3e-4, n_epochs=2, batch_size=16)
        T   = 48
        m   = ppo.update(dict(
            obs=torch.randn(T, 32), actions=torch.randint(0, 4, (T,)),
            log_probs=torch.randn(T), values=torch.randn(T),
            rewards=torch.randn(T), dones=torch.zeros(T),
            next_obs=torch.randn(32)))
        for k in ["policy_loss", "value_loss", "entropy", "approx_kl"]:
            assert k in m, f"Missing metric: {k}"

    def test_ppo_update_handles_discrete_actions_with_extra_dim(self):
        """Trainer stores discrete actions; PPO should treat them as class indices."""
        from src.agents.ppo import LatentActorCritic, PPO
        ac = LatentActorCritic(latent_dim=32, action_dim=4)
        ppo = PPO(ac, lr=3e-4, n_epochs=2, batch_size=16)
        T = 32
        m = ppo.update(dict(
            obs=torch.randn(T, 32),
            actions=torch.randint(0, 4, (T, 1)),
            log_probs=torch.randn(T),
            values=torch.randn(T),
            rewards=torch.randn(T),
            dones=torch.zeros(T),
            next_obs=torch.randn(32),
        ))
        for k in ["policy_loss", "value_loss", "entropy", "approx_kl"]:
            assert np.isfinite(m[k]), f"Metric {k} should be finite, got {m[k]}"

    def test_ppo_update_changes_params(self):
        """PPO update must actually change the parameters."""
        from src.agents.ppo import LatentActorCritic, PPO
        ac    = LatentActorCritic(latent_dim=32, action_dim=4)
        params_before = {n: p.clone() for n, p in ac.named_parameters()}
        ppo = PPO(ac, lr=1e-2, n_epochs=2, batch_size=16)
        T   = 32
        ppo.update(dict(
            obs=torch.randn(T, 32), actions=torch.randint(0, 4, (T,)),
            log_probs=torch.randn(T), values=torch.randn(T),
            rewards=torch.ones(T), dones=torch.zeros(T),
            next_obs=torch.randn(32)))
        changed = any(not torch.allclose(p, params_before[n])
                      for n, p in ac.named_parameters())
        assert changed, "PPO update did not change any parameters"

    def test_ppo_update_changes_shared_encoder_params(self):
        """Stage 2/3 PPO should be able to optimize the shared LeWM encoder."""
        from src.models.lewm.world_model import build_lewm
        from src.agents.ppo import LatentActorCritic, PPO

        lewm = build_lewm(dict(
            encoder_type="cnn", latent_dim=32, image_size=32, in_channels=3,
            action_dim=4, history_size=3, max_seq_len=8, lambda_reg=0.1,
            sigreg_num_proj=32, sigreg_knots=17, predictor_depth=2,
            predictor_heads=4, predictor_hidden_dim=64, predictor_mlp_dim=128,
            predictor_dropout=0.1,
        ))
        ac = LatentActorCritic(latent_dim=32, action_dim=4)
        ppo = PPO(
            ac,
            lr=1e-2,
            n_epochs=2,
            batch_size=16,
            obs_to_policy_input=lewm.encode,
            shared_modules=[lewm.encoder],
        )

        T = 32
        obs = torch.randint(0, 255, (T, 3, 32, 32), dtype=torch.uint8)
        next_obs = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)

        with torch.no_grad():
            dist, values = ac(lewm.encode(obs))
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        params_before = {n: p.clone() for n, p in lewm.encoder.named_parameters()}
        ppo.update(dict(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=torch.ones(T),
            dones=torch.zeros(T),
            next_obs=next_obs,
        ))

        changed = any(
            not torch.allclose(param, params_before[name])
            for name, param in lewm.encoder.named_parameters()
        )
        assert changed, "PPO did not update the shared LeWM encoder"

    def test_compute_gae_respects_terminal_transitions(self):
        """Terminal transitions must not bootstrap from the next state's value."""
        from src.agents.ppo import LatentActorCritic, PPO
        ac = LatentActorCritic(latent_dim=32, action_dim=4)
        ppo = PPO(ac, gamma=0.99, gae_lambda=0.95)

        rewards = torch.tensor([1.0, 2.0])
        values = torch.tensor([0.5, 0.25])
        dones = torch.tensor([0.0, 1.0])
        next_value = torch.tensor(10.0)

        advantages, returns = ppo.compute_gae(rewards, values, dones, next_value)

        expected_last_adv = rewards[1] - values[1]
        expected_first_adv = (
            rewards[0] + ppo.gamma * values[1] - values[0]
            + ppo.gamma * ppo.gae_lambda * expected_last_adv
        )

        assert torch.isclose(advantages[1], expected_last_adv)
        assert torch.isclose(advantages[0], expected_first_adv)
        assert torch.isclose(returns[1], rewards[1])

# ---------------------------------------------------------------------------
# ReplayBuffer Tests
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_sample_shapes(self):
        from src.utils.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(500, (3, 32, 32), 4)
        for _ in range(200):
            buf.add(np.zeros((3,32,32),np.uint8), np.zeros(4,np.float32),
                    0.1, np.zeros((3,32,32),np.uint8), False)
        b = buf.sample(16)
        assert b["obs"].shape    == (16, 3, 32, 32)
        assert b["actions"].shape == (16, 4)
        assert b["rewards"].shape == (16, 1)

    def test_trajectory_sample_shapes(self):
        from src.utils.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(500, (3, 32, 32), 4)
        for _ in range(300):
            buf.add(np.zeros((3,32,32),np.uint8), np.zeros(4,np.float32),
                    0.1, np.zeros((3,32,32),np.uint8), False)
        t = buf.sample_trajectories(4, 8)
        assert t["obs"].shape    == (4, 8, 3, 32, 32)
        assert t["actions"].shape == (4, 8, 4)

    def test_circular_overwrite(self):
        """Buffer should overwrite oldest entries when full."""
        from src.utils.replay_buffer import ReplayBuffer
        cap = 50
        buf = ReplayBuffer(cap, (3, 16, 16), 1)
        for i in range(cap + 20):
            buf.add(np.zeros((3,16,16),np.uint8), np.zeros(1,np.float32),
                    float(i), np.zeros((3,16,16),np.uint8), False)
        assert len(buf) == cap

# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestConfigs:
    @pytest.mark.parametrize("stage", ["stage1", "stage2", "stage3"])
    def test_yaml_loads(self, stage):
        from src.training.factory import load_config
        c = load_config(f"configs/{stage}.yaml")
        assert c["stage"] == stage
        assert "lewm" in c
        assert "training" in c

    def test_stage_differences(self):
        """Stage 2 should have no intrinsic_reward section."""
        from src.training.factory import load_config
        s1 = load_config("configs/stage1.yaml")
        s2 = load_config("configs/stage2.yaml")
        s3 = load_config("configs/stage3.yaml")
        assert "intrinsic_reward" in s1
        assert "intrinsic_reward" not in s2
        assert "intrinsic_reward" in s3

    def test_build_stage_uses_full_discrete_action_space(self, monkeypatch):
        """Discrete PPO heads should match env.action_space.n, not a scalar action encoding."""
        import gymnasium as gym
        from src.training import factory

        class DummyEnv:
            observation_space = gym.spaces.Box(
                low=0, high=255, shape=(3, 32, 32), dtype=np.uint8
            )
            action_space = gym.spaces.Discrete(7)

            def reset(self, seed=None):
                return np.zeros((3, 32, 32), dtype=np.uint8), {}

            def step(self, action):
                return np.zeros((3, 32, 32), dtype=np.uint8), 0.0, False, False, {}

        monkeypatch.setattr(factory, "make_pixel_env", lambda **kwargs: DummyEnv())

        trainer = factory.build_stage({
            "stage": "stage1",
            "device": "cpu",
            "seed": 0,
            "env": {"id": "DummyEnv-v0", "image_size": 32},
            "agent": {"hidden_dim": 64},
            "lewm": {"encoder_type": "cnn", "latent_dim": 32},
            "training": {"total_steps": 8, "rollout_steps": 4},
            "intrinsic_reward": {"lambda_int": 0.01, "normalize": True},
        })

        assert trainer.actor_critic.actor.out_features == 7
