"""Residual PPO contracts: GAE, updates, bounded actions, controller, online smoke."""
from __future__ import annotations
import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.testing import assert_close
from starter.interfaces import ACTION_DIM
from starter.policies.diffusion import train_online
from starter.policies.diffusion_model import FlowMatchingPolicy, load_policy
from starter.policies.residual_model import ResidualController, ResidualPolicy
from starter.policies.residual_ppo import (
    PPOOptions, _policy_device, generalized_advantage, ppo_update,
)
POLICY_CONFIG = dict(
    horizon=3, image_size=16, inference_steps=2, n_obs_steps=2, action_steps=2,
    vision_dim=16, vision_depth=1, vision_heads=2, model_dim=16, depth=1,
    num_heads=2, dropout=0.0,
)
LOW = [-0.2, -0.2, -0.1, -1.0, -1.0, -1.0, -1.0, 0.0]
HIGH = [0.2, 0.2, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0]
BOUNDS = dict(state_min=LOW * 2, state_max=HIGH * 2, action_min=LOW * 2, action_max=HIGH * 2)
def _policy():
    return FlowMatchingPolicy(**POLICY_CONFIG, bounds=BOUNDS).eval()
def _residual(hidden_dim=8):
    return ResidualPolicy(_policy(), hidden_dim=hidden_dim, noise_seed=0).eval()
def _wake(model, scale=0.05):
    # Zero-initialized gates make the tiny base observation-independent.
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in model.base.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * scale)
    return model
def _observation(seed=0):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = (17 * (seed + 1)) % 256
    state = np.zeros(ACTION_DIM, dtype=np.float32)
    state[[0, 8]], state[[3, 11]], state[[7, 15]] = 0.01 * seed, 1.0, 0.5
    return {"image": image, "proprio": state}
def _gae(rewards, values, next_values, terminated, truncated, valid, gamma=0.5, lam=0.5):
    def tensor(data, dtype=torch.float32):
        return torch.tensor(data, dtype=dtype)
    return generalized_advantage(
        tensor(rewards), tensor(values), tensor(next_values), tensor(terminated, torch.bool),
        tensor(truncated, torch.bool), tensor(valid, torch.bool), gamma, lam,
    )
class _CPUFastCase(unittest.TestCase):
    def setUp(self): self._threads = torch.get_num_threads(); torch.set_num_threads(1)
    def tearDown(self): torch.set_num_threads(self._threads)
class GeneralizedAdvantageTest(_CPUFastCase):
    def test_terminal_step_does_not_bootstrap(self):
        advantages, returns = _gae(
            [[1.0], [2.0]], [[0.5], [0.5]], [[0.25], [0.75]],
            [[False], [True]], [[False], [False]], [[True], [True]],
        )
        assert_close(advantages.flatten(), torch.tensor([1.0, 1.5]))
        assert_close(returns.flatten(), torch.tensor([1.5, 2.0]))
    def test_truncated_step_bootstraps(self):
        advantages, returns = _gae(
            [[1.0], [2.0]], [[0.5], [0.5]], [[0.25], [0.75]],
            [[False], [False]], [[False], [True]], [[True], [True]],
        )
        assert_close(advantages.flatten(), torch.tensor([1.09375, 1.875]))
        assert_close(returns.flatten(), torch.tensor([1.59375, 2.375]))
    def test_padding_resets_carry_and_ignores_future_garbage(self):
        advantages, returns = _gae(
            [[1.0], [2.0], [1e6]], [[0.5], [0.5], [1e6]], [[0.0], [0.0], [0.0]],
            [[False]] * 3, [[False]] * 3, [[True], [True], [False]],
        )
        assert_close(advantages.flatten(), torch.tensor([0.875, 1.5, 0.0]))
        assert_close(returns.flatten()[:2], torch.tensor([1.375, 2.0]))
class PPOUpdateTest(_CPUFastCase):
    def test_update_moves_residual_heads_and_freezes_base(self):
        base = _policy()
        policy = ResidualPolicy(base, hidden_dim=8, noise_seed=0)
        head, optimizer = policy.head, torch.optim.Adam(policy.head.parameters(), lr=3e-3)
        torch.manual_seed(0)
        steps, envs = 4, 2
        features = torch.randn(steps, envs, policy.feature_dim)
        latents = torch.randn(steps, envs, ACTION_DIM)
        with torch.no_grad():
            distribution = head.distribution(features)
            log_probs = distribution.log_prob(latents).sum(-1)
            values = head.value(features)
        data = {
            "features": features, "latents": latents, "log_probs": log_probs, "values": values,
            "rewards": torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]]),
            "next_values": torch.zeros(steps, envs),
            "terminated": torch.tensor([[False, False], [False, False], [False, False], [True, False]]),
            "truncated": torch.zeros(steps, envs, dtype=torch.bool),
            "valid": torch.ones(steps, envs, dtype=torch.bool),
        }
        before = {
            name: {key: value.detach().clone() for key, value in module.state_dict().items()}
            for name, module in (("actor", head.actor), ("critic", head.critic))
        }
        std_before = head.log_std.detach().clone()
        base_before = {name: value.detach().clone() for name, value in base.state_dict().items()}
        stats = ppo_update(
            head, optimizer, data,
            PPOOptions(offline_checkpoint="unused", ppo_epochs=2, minibatch_size=2,
                       target_kl=0.5, lr=3e-3),
        )
        self.assertGreater(stats["gradient_steps"], 0)
        for name, module in (("actor", head.actor), ("critic", head.critic)):
            self.assertTrue(any(
                not torch.equal(value, before[name][key])
                for key, value in module.state_dict().items()
            ), name)
        self.assertFalse(torch.equal(head.log_std, std_before))
        for name, value in base.state_dict().items():
            self.assertEqual(value.cpu().numpy().tobytes(), base_before[name].cpu().numpy().tobytes(), name)
        self.assertTrue(all(not parameter.requires_grad for parameter in base.parameters()))
class BoundedComposeTest(_CPUFastCase):
    def test_extreme_latents_keep_unit_quaternions_and_grippers(self):
        policy = _residual()
        base_actions = torch.zeros(4, ACTION_DIM)
        base_actions[:, [3, 11, 7, 15]] = torch.tensor([1.0, 1.0, 0.5, 0.5])
        latents = torch.tensor([[1e6] * ACTION_DIM, [-1e6] * ACTION_DIM,
                                [0.0] * ACTION_DIM, [50.0] * ACTION_DIM])
        actions = policy.compose(base_actions, latents)
        self.assertEqual(tuple(actions.shape), (4, ACTION_DIM))
        self.assertTrue(torch.isfinite(actions).all())
        for start in (3, 11):
            quaternion = actions[:, start:start + 4]
            assert_close(quaternion.norm(dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5)
            self.assertTrue((quaternion[:, 0] >= 0).all())
        self.assertTrue(((actions[:, [7, 15]] >= 0) & (actions[:, [7, 15]] <= 1)).all())
class ResidualControllerTest(_CPUFastCase):
    def test_batched_histories_and_chunks_stay_independent(self):
        policy = _residual()
        controller = ResidualController(policy, 2)
        controller.observe([_observation(0), _observation(5)], [0, 1])
        features, actions = controller.prepare([0, 1])
        self.assertEqual(tuple(features.shape), (2, policy.feature_dim))
        self.assertEqual(tuple(actions.shape), (2, ACTION_DIM))
        self.assertFalse(torch.allclose(features[0], features[1]))
        self.assertIsNot(controller.images[0], controller.images[1])
        self.assertIsNot(controller.chunks[0], controller.chunks[1])
        controller.advance([0])
        self.assertEqual(controller.cursors[:2], [1, 0])
        _features, actions = controller.prepare([0, 1])
        assert_close(actions[0], controller.chunks[0][1])
        assert_close(actions[1], controller.chunks[1][0])
    def test_chunk_replans_only_after_action_steps(self):
        policy = _wake(_residual())
        controller = ResidualController(policy, 1)
        controller.observe([_observation(0)], [0])
        controller.prepare([0])
        chunk = controller.chunks[0].clone()
        for cursor in range(1, policy.base.action_steps):
            controller.advance([0])
            controller.observe([_observation(10 + cursor)], [0])
            _features, actions = controller.prepare([0])
            self.assertTrue(torch.equal(controller.chunks[0], chunk))
            assert_close(actions[0], chunk[cursor])
        controller.advance([0])
        controller.observe([_observation(40)], [0])
        _features, actions = controller.prepare([0])
        self.assertFalse(torch.equal(controller.chunks[0], chunk))
        self.assertEqual(controller.cursors[0], 0)
        assert_close(actions[0], controller.chunks[0][0])
def _mapped(mask):
    with patch("torch.cuda.is_available", return_value=True), patch.dict(os.environ, mask, clear=True):
        return str(_policy_device({"policy_gpu": 3}))
class PolicyDeviceTest(_CPUFastCase):
    def test_visible_device_mask_maps_policy_gpu(self):
        self.assertEqual(_mapped({"CUDA_VISIBLE_DEVICES": "3"}), "cuda:0")
        self.assertEqual(_mapped({"CUDA_VISIBLE_DEVICES": "2,3"}), "cuda:1")
        self.assertEqual(_mapped({}), "cuda:3")
class ResidualPolicyTest(_CPUFastCase):
    def test_single_action_feedback_and_deterministic_reset(self):
        policy, observation = _residual(), _observation(3)
        first = policy.predict_action_chunk(observation["image"], observation["proprio"])
        self.assertEqual(tuple(first.shape), (1, ACTION_DIM))
        policy.reset()
        second = policy.predict_action_chunk(observation["image"], observation["proprio"])
        policy.reset()
        third = policy.predict_action_chunk(observation["image"], observation["proprio"])
        assert_close(first, second)
        assert_close(second, third)
class _FakeBatchEnv:
    """Two scenes: index 0 terminates at step 2, index 1 truncates at step 3."""
    instances = []
    def __init__(self, host, port):
        self.host, self.port, self.closed = host, port, False
        self.reset_calls = self.step_calls = 0
        self.observations, self.received_actions, self.last_observation = [], [], None
        self.counts = {}
        type(self).instances.append(self)
    def reset_batch(self, seeds):
        self.reset_calls += 1
        self.counts = {index: 0 for index in range(len(seeds))}
        observations = [_observation(index) for index in range(len(seeds))]
        self.last_observation = observations[-1]
        return observations, [{"seed": seed} for seed in seeds]
    def step_batch(self, actions, active):
        results = []
        for index, action in zip(active, actions):
            self.step_calls += 1
            self.received_actions.append(list(action))
            self.counts[index] += 1
            terminated = index == 0 and self.counts[index] >= 2
            truncated = index == 1 and self.counts[index] >= 3
            score = 100.0 if terminated else 15.0 if truncated else 0.0
            observation = _observation(index + self.counts[index])
            self.observations.append(observation)
            self.last_observation = observation
            results.append((observation, score, terminated, truncated, {"score": score}))
        return results
    def close(self):
        self.closed = True
def _save_offline(path):
    torch.save({"model": _policy().state_dict(), "policy": dict(POLICY_CONFIG)}, path)
def _config(offline_checkpoint, **overrides):
    options = {
        "offline_checkpoint": str(offline_checkpoint), "total_timesteps": 15,
        "rollout_steps": 3, "ppo_epochs": 2, "minibatch_size": 4,
        "hidden_dim": 16, "target_kl": 0.5, "checkpoint_every": 1,
    }
    options.update(overrides)
    return {"seed": 0, "policy_gpu": 0, "num_envs": 2, "env_host": "127.0.0.1",
            "env_port": 1, "residual_ppo": options}
def _run(env, offline, root, **overrides):
    patches = (
        patch("starter.policies.residual_ppo.RemoteEnv", env),
        patch("starter.policies.residual_ppo.dp_checkpoint_dir", return_value=root / "run" / "checkpoints"),
        patch("torch.cuda.is_available", return_value=False),
    )
    with patches[0], patches[1], patches[2]:
        return train_online(_config(offline, **overrides))
PLOT_NAMES = ("return_curve.png", "success_rate.png", "episode_length.png", "loss_curve.png")
def _assert_plots(case, run_dir):
    plots = run_dir / "plots"
    for name in PLOT_NAMES:
        path = plots / name
        case.assertTrue(path.is_file(), name)
        case.assertGreater(path.stat().st_size, 0, name)
        case.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n", name)
class TrainOnlineSmokeTest(_CPUFastCase):
    def test_tiny_flow_matching_train_online(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            offline = root / "offline.ckpt"
            _save_offline(offline)
            _FakeBatchEnv.instances.clear()
            episodes_seen = []

            def spy_ppo_update(*args, **kwargs):
                with (root / "run" / "episodes.csv").open() as handle:
                    episodes_seen.append(len(list(csv.DictReader(handle))))
                return ppo_update(*args, **kwargs)

            output = io.StringIO()
            with patch("starter.policies.residual_ppo.ppo_update", spy_ppo_update), \
                    contextlib.redirect_stdout(output):
                last = _run(_FakeBatchEnv, offline, root)
            console = output.getvalue()
            self.assertEqual(episodes_seen, [2, 4, 6])
            checkpoint_dir = root / "run" / "checkpoints"
            self.assertEqual(last, checkpoint_dir / "last.ckpt")
            self.assertEqual(
                sorted(path.name for path in checkpoint_dir.glob("*.ckpt")),
                ["last.ckpt", "update_000001.ckpt", "update_000002.ckpt", "update_000003.ckpt"],
            )
            with (root / "run" / "metrics.csv").open() as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual([int(row["timesteps"]) for row in metrics], [5, 10, 15])
            for row in metrics:
                for key, value in {
                    "mean_episode_return": 57.5, "rolling_mean_episode_return": 57.5,
                    "mean_episode_length": 2.5, "rolling_mean_episode_length": 2.5,
                    "success_rate": 0.5, "rolling_success_rate": 0.5,
                }.items():
                    self.assertAlmostEqual(float(row[key]), value, msg=key)
            self.assertIn("episode_return=57.5", console)
            self.assertIn("rolling_return=57.5", console)
            self.assertIn("rolling_success_rate=0.5", console)
            saved_config = json.loads((root / "run" / "config.json").read_text())
            self.assertEqual(saved_config["residual_ppo"]["stats_window"], 100)
            _assert_plots(self, root / "run")
            with (root / "run" / "episodes.csv").open() as handle:
                episodes = list(csv.DictReader(handle))
            self.assertEqual(len(episodes), 6)
            self.assertEqual([int(row["timesteps"]) for row in episodes], [4, 5, 9, 10, 14, 15])
            self.assertEqual([float(row["return"]) for row in episodes], [100.0, 15.0] * 3)
            self.assertEqual([row["success"] for row in episodes], ["True", "False"] * 3)
            self.assertEqual([int(row["length"]) for row in episodes], [2, 3] * 3)
            for row in episodes:
                if row["final_score"] == "15.0":
                    self.assertEqual((row["success"], row["truncated"]), ("False", "True"))
                else:
                    self.assertEqual(
                        (row["final_score"], row["success"], row["terminated"]), ("100.0", "True", "True"))
            env = _FakeBatchEnv.instances[-1]
            self.assertTrue(env.closed)
            self.assertEqual(env.step_calls, 15)
            self.assertGreaterEqual(env.reset_calls, 2)
            self.assertEqual(len(env.received_actions), 15)
            self.assertTrue(all(len(action) == ACTION_DIM for action in env.received_actions))
            self.assertIs(env.observations[-1], env.last_observation)
            reloaded = load_policy(last)
            self.assertIsInstance(reloaded, ResidualPolicy)
            saved = torch.load(last, map_location="cpu", weights_only=True)
            offline_saved = torch.load(offline, map_location="cpu", weights_only=True)
            for name, value in offline_saved["model"].items():
                self.assertEqual(reloaded.base.state_dict()[name].cpu().numpy().tobytes(),
                                 value.cpu().numpy().tobytes(), name)
            original = ResidualPolicy(load_policy(offline), **saved["residual_config"]).eval()
            original.head.load_state_dict(saved["residual"])
            original.base_noise.copy_(saved["base_noise"])
            observation = _observation(7)
            loaded = reloaded.predict_action_chunk(observation["image"], observation["proprio"])
            expected = original.predict_action_chunk(observation["image"], observation["proprio"])
            self.assertEqual(tuple(loaded.shape), (1, ACTION_DIM))
            assert_close(loaded, expected)
            reloaded.reset()
            first = reloaded.predict_action_chunk(observation["image"], observation["proprio"])
            reloaded.reset()
            second = reloaded.predict_action_chunk(observation["image"], observation["proprio"])
            assert_close(first, second)
    def test_failure_closes_env(self):
        class _FailingEnv(_FakeBatchEnv):
            def step_batch(self, actions, active):
                raise RuntimeError("step failed")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            offline = root / "offline.ckpt"
            _save_offline(offline)
            _FakeBatchEnv.instances.clear()
            with self.assertRaisesRegex(RuntimeError, "step failed"):
                _run(_FailingEnv, offline, root)
            self.assertTrue(_FakeBatchEnv.instances[-1].closed)
    def test_empty_episode_metrics_are_blank_and_plots_render(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            offline = root / "offline.ckpt"
            _save_offline(offline)
            _FakeBatchEnv.instances.clear()
            _run(_FakeBatchEnv, offline, root, total_timesteps=1)
            with (root / "run" / "episodes.csv").open() as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            with (root / "run" / "metrics.csv").open() as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), 1)
            for key in (
                "mean_episode_return", "mean_episode_length", "success_rate", "mean_max_score",
                "rolling_mean_episode_return", "rolling_success_rate", "rolling_mean_episode_length",
            ):
                self.assertEqual(metrics[0][key], "", key)
            _assert_plots(self, root / "run")
    def test_keyboard_interrupt_writes_plots_and_closes_env(self):
        class _InterruptedEnv(_FakeBatchEnv):
            def step_batch(self, actions, active):
                raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            offline = root / "offline.ckpt"
            _save_offline(offline)
            _FakeBatchEnv.instances.clear()
            with self.assertRaises(KeyboardInterrupt):
                _run(_InterruptedEnv, offline, root)
            self.assertTrue(_FakeBatchEnv.instances[-1].closed)
            _assert_plots(self, root / "run")
if __name__ == "__main__":
    unittest.main()
