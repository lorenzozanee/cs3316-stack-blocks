"""Online FPO contracts: GAE, clipped policy loss, CFM error, and one real update."""

from __future__ import annotations

import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import torch

from starter.interfaces import ACTION_DIM
from starter.policies import fpo
from starter.policies.diffusion import train_online
from starter.policies.diffusion_model import FlowMatchingPolicy, load_policy

POLICY_CONFIG = {
    "horizon": 3,
    "image_size": 16,
    "inference_steps": 2,
    "n_obs_steps": 2,
    "action_steps": 2,
    "vision_dim": 16,
    "vision_depth": 1,
    "vision_heads": 2,
    "model_dim": 32,
    "depth": 1,
    "num_heads": 2,
    "dropout": 0.0,
}


def _identity_state(step: int = 0) -> np.ndarray:
    state = np.zeros(ACTION_DIM, dtype=np.float32)
    state[0] = 0.01 * step
    state[3:7] = (1.0, 0.0, 0.0, 0.0)
    state[7] = 0.5
    state[11:15] = (1.0, 0.0, 0.0, 0.0)
    state[15] = 0.5
    return state


def _small_policy(bounds=None) -> FlowMatchingPolicy:
    if bounds is None:
        bounds = {
            f"{key}_{side}": [(-1.0 if side == "min" else 1.0)] * ACTION_DIM
            for key in ("state", "action")
            for side in ("min", "max")
        }
    torch.manual_seed(0)
    return FlowMatchingPolicy(**POLICY_CONFIG, bounds=bounds)


def _write_checkpoint(path: Path) -> Path:
    model = _small_policy()
    torch.save({"model": model.state_dict(), "policy": dict(POLICY_CONFIG)}, path)
    return path


def _online_config(root: Path, checkpoint: Path, *, run_stamp=None, **overrides) -> dict:
    options = {
        "checkpoint": str(checkpoint),
        "device": "cpu",
        "updates": 1,
        "rollout_chunks": 4,
        "minibatch_size": 4,
        "microbatch_size": 1,
        "cfm_samples": 2,
        "update_epochs": 1,
        "value_warmup_updates": 0,
        "output_dir": str(root / "online"),
        "checkpoint_every": 1,
    }
    options.update(overrides)
    config = {
        "bench_name": "RoboDojo",
        "data_task": "stack_blocks",
        "env_cfg": "arx_x5",
        "action_type": "ee",
        "env_host": "127.0.0.1",
        "env_port": 5555,
        "num_envs": 2,
        "seed": 0,
        "fpo": options,
    }
    if run_stamp is not None:
        config["run_stamp"] = run_stamp
    return config


class GaeTest(unittest.TestCase):
    def test_masks_cross_env_carry_and_bootstraps_truncation(self):
        env_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        rewards = torch.tensor([1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 13.0, 1.0])
        values = torch.tensor([10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0])
        # env 0 terminates at its third step (next value zeroed by the caller).
        # env 1 is truncated at its fourth step and keeps its bootstrap value.
        next_values = torch.tensor([11.0, 20.0, 10.0, 20.0, 30.0, 20.0, 12.0, 30.0])
        discounts = torch.full((len(rewards),), 0.5)
        traces = torch.full((len(rewards),), 0.5)  # gamma * gae_lambda
        dones = torch.tensor([False, False, False, False, True, False, False, True])

        advantages, returns = fpo.compute_gae(
            rewards, values, next_values, discounts, traces, dones, env_ids,
        )

        expected_advantages = torch.tensor(
            [-3.0, -16.25, 1.0, -14.5, 8.0, -11.0, 9.0, -4.0],
        )
        torch.testing.assert_close(advantages, expected_advantages, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(returns, expected_advantages + values, rtol=1e-6, atol=1e-6)
        self.assertEqual(tuple(advantages.shape), (len(rewards),))


class ClippedPolicyLossTest(unittest.TestCase):
    def _loss(self, new_loss, old_loss, advantage, clip_epsilon=0.2):
        return fpo.clipped_policy_loss(
            torch.as_tensor(new_loss, dtype=torch.float32),
            torch.as_tensor(old_loss, dtype=torch.float32),
            torch.as_tensor(advantage, dtype=torch.float32),
            clip_epsilon,
        )

    def test_equal_losses_have_unit_ratio(self):
        for advantage, expected in ((2.0, -2.0), (-2.0, 2.0)):
            loss, metrics = self._loss([0.0], [0.0], [advantage])
            self.assertAlmostEqual(float(loss), expected, places=6)
            self.assertAlmostEqual(float(metrics["approx_kl"]), 0.0, places=6)
            self.assertAlmostEqual(float(metrics["clip_fraction"]), 0.0, places=6)

    def test_positive_advantage_clips_upward_only(self):
        # ratio 2 is clipped to 1 + 0.2; ratio 0.5 lowers the surrogate unclipped.
        loss_up, metrics_up = self._loss([-math.log(2.0)], [0.0], [1.0])
        self.assertAlmostEqual(float(loss_up), -1.2, places=6)
        self.assertAlmostEqual(float(metrics_up["clip_fraction"]), 1.0, places=6)

        loss_down, metrics_down = self._loss([math.log(2.0)], [0.0], [1.0])
        self.assertAlmostEqual(float(loss_down), -0.5, places=6)
        self.assertAlmostEqual(float(metrics_down["clip_fraction"]), 1.0, places=6)

    def test_negative_advantage_clips_downward_only(self):
        loss_down, metrics_down = self._loss([math.log(2.0)], [0.0], [-1.0])
        self.assertAlmostEqual(float(loss_down), 0.8, places=6)
        self.assertAlmostEqual(float(metrics_down["clip_fraction"]), 1.0, places=6)

        loss_up, metrics_up = self._loss([-math.log(2.0)], [0.0], [-1.0])
        self.assertAlmostEqual(float(loss_up), 2.0, places=6)
        self.assertAlmostEqual(float(metrics_up["clip_fraction"]), 1.0, places=6)
        self.assertTrue(math.isfinite(float(metrics_up["approx_kl"])))

    def test_sequence_losses_broadcast_batch_advantages(self):
        new_losses = torch.zeros((1, 3), dtype=torch.float32)
        old_losses = torch.zeros((1, 3), dtype=torch.float32)
        loss, metrics = self._loss(new_losses, old_losses, [2.0])
        self.assertAlmostEqual(float(loss), -2.0, places=6)
        self.assertAlmostEqual(float(metrics["clip_fraction"]), 0.0, places=6)


class CfmErrorTest(unittest.TestCase):
    def test_matches_flow_matching_error_on_valid_prefix(self):
        bounds = {
            f"state_{side}": [(-1.0 if side == "min" else 1.0)] * ACTION_DIM
            for side in ("min", "max")
        }
        bounds.update(
            {
                "action_min": [-0.5] * ACTION_DIM,
                "action_max": [1.5] * ACTION_DIM,
            }
        )
        policy = _small_policy(bounds).eval()
        images = torch.randint(0, 256, (2, 2, 3, 16, 16), dtype=torch.uint8)
        states = torch.zeros((2, 2, ACTION_DIM))
        states[..., 3] = 1.0
        states[..., 11] = 1.0
        actions = torch.rand((2, 3, ACTION_DIM)) * 2 - 1
        times = torch.tensor([0.2, 0.7])
        noise = torch.randn((2, 3, ACTION_DIM))
        valid_steps = torch.tensor([2, 3])

        with torch.no_grad():
            visual = policy.encode_visual(images)
            error = fpo.cfm_error(
                policy, visual, states, actions, times, noise, valid_steps,
            )
            interpolated = torch.lerp(noise, actions, times[:, None, None])
            velocity = policy(interpolated, times, policy.condition_from_visual(visual, states))
            per_step = (velocity - (actions - noise)).square().mean(dim=-1)
            mask = torch.arange(3)[None] < valid_steps[:, None]
            expected = (per_step * mask).sum(dim=-1)

        self.assertEqual(tuple(error.shape), (2,))
        torch.testing.assert_close(error, expected, rtol=1e-5, atol=1e-6)


class _FakeRemoteEnv:
    """Two scenes: env 0 finishes by success, env 1 by the step limit."""

    instances: ClassVar[list[_FakeRemoteEnv]] = []
    num_envs = 2
    terminate_at = 1
    truncate_at = 4
    success_score = 100.0

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.reset_calls: list[list[bool]] = []
        self.step_calls: list[list[int]] = []
        self.executed: list[tuple[int, int]] = []
        self.steps = [0] * self.num_envs
        self.dones = [False] * self.num_envs
        self.closed = False
        type(self).instances.append(self)

    def reset_batch(self, seeds):
        self.reset_calls.append(list(self.dones))
        self.steps = [0] * self.num_envs
        self.dones = [False] * self.num_envs
        observations = [self._observation(index, 0, reset=True) for index in range(self.num_envs)]
        return observations, [{"seed": seed} for seed in seeds]

    def step_batch(self, actions, active):
        active = [int(index) for index in active]
        self.step_calls.append(active)
        results = []
        for action, index in zip(actions, active):
            if self.dones[index]:
                raise AssertionError(f"stepped env {index} after it finished")
            if len(action) != ACTION_DIM:
                raise AssertionError(f"expected {ACTION_DIM}-D actions, got {len(action)}")
            self.steps[index] += 1
            self.executed.append((index, self.steps[index]))
            terminated = index == 0 and self.steps[index] >= self.terminate_at
            truncated = index == 1 and self.steps[index] >= self.truncate_at
            self.dones[index] = terminated or truncated
            score = self.success_score if terminated else 0.0
            info = {"success": terminated, "external_score": score, "step_id": self.steps[index]}
            results.append((
                self._observation(index, self.steps[index]),
                score,
                terminated,
                truncated,
                info,
            ))
        return results

    def privileged(self, index):
        del index
        return {"blocks": {}}

    def close(self):
        self.closed = True

    @staticmethod
    def _observation(index, step, reset=False):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        # 200 + env marks a reset frame; 50 + 10 * env + step marks a step frame.
        image[..., 0] = 200 + index if reset else 50 + 10 * index + step
        image[..., 1] = 90
        image[..., 2] = 120
        state = _identity_state(step)
        state[1] = 0.1 * index
        return {"image": image, "proprio": state.tolist()}


class OnlineTrainingTest(unittest.TestCase):
    def setUp(self):
        _FakeRemoteEnv.instances.clear()

    def _run(self, root: Path, checkpoint: Path, *, run_stamp=None, **overrides):
        config = _online_config(root, checkpoint, run_stamp=run_stamp, **overrides)
        encoded: list[torch.Tensor] = []
        created: list[FlowMatchingPolicy] = []
        vision_grads: list[str] = []
        gae_inputs: list[dict] = []

        original_init = FlowMatchingPolicy.__init__
        original_encode = FlowMatchingPolicy.encode_visual
        original_gae = fpo.compute_gae
        original_update = fpo.FPOTrainer.update

        def update(trainer, rollout, **kwargs):
            for record in rollout:
                self.assertFalse(record.old_losses.requires_grad)
                self.assertIsNone(record.old_losses.grad_fn)
                self.assertFalse(record.visual.requires_grad)
            return original_update(trainer, rollout, **kwargs)

        def init(module, *args, **kwargs):
            original_init(module, *args, **kwargs)
            created.append(module)
            for name, parameter in module.named_parameters():
                if name.startswith("vision."):
                    parameter.register_post_accumulate_grad_hook(
                        lambda parameter, name=name: vision_grads.append(name)
                    )

        def encode(module, images):
            encoded.append(images.detach().clone())
            return original_encode(module, images)

        def gae(rewards, values, next_values, discounts, traces, dones, env_ids):
            gae_inputs.append(
                {
                    "rewards": rewards.detach().clone(),
                    "values": values.detach().clone(),
                    "next_values": next_values.detach().clone(),
                    "discounts": discounts.detach().clone(),
                    "traces": traces.detach().clone(),
                    "dones": dones.detach().clone(),
                    "env_ids": env_ids.detach().clone(),
                }
            )
            return original_gae(rewards, values, next_values, discounts, traces, dones, env_ids)

        with ExitStack() as stack:
            stack.enter_context(patch.object(FlowMatchingPolicy, "__init__", init))
            stack.enter_context(patch.object(FlowMatchingPolicy, "encode_visual", encode))
            stack.enter_context(patch.object(fpo, "compute_gae", gae))
            stack.enter_context(patch.object(fpo.FPOTrainer, "update", update))
            stack.enter_context(patch("starter.policies.fpo.RemoteEnv", _FakeRemoteEnv))
            result = train_online(config)
        return result, encoded, created, vision_grads, gae_inputs

    def test_one_update_trains_actions_and_freezes_vision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            start = _write_checkpoint(root / "offline.ckpt")
            checkpoint, encoded, created, vision_grads, gae_inputs = self._run(root, start)

            self.assertIsInstance(checkpoint, Path)
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(len(_FakeRemoteEnv.instances), 1)
            env = _FakeRemoteEnv.instances[0]
            self.assertTrue(env.closed)
            # The fake raises if a finished scene is stepped again; also check
            # that neither scene ran past its own episode length.
            lengths = {0: _FakeRemoteEnv.terminate_at, 1: _FakeRemoteEnv.truncate_at}
            for index, step in env.executed:
                self.assertLessEqual(step, lengths[index], f"env {index} ran past its episode")
            self.assertEqual(sorted({index for index, _ in env.executed}), [0, 1])
            self.assertGreaterEqual(len(env.reset_calls), 2, "the scenes never started a second episode")
            for reset in env.reset_calls[1:]:
                self.assertTrue(all(reset), "reset an episode while another scene was still running")

            # The observation history carries the last two frames into the next
            # action chunk: env 1 steps twice per chunk and is truncated after
            # four steps, so its second chunk must be conditioned on steps 1-2.
            histories = [self._frame_steps(images) for images in encoded]
            self.assertTrue(
                any(history == [1, 2] for env_id, history in histories if env_id == 1),
                f"env 1 never conditioned on the history of its first two steps: {histories}",
            )
            for images in encoded:
                self.assertEqual(
                    tuple(images.shape[1:]), (POLICY_CONFIG["n_obs_steps"], 3, 16, 16),
                )

            self.assertTrue(created)
            self.assertEqual(vision_grads, [])
            self._assert_gae_covers_both_outcomes(gae_inputs)

            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for key in ("model", "policy", "value", "actor_optimizer", "value_optimizer", "seed"):
                self.assertIn(key, saved)
            self.assertEqual(saved["seed"], 0)
            self.assertTrue(saved["actor_optimizer"]["param_groups"])
            self.assertTrue(saved["value_optimizer"]["param_groups"])
            self.assertTrue(saved["value"])
            for name in ("return_curve.png", "success_rate.png", "loss_curve.png", "kl_clip.png"):
                plot = checkpoint.parent.parent / "plots" / name
                self.assertTrue(plot.is_file(), name)
                self.assertGreater(plot.stat().st_size, 0, name)
            staged = torch.load(
                checkpoint.with_name("iteration_0001.ckpt"), map_location="cpu", weights_only=True,
            )
            for key in ("model", "policy", "value"):
                self.assertIn(key, staged)
            self.assertNotIn("actor_optimizer", staged)
            self.assertNotIn("value_optimizer", staged)

            offline = torch.load(start, map_location="cpu", weights_only=True)["model"]
            only_vision = [name for name in offline if name.startswith("vision.")]
            self.assertTrue(only_vision)
            for name in only_vision:
                self.assertTrue(torch.equal(offline[name], saved["model"][name]), name)
            changed = [
                name
                for name, tensor in offline.items()
                if not name.startswith("vision.")
                and name in saved["model"]
                and not torch.equal(tensor, saved["model"][name])
            ]
            self.assertTrue(changed, "the online update did not move any action weights")

            policy = load_policy(checkpoint)
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            with torch.no_grad():
                actions = policy.predict_action_chunk(image, _identity_state())
            self.assertEqual(tuple(actions.shape), (POLICY_CONFIG["horizon"], ACTION_DIM))
            self.assertTrue(torch.isfinite(actions).all())

    def test_resume_keeps_optimizer_and_critic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            start = _write_checkpoint(root / "offline.ckpt")
            first, *_ = self._run(root, start, run_stamp="first")
            second_config = _online_config(
                root, start, run_stamp="resumed", resume=str(first), updates=2,
            )
            captured: list[dict] = []
            original_gae = fpo.compute_gae

            def gae(rewards, values, next_values, discounts, traces, dones, env_ids):
                captured.append(
                    {
                        "next_values": next_values.detach().clone(),
                        "dones": dones.detach().clone(),
                        "env_ids": env_ids.detach().clone(),
                    }
                )
                return original_gae(rewards, values, next_values, discounts, traces, dones, env_ids)

            with patch("starter.policies.fpo.RemoteEnv", _FakeRemoteEnv), \
                    patch.object(fpo, "compute_gae", gae):
                second = train_online(second_config)
            self.assertIsInstance(second, Path)
            self.assertTrue(second.is_file())
            self.assertEqual(first.parent.parent.name, "first")
            self.assertEqual(second.parent.parent.name, "resumed")
            saved = torch.load(second, map_location="cpu", weights_only=True)
            for key in ("model", "policy", "value", "actor_optimizer", "value_optimizer"):
                self.assertIn(key, saved)
            self.assertEqual(saved["iteration"], 2)
            load_policy(second)

            # The value head is trained during the first run, so the resumed
            # rollout can show that truncated steps keep their bootstrap value
            # while terminated steps are zeroed.
            dones = torch.cat([item["dones"] for item in captured]).bool()
            env_ids = torch.cat([item["env_ids"] for item in captured]).long()
            next_values = torch.cat([item["next_values"] for item in captured])
            terminated = dones & (env_ids == 0)
            truncated = dones & (env_ids == 1)
            self.assertTrue(terminated.any())
            self.assertTrue(truncated.any())
            self.assertTrue(torch.all(next_values[terminated] == 0))
            self.assertTrue(torch.any(next_values[truncated] != 0))

    def test_success_metric_uses_environment_success_signal(self):
        class LaggingScoreEnv(_FakeRemoteEnv):
            success_score = 0.0

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            start = _write_checkpoint(root / "offline.ckpt")
            trainer = fpo.FPOTrainer(_online_config(root, start))
            collector = fpo.RolloutCollector(trainer, LaggingScoreEnv(), 2, 0)
            collector.collect(2)
            self.assertTrue(collector.episodes)
            self.assertEqual(collector.episodes[0]["score"], 0.0)
            self.assertEqual(collector.episodes[0]["success"], 1)

    def test_preflight_reports_without_an_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            start = _write_checkpoint(root / "offline.ckpt")
            config = _online_config(root, start, preflight=True)
            with patch("starter.policies.fpo.RemoteEnv", _FakeRemoteEnv):
                report = train_online(config)
            self.assertIsInstance(report, Path)
            self.assertTrue(report.is_file())
            self.assertEqual(report.name, "preflight.json")
            self.assertFalse((report.parent / "checkpoints" / "last.ckpt").exists())
            for env in _FakeRemoteEnv.instances:
                self.assertEqual(env.reset_calls, [])
                self.assertEqual(env.step_calls, [])

    @staticmethod
    def _frame_steps(images: torch.Tensor) -> tuple[int, list[int]]:
        frames = [int(value) for value in images[0, :, 0, 0, 0]]
        if all(200 <= frame < 210 for frame in frames):
            return frames[0] - 200, []
        env_id = (frames[0] - 50) // 10
        return env_id, [frame - 50 - 10 * env_id for frame in frames]

    def _assert_gae_covers_both_outcomes(self, gae_inputs):
        if not gae_inputs:
            raise AssertionError("the online update never computed GAE")
        dones = torch.cat([item["dones"] for item in gae_inputs]).bool()
        env_ids = torch.cat([item["env_ids"] for item in gae_inputs]).long()
        for item in gae_inputs:
            for key in ("rewards", "values", "next_values", "discounts", "traces", "dones", "env_ids"):
                self.assertEqual(item[key].ndim, 1, f"{key} must be flat")
        self.assertIn(0, env_ids[dones].tolist())
        self.assertIn(1, env_ids[dones].tolist())


if __name__ == "__main__":
    unittest.main()
