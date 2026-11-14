"""Wave collection and residual PPO integration without Isaac or real scores."""

from __future__ import annotations

import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from starter.config import load_config
from starter.interfaces import ACTION_DIM, PARTIAL_SCORE, SUCCESS_SCORE
from starter.policies import act_online
from starter.policies.act_inference import FEATURE_DIM, load_act_runner
from starter.policies.act_model import ACTPolicy
from starter.policies.act_online import collect_wave
from tests.test_act import small_config

ROOT = Path(__file__).resolve().parents[1]


def _observation(scene: int, step: int) -> dict:
    shade = np.array([(scene * 37 + step * 11) % 256, 64, 128], dtype=np.uint8)
    return {
        "image": np.broadcast_to(shade, (8, 8, 3)).copy(),
        "proprio": np.linspace(-0.2, 0.3, ACTION_DIM, dtype=np.float32),
    }


class _ScriptedRunner:
    """Deterministic fake of FrozenACTRunner for collect_wave tests."""

    def __init__(self):
        self.stats = {"action_std": np.full(ACTION_DIM, 0.1, dtype=np.float32)}
        self.reset_calls = 0
        self.calls = []

    def reset(self):
        self.reset_calls += 1

    def predict_base(self, obs_list, env_ids, with_features=False):
        self.calls.append([int(obs["scene"]) for obs in obs_list])
        features = torch.tensor(
            [[float(obs["scene"]), float(obs["step"]), 1.0] for obs in obs_list],
            dtype=torch.float32,
        )
        chunks = []
        for obs in obs_list:
            chunk = np.zeros((1, ACTION_DIM), dtype=np.float32)
            chunk[0, :3] = 0.01 * float(obs["scene"])
            chunk[0, (3, 11)] = 1.0
            chunk[0, (7, 15)] = 0.5
            chunks.append(chunk)
        return chunks, features if with_features else None


class _RecordingActor:
    """Residual actor whose value is the feature sum and whose actions are raw."""

    def __init__(self):
        self.calls = []

    def act(self, features, deterministic=False):
        self.calls.append(
            {
                "features": features.detach().clone(),
                "deterministic": deterministic,
                "values": features.sum(-1).detach().clone(),
            }
        )
        raw = torch.zeros((features.shape[0], ACTION_DIM), dtype=features.dtype)
        log_probs = torch.zeros(features.shape[0], dtype=features.dtype)
        return raw, log_probs, features.sum(-1)


class _ScriptedEnv:
    """One episode per scene with a public score recorded per step."""

    def __init__(self, scripts, scores):
        self.scripts = scripts
        self.scores = scores
        self.reset_calls = []
        self.step_calls = []
        self._steps = {}

    def reset_batch(self, seeds):
        self.reset_calls.append([int(seed) for seed in seeds])
        self._steps = {scene: 0 for scene in range(len(seeds))}
        return [self._observation(scene, 0) for scene in range(len(seeds))], [
            {} for _ in seeds
        ]

    @staticmethod
    def _observation(scene, step):
        return {"scene": int(scene), "step": int(step)}

    def step_batch(self, actions, active):
        self.step_calls.append([int(scene) for scene in active])
        if len(actions) != len(active):
            raise AssertionError("actions and active scenes must align")
        results = []
        for scene in active:
            self._steps[scene] += 1
            step = self._steps[scene]
            script = self.scripts[scene]
            score = float(self.scores[scene][step - 1])
            ended = step >= script["length"]
            terminated = bool(ended and script["terminated"])
            truncated = bool(ended and not script["terminated"])
            results.append(
                (self._observation(scene, step), score, terminated, truncated, {})
            )
        return results


class CollectWaveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_active_subsets_preserve_order_and_never_step_ended_scenes(self):
        scripts = {
            0: {"length": 3, "terminated": True},
            1: {"length": 1, "terminated": False},
            2: {"length": 4, "terminated": False},
            3: {"length": 2, "terminated": True},
        }
        scores = {0: [0.0, 10.0, 100.0], 1: [0.0], 2: [0.0, 5.0, 15.0, 15.0],
                  3: [0.0, 100.0]}
        env = _ScriptedEnv(scripts, scores)
        runner = _ScriptedRunner()
        actor = _RecordingActor()
        scores_before = copy.deepcopy(env.scores)

        batch, episodes = collect_wave(
            env, runner, actor, [0, 1, 2, 3], gamma=0.9, gae_lambda=0.8,
            residual_scale=0.05,
        )

        self.assertEqual(env.reset_calls, [[0, 1, 2, 3]])
        self.assertEqual(env.step_calls, [[0, 1, 2, 3], [0, 2, 3], [0, 2], [2]])
        self.assertEqual(runner.reset_calls, 1)
        # Inactive scenes appear only in the step where they ended.
        self.assertNotIn(1, env.step_calls[1])
        self.assertNotIn(3, env.step_calls[2])
        self.assertNotIn(0, env.step_calls[3])
        # Bootstrap forwards include truncated frames but skip true terminals:
        # scene 1 bootstraps at step 1, scene 3 skips at step 2, scene 0 skips
        # at step 3, and scene 2 bootstraps once more when truncated.
        self.assertEqual(
            runner.calls, [[0, 1, 2, 3], [0, 1, 2, 3], [0, 2], [2], [2]]
        )

        self.assertEqual(len(batch.features), 3 + 1 + 4 + 2)
        scene_column = [int(value) for value in batch.features[:, 0]]
        self.assertEqual(scene_column, [0, 0, 0, 1, 2, 2, 2, 2, 3, 3])
        self.assertEqual(len(batch.actions), len(batch.features))
        for field in ("log_probs", "values", "advantages", "returns"):
            self.assertEqual(len(getattr(batch, field)), len(batch.features))

        self.assertEqual([episode["seed"] for episode in episodes], [1, 3, 0, 2])
        self.assertEqual([episode["length"] for episode in episodes], [1, 2, 3, 4])
        self.assertEqual([episode["final_score"] for episode in episodes],
                         [0.0, 100.0, 100.0, 15.0])
        self.assertEqual([episode["terminated"] for episode in episodes],
                         [False, True, True, False])
        self.assertEqual([episode["truncated"] for episode in episodes],
                         [True, False, False, True])
        self.assertEqual(env.scores, scores_before)

    def test_truncation_bootstraps_from_terminal_observation(self):
        scripts = {
            0: {"length": 1, "terminated": True},
            1: {"length": 1, "terminated": False},
        }
        scores = {0: [SUCCESS_SCORE], 1: [PARTIAL_SCORE]}
        env = _ScriptedEnv(scripts, scores)
        runner = _ScriptedRunner()
        actor = _RecordingActor()

        batch, episodes = collect_wave(
            env, runner, actor, [10, 11], gamma=0.9, gae_lambda=0.8,
            residual_scale=0.05,
        )

        # Terminated scene 0: next value is zero. Truncated scene 1: the value
        # of its terminal observation (feature sum 1 + 1 + 1) is bootstrapped.
        expected_returns = torch.tensor([SUCCESS_SCORE, PARTIAL_SCORE + 0.9 * 3.0])
        expected_advantages = torch.tensor(
            [SUCCESS_SCORE - 1.0, PARTIAL_SCORE + 0.9 * 3.0 - 2.0]
        )
        torch.testing.assert_close(batch.returns, expected_returns)
        torch.testing.assert_close(batch.advantages, expected_advantages)
        self.assertEqual(len(batch.features), 2)

        # Exactly one bootstrap forward, using only the truncated terminal obs.
        self.assertEqual(runner.calls, [[0, 1], [1]])
        self.assertEqual(len(actor.calls), 2)
        self.assertFalse(actor.calls[0]["deterministic"])
        self.assertTrue(actor.calls[1]["deterministic"])
        torch.testing.assert_close(
            actor.calls[1]["features"], torch.tensor([[1.0, 1.0, 1.0]])
        )
        self.assertEqual([episode["seed"] for episode in episodes], [10, 11])
        self.assertEqual([episode["terminated"] for episode in episodes], [True, False])


class _OnlineFakeEnv:
    """Two-step episodes with public scores 0, 15 and 100 only."""

    SCRIPTS = {
        0: {"length": 2, "final": SUCCESS_SCORE, "terminated": True},
        1: {"length": 1, "final": PARTIAL_SCORE, "terminated": True},
        2: {"length": 2, "final": PARTIAL_SCORE, "terminated": False},
        3: {"length": 2, "final": 0.0, "terminated": False},
    }

    def __init__(self):
        self.step_calls = []
        self.scores_seen = []
        self.closed = False
        self._state = {}

    def reset_batch(self, seeds):
        self._state = {}
        for scene, seed in enumerate(seeds):
            script = dict(self.SCRIPTS[int(seed) % len(self.SCRIPTS)])
            self._state[scene] = {"step": 0, "done": False, **script}
        return [_observation(scene, 0) for scene in range(len(seeds))], [
            {} for _ in seeds
        ]

    def step_batch(self, actions, active):
        self.step_calls.append([int(scene) for scene in active])
        if len(actions) != len(active):
            raise AssertionError("actions and active scenes must align")
        results = []
        for scene in active:
            state = self._state[scene]
            if state["done"]:
                raise AssertionError(f"inactive scene {scene} was stepped")
            state["step"] += 1
            state["done"] = state["step"] >= state["length"]
            score = float(state["final"]) if state["done"] else 0.0
            terminated = bool(state["done"] and state["terminated"])
            truncated = bool(state["done"] and not state["terminated"])
            self.scores_seen.append(score)
            results.append(
                (_observation(scene, state["step"]), score, terminated, truncated, {})
            )
        return results

    def close(self):
        self.closed = True


def _online_config(base_checkpoint: Path) -> dict:
    config = load_config(ROOT / "configs/env.yaml")
    config.update(load_config(ROOT / "configs/act_online.yaml"))
    config.update(
        base_checkpoint=str(base_checkpoint),
        device="cpu",
        finetune_updates=2,
        finetune_episodes=2,
        num_envs=2,
        hidden_dim=16,
        lr=1e-2,
        ppo_epochs=2,
        minibatch_size=2,
        target_kl=10.0,
        save_freq=1,
        run_stamp="unit-test",
    )
    return config


class TrainOnlineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _write_base(self, root: Path) -> tuple[Path, dict, dict]:
        config = small_config()
        policy = ACTPolicy(config, pretrained_backbone=False)
        base_state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        directory = root / "base"
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        torch.save(policy.state_dict(), directory / "policy_last.ckpt")
        action_mean = np.zeros(ACTION_DIM, dtype=np.float32)
        action_mean[[3, 11]] = 1.0
        action_mean[[7, 15]] = 0.5
        stats = {
            "qpos_mean": np.zeros(ACTION_DIM, dtype=np.float32),
            "qpos_std": np.ones(ACTION_DIM, dtype=np.float32),
            "action_mean": action_mean,
            "action_std": np.full(ACTION_DIM, 0.1, dtype=np.float32),
        }
        with (directory / "dataset_stats.pkl").open("wb") as stream:
            pickle.dump(stats, stream)
        return directory / "policy_last.ckpt", config, base_state

    def test_two_ppo_updates_freeze_base_and_save_loadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, base_config, base_state = self._write_base(root)
            output = root / "online"
            env = _OnlineFakeEnv()
            config = _online_config(checkpoint)
            real_residual = act_online.ResidualActorCritic
            captured = {}

            def build_residual(**kwargs):
                model = real_residual(**kwargs)
                captured["initial"] = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
                captured["model"] = model
                return model

            with mock.patch.object(act_online, "ResidualActorCritic", side_effect=build_residual), \
                    mock.patch.object(act_online, "RemoteEnv", return_value=env), \
                    mock.patch.object(act_online, "online_output_dir", return_value=output):
                act_online.train_online(config)

            self.assertTrue(env.closed)
            self.assertEqual(env.step_calls, [[0, 1], [0], [0, 1], [0, 1]])
            self.assertEqual(set(env.scores_seen), {0.0, PARTIAL_SCORE, SUCCESS_SCORE})

            records = [
                json.loads(line)
                for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["update"] for record in records], [1, 2])
            self.assertEqual([record["global_step"] for record in records], [3, 7])
            self.assertEqual([record["episodes"] for record in records], [2, 4])
            self.assertAlmostEqual(records[0]["mean_final_score"], (SUCCESS_SCORE + PARTIAL_SCORE) / 2)
            self.assertAlmostEqual(records[1]["mean_final_score"], PARTIAL_SCORE / 2)
            self.assertAlmostEqual(records[0]["success_rate"], 0.5)
            self.assertAlmostEqual(records[1]["success_rate"], 0.0)
            self.assertEqual(len((output / "metrics.csv").read_text().splitlines()), 3)
            for record in records:
                for key in ("policy_loss", "value_loss", "entropy", "approx_kl",
                            "clip_fraction", "updates"):
                    self.assertTrue(np.isfinite(record[key]), (key, record))

            payload = torch.load(output / "policy_last.ckpt", map_location="cpu",
                                 weights_only=True)
            self.assertEqual(payload["format"], "act_residual_ppo_v1")
            self.assertEqual(payload["update"], 2)
            self.assertEqual(payload["global_step"], 7)
            self.assertEqual(payload["base_config"], base_config)
            for key, value in base_state.items():
                torch.testing.assert_close(payload["base_state_dict"][key], value, msg=key)
            for key, value in payload["residual_state_dict"].items():
                self.assertFalse(torch.equal(value, captured["initial"][key]), key)

            runner, residual, residual_scale = load_act_runner(
                output / "policy_last.ckpt", "cpu"
            )
            self.assertIsNotNone(residual)
            self.assertAlmostEqual(residual_scale, config["residual_scale"])
            for key, value in runner.policy.state_dict().items():
                torch.testing.assert_close(value, base_state[key], msg=key)
            for key, value in residual.state_dict().items():
                torch.testing.assert_close(
                    value.cpu(), payload["residual_state_dict"][key], msg=key
                )

            chunks, features = runner.predict_base(
                [_observation(0, 0)], [0], with_features=True
            )
            self.assertEqual(chunks[0].shape, (1, ACTION_DIM))
            self.assertEqual(tuple(features.shape), (1, FEATURE_DIM))
            with torch.no_grad():
                actions, log_probs, values = residual.act(features)
            self.assertEqual(tuple(actions.shape), (1, ACTION_DIM))
            self.assertEqual(tuple(log_probs.shape), (1,))
            self.assertEqual(tuple(values.shape), (1,))


if __name__ == "__main__":
    unittest.main()
