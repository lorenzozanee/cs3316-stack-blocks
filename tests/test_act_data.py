"""Contracts for the ACT dataset over raw starter HDF5 episodes."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party"))

import h5py
import numpy as np
import torch

from XPolicyLab.utils.process_data import encode_image_bit
from starter.data.act import ACTDataset, compute_normalization_stats, load_act_data
from starter.data.demos import episode_files
from starter.interfaces import ARM_KEYS, ARM_WIDTHS


def _solid(rgb, height=48, width=64):
    return np.full((height, width, 3), rgb, dtype=np.uint8)


def _write_episode(directory, name, states, actions, frames):
    path = Path(directory) / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for group_name, values in (("state", states), ("action", actions)):
            group = handle.create_group(group_name)
            offset = 0
            for key, width in zip(ARM_KEYS, ARM_WIDTHS):
                group.create_dataset(key, data=values[:, offset : offset + width])
                offset += width
        colors = handle.create_group("vision/cam_head").create_dataset(
            "colors",
            shape=(len(frames),),
            dtype=h5py.special_dtype(vlen=np.dtype("uint8")),
        )
        for index, frame in enumerate(frames):
            colors[index] = np.frombuffer(
                encode_image_bit(frame), dtype=np.uint8
            )
    return path


def _write_bad_episode(directory, name):
    path = Path(directory) / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("state")
        group.create_dataset("left_ee_pose", data=np.zeros((2, 6), dtype=np.float32))
        group.create_dataset(
            "left_ee_joint_state", data=np.zeros((2, 1), dtype=np.float32)
        )
        group.create_dataset("right_ee_pose", data=np.zeros((2, 7), dtype=np.float32))
        group.create_dataset(
            "right_ee_joint_state", data=np.zeros((2, 1), dtype=np.float32)
        )
        handle.create_group("action").create_dataset(
            "left_ee_pose", data=np.zeros((2, 7), dtype=np.float32)
        )
    return path


def _reference_stats(states, actions):
    def stack_padded(sequences):
        max_len = max(len(sequence) for sequence in sequences)
        padded = [
            np.concatenate(
                [sequence, np.repeat(sequence[-1:], max_len - len(sequence), axis=0)],
                axis=0,
            )
            for sequence in sequences
        ]
        return torch.from_numpy(np.stack(padded, axis=0)), max_len

    qpos, _ = stack_padded(states)
    action, max_action_len = stack_padded(actions)
    return {
        "action_mean": action.mean(dim=[0, 1]).numpy(),
        "action_std": torch.clip(action.std(dim=[0, 1]), 1e-2, np.inf).numpy(),
        "qpos_mean": qpos.mean(dim=[0, 1]).numpy(),
        "qpos_std": torch.clip(qpos.std(dim=[0, 1]), 1e-2, np.inf).numpy(),
        "example_qpos": qpos[-1],
    }, max_action_len


class ACTStatsTest(unittest.TestCase):
    def test_stats_match_upstream_last_frame_padding(self):
        rng = np.random.RandomState(0)
        lengths = (3, 5, 4)
        states = [
            rng.normal(size=(length, 16)).astype(np.float32) for length in lengths
        ]
        actions = [
            rng.normal(size=(length, 16)).astype(np.float32) for length in lengths
        ]
        for action in actions:
            action[:, 0] = 0.75  # constant column exercises the std clip floor
        frames = [_solid((10, 20, 30)) for _ in range(max(lengths))]

        with tempfile.TemporaryDirectory() as tmp:
            names = ("episode_000011.hdf5", "episode_000002.hdf5", "episode_000007.hdf5")
            paths = [
                _write_episode(tmp, name, state, action, frames[: len(state)])
                for name, state, action in zip(names, states, actions)
            ]
            ordered = episode_files(Path(tmp))
            self.assertEqual([path.name for path in ordered], sorted(names))

            stats, max_action_len = compute_normalization_stats(ordered)
            expected, expected_max = _reference_stats(
                [states[names.index(path.name)] for path in ordered],
                [actions[names.index(path.name)] for path in ordered],
            )

        self.assertEqual(max_action_len, expected_max)
        self.assertEqual(
            set(stats),
            {"action_mean", "action_std", "qpos_mean", "qpos_std", "example_qpos"},
        )
        for key in ("action_mean", "action_std", "qpos_mean", "qpos_std"):
            np.testing.assert_allclose(stats[key], expected[key], rtol=1e-6, atol=1e-6)
        self.assertIsInstance(stats["example_qpos"], torch.Tensor)
        self.assertEqual(
            stats["example_qpos"].shape, (expected_max, 16)
        )
        torch.testing.assert_close(
            stats["example_qpos"], expected["example_qpos"]
        )
        np.testing.assert_allclose(stats["action_std"][0], 1e-2)
        self.assertEqual(stats["action_mean"].shape, (16,))

    def test_float64_episodes_yield_float32_stats_and_tensors(self):
        rng = np.random.RandomState(7)
        lengths = (3, 4)
        states = [
            rng.normal(size=(length, 16)).astype(np.float64) for length in lengths
        ]
        actions = [
            rng.normal(size=(length, 16)).astype(np.float64) for length in lengths
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = [
                _write_episode(
                    tmp,
                    f"episode_{index + 1:06d}.hdf5",
                    state,
                    action,
                    [_solid((10, 20, 30)) for _ in range(len(state))],
                )
                for index, (state, action) in enumerate(zip(states, actions))
            ]
            stats, max_action_len = compute_normalization_stats(paths)
            expected, _ = _reference_stats(
                [state.astype(np.float32) for state in states],
                [action.astype(np.float32) for action in actions],
            )

            for key in ("action_mean", "action_std", "qpos_mean", "qpos_std"):
                self.assertEqual(stats[key].dtype, np.float32)
                np.testing.assert_allclose(stats[key], expected[key], rtol=1e-6)
            self.assertEqual(stats["example_qpos"].dtype, torch.float32)

            dataset = ACTDataset(paths, ["cam_head"], stats, max_action_len)
            with mock.patch("starter.data.act.np.random.choice", return_value=1):
                _image, qpos, action, _is_pad = dataset[0]
            self.assertEqual(qpos.dtype, torch.float32)
            self.assertEqual(action.dtype, torch.float32)

    def test_invalid_dimensions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_bad_episode(tmp, "episode_000001.hdf5")
            with self.assertRaises(ValueError):
                compute_normalization_stats([path])

    def test_empty_episode_list_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_normalization_stats([])


class ACTDatasetTest(unittest.TestCase):
    def _build(self, tmp):
        states = (np.arange(4 * 16, dtype=np.float32) * 0.01).reshape(4, 16)
        actions = (np.arange(4 * 16, dtype=np.float32) + 100.0).reshape(4, 16) * 0.1
        frames = [
            _solid((255, 0, 0)),
            _solid((0, 255, 0)),
            _solid((0, 0, 255)),
            _solid((255, 255, 255)),
        ]
        path = _write_episode(tmp, "episode_000003.hdf5", states, actions, frames)
        stats, max_action_len = compute_normalization_stats([path])
        return path, states, actions, stats, max_action_len

    def test_sample_uses_start_offset_and_zero_pads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, states, actions, stats, max_action_len = self._build(tmp)
            dataset = ACTDataset([path], ["cam_head"], stats, max_action_len)

            with mock.patch("starter.data.act.np.random.choice", return_value=2):
                _image, qpos, action, is_pad = dataset[0]

            expected_qpos = (
                torch.from_numpy(states[2]).float()
                - torch.as_tensor(stats["qpos_mean"])
            ) / torch.as_tensor(stats["qpos_std"])
            expected_action = np.zeros((max_action_len, 16), dtype=np.float32)
            expected_action[:3] = actions[1:4]
            expected_action = (
                torch.from_numpy(expected_action).float()
                - torch.as_tensor(stats["action_mean"])
            ) / torch.as_tensor(stats["action_std"])

            torch.testing.assert_close(qpos, expected_qpos)
            torch.testing.assert_close(action, expected_action)
            self.assertEqual(is_pad.tolist(), [False, False, False, True])

            with mock.patch("starter.data.act.np.random.choice", return_value=0):
                _image, _qpos, action, is_pad = dataset[0]
            expected_action = (
                torch.from_numpy(actions).float()
                - torch.as_tensor(stats["action_mean"])
            ) / torch.as_tensor(stats["action_std"])
            torch.testing.assert_close(action, expected_action)
            self.assertEqual(is_pad.tolist(), [False, False, False, False])

    def test_image_is_resized_rgb_chw(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _states, _actions, stats, max_action_len = self._build(tmp)
            dataset = ACTDataset([path], ["cam_head"], stats, max_action_len)

            with mock.patch("starter.data.act.np.random.choice", return_value=2):
                image, _qpos, _action, _is_pad = dataset[0]

            self.assertEqual(tuple(image.shape), (1, 3, 480, 640))
            self.assertEqual(image.dtype, torch.float32)
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)
            corner = image[0, :, 0, 0]
            # Frame 2 is pure blue, so RGB order puts the blue channel second.
            self.assertGreater(float(corner[2]), 0.9)
            self.assertLess(float(corner[0]), 0.1)
            self.assertLess(float(corner[1]), 0.1)

    def test_constructor_consumes_rng_and_exposes_max_action_len(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _states, _actions, stats, max_action_len = self._build(tmp)
            with mock.patch(
                "starter.data.act.np.random.choice", return_value=0
            ) as choice:
                dataset = ACTDataset([path], ["cam_head"], stats, max_action_len)
            choice.assert_called_once()
            self.assertEqual(dataset.max_action_len, max_action_len)

    def test_missing_camera_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _states, _actions, stats, max_action_len = self._build(tmp)
            with self.assertRaises(ValueError):
                ACTDataset([path], ["cam_left_wrist"], stats, max_action_len)
            with self.assertRaises(ValueError):
                ACTDataset([path], ["cam_head", "cam_head"], stats, max_action_len)


class LoadActDataTest(unittest.TestCase):
    def _dataset_dir(self, tmp):
        rng = np.random.RandomState(4)
        names = (
            "episode_000031.hdf5",
            "episode_000001.hdf5",
            "episode_000020.hdf5",
            "episode_000009.hdf5",
            "episode_000004.hdf5",
        )
        for index, name in enumerate(names):
            length = 3 + index
            _write_episode(
                tmp,
                name,
                rng.normal(size=(length, 16)).astype(np.float32),
                rng.normal(size=(length, 16)).astype(np.float32),
                [_solid((index * 20, 30, 40)) for _ in range(length)],
            )
        return Path(tmp), names

    def test_all_episodes_in_sorted_order_and_loader_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _names = self._dataset_dir(tmp)
            train, stats = load_act_data(
                root, batch_size=2, camera_names=["cam_head"], num_workers=0
            )

            files = episode_files(root)
            self.assertEqual(train.dataset.episode_paths, files)
            self.assertEqual(len(train.dataset), 5)
            self.assertIsInstance(train.sampler, torch.utils.data.RandomSampler)
            self.assertTrue(train.pin_memory)
            self.assertFalse(train.drop_last)
            self.assertEqual(train.batch_size, 2)
            self.assertIsNone(train.prefetch_factor)
            self.assertEqual(
                set(stats),
                {"action_mean", "action_std", "qpos_mean", "qpos_std", "example_qpos"},
            )

            _train, _stats = load_act_data(
                root, batch_size=2, camera_names=["cam_head"], num_workers=1
            )
            self.assertEqual(_train.prefetch_factor, 1)

    def test_missing_data_is_rejected_and_one_episode_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            with self.assertRaises(FileNotFoundError):
                load_act_data(missing, 2, ["cam_head"])

            path = _write_episode(
                tmp,
                "episode_000001.hdf5",
                np.zeros((3, 16), dtype=np.float32),
                np.zeros((3, 16), dtype=np.float32),
                [_solid((1, 2, 3)) for _ in range(3)],
            )
            train, _stats = load_act_data(Path(tmp), 2, ["cam_head"])
            self.assertEqual(len(train.dataset), 1)
            self.assertEqual(train.dataset.episode_paths, [path])


if __name__ == "__main__":
    unittest.main()
