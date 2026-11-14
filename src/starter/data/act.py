"""ACT data pipeline over the raw starter HDF5 episodes.

The layout mirrors upstream ``policy/ACT/utils.py``: one sample per episode,
a random observation timestamp, the expert actions from ``max(0, t - 1)``
zero-padded to the global maximum episode length, and dataset-wide qpos/action
normalization statistics.

Episodes store 16-D end-effector state and action groups packed in
``interfaces.ARM_KEYS`` order, and encoded head-camera frames. Only
``cam_head`` is exposed to the policy.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from starter.data.demos import episode_files
from starter.interfaces import ACTION_DIM, ARM_KEYS, ARM_WIDTHS, HEAD_CAMERA

IMAGE_SIZE = (640, 480)
STD_MIN = 1e-2


class ACTDataset(torch.utils.data.Dataset):
    """One random-start sample per episode, matching upstream ``EpisodicDataset``."""

    def __init__(
        self,
        episode_paths: list[Path],
        camera_names: list[str],
        norm_stats: dict[str, np.ndarray],
        max_action_len: int,
    ) -> None:
        if not episode_paths:
            raise ValueError("ACTSDataset requires at least one episode.")
        if max_action_len <= 0:
            raise ValueError(f"max_action_len must be positive, got {max_action_len}.")
        self.episode_paths = [Path(path) for path in episode_paths]
        self.camera_names = _validate_cameras(camera_names)
        self.norm_stats = norm_stats
        self.max_action_len = int(max_action_len)
        self.is_sim = None
        # Upstream initializes is_sim with one sample, consuming the global
        # NumPy RNG before training starts. Keep that RNG behavior.
        self.__getitem__(0)

    def __len__(self) -> int:
        return len(self.episode_paths)

    def __getitem__(self, index: int):
        path = self.episode_paths[index]
        if not path.is_file():
            raise FileNotFoundError(f"Episode file not found: {path}")

        with h5py.File(path, "r") as handle:
            state_columns, episode_len = _column_datasets(handle, "state", path)
            action_columns, action_len = _column_datasets(handle, "action", path)
            if action_len != episode_len:
                raise ValueError(
                    f"{path}: state and action lengths differ "
                    f"({episode_len} vs {action_len})."
                )

            start_ts = int(np.random.choice(episode_len))
            qpos = _read_state_row(state_columns, start_ts)
            action_start = max(0, start_ts - 1)
            action = _read_action_range(action_columns, action_start)
            images = [
                _read_image(handle, camera, start_ts, path)
                for camera in self.camera_names
            ]

        action_len = episode_len - action_start
        if action_len > self.max_action_len:
            raise ValueError(
                f"{path}: action length {action_len} exceeds global maximum "
                f"{self.max_action_len}."
            )

        padded_action = np.zeros((self.max_action_len, ACTION_DIM), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)
        is_pad[:action_len] = False

        image_data = torch.from_numpy(np.stack(images, axis=0))
        image_data = torch.einsum("k h w c -> k c h w", image_data)
        image_data = image_data / 255.0

        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad_data = torch.from_numpy(is_pad).bool()

        action_data = (action_data - torch.as_tensor(self.norm_stats["action_mean"])) / (
            torch.as_tensor(self.norm_stats["action_std"])
        )
        qpos_data = (qpos_data - torch.as_tensor(self.norm_stats["qpos_mean"])) / (
            torch.as_tensor(self.norm_stats["qpos_std"])
        )

        return image_data, qpos_data, action_data, is_pad_data


def compute_normalization_stats(
    episode_paths: list[Path],
) -> tuple[dict[str, np.ndarray], int]:
    """Dataset-wide stats with upstream's last-frame padding and unbiased std.

    Returns ``(stats, max_action_len)``. Episode-last frames pad every shorter
    trajectory to the global maximum before the mean/std are taken, and stds
    are clipped to ``1e-2`` exactly as in upstream ``get_norm_stats``.
    """

    paths = [Path(path) for path in episode_paths]
    if not paths:
        raise ValueError("Cannot compute normalization stats without episodes.")

    qpos_episodes: list[np.ndarray] = []
    action_episodes: list[np.ndarray] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Episode file not found: {path}")
        with h5py.File(path, "r") as handle:
            state_columns, state_len = _column_datasets(handle, "state", path)
            action_columns, action_len = _column_datasets(handle, "action", path)
            if state_len != action_len:
                raise ValueError(
                    f"{path}: state and action lengths differ "
                    f"({state_len} vs {action_len})."
                )
            qpos_episodes.append(_read_full(state_columns))
            action_episodes.append(_read_full(action_columns))

    max_qpos_len = max(len(qpos) for qpos in qpos_episodes)
    max_action_len = max(len(action) for action in action_episodes)

    padded_qpos = [_pad_last_frame(qpos, max_qpos_len) for qpos in qpos_episodes]
    padded_action = [
        _pad_last_frame(action, max_action_len) for action in action_episodes
    ]

    all_qpos = torch.from_numpy(np.stack(padded_qpos, axis=0))
    all_action = torch.from_numpy(np.stack(padded_action, axis=0))

    action_mean = all_action.mean(dim=[0, 1], keepdim=True)
    action_std = torch.clip(all_action.std(dim=[0, 1], keepdim=True), STD_MIN, np.inf)
    qpos_mean = all_qpos.mean(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(all_qpos.std(dim=[0, 1], keepdim=True), STD_MIN, np.inf)

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        # Upstream keeps the last episode's qpos after last-frame padding.
        "example_qpos": torch.from_numpy(padded_qpos[-1]),
    }
    return stats, max_action_len


def load_act_data(
    data_path: Path,
    batch_size: int,
    camera_names: list[str],
    num_workers: int = 1,
):
    """Build the training loader and stats over every sorted episode.

    All episodes are used for training; there is no validation split. The
    loader shuffles each epoch, and ``ACTSDataset`` keeps upstream's
    constructor-time RNG consumption.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"num_workers cannot be negative, got {num_workers}.")
    camera_names = _validate_cameras(camera_names)

    files = episode_files(Path(data_path))
    if not files:
        raise FileNotFoundError(f"No episode files found under {data_path}.")

    stats, max_action_len = compute_normalization_stats(files)
    train_dataset = ACTDataset(files, camera_names, stats, max_action_len)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "pin_memory": True,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    return train_loader, stats


def _validate_cameras(camera_names: list[str]) -> list[str]:
    names = list(camera_names)
    if names != [HEAD_CAMERA]:
        raise ValueError(
            f"camera_names must be exactly ['{HEAD_CAMERA}'], got {names}."
        )
    return names


def _column_datasets(
    handle: h5py.File, group_name: str, path: Path
) -> tuple[list[h5py.Dataset], int]:
    """Resolve the packed ARM_KEYS columns of one state/action group."""

    if group_name not in handle:
        raise KeyError(f"{path}: missing '{group_name}' group.")
    group = handle[group_name]

    columns: list[h5py.Dataset] = []
    length: int | None = None
    for key, width in zip(ARM_KEYS, ARM_WIDTHS):
        name = key if key in group else key + "s"
        if name not in group:
            raise KeyError(f"{path}: missing '{group_name}/{name}'.")
        dataset = group[name]
        shape = tuple(dataset.shape)
        if len(shape) != 2 or shape[1] != width:
            raise ValueError(
                f"{path}: '{group_name}/{name}' must have shape (T, {width}), "
                f"got {shape}."
            )
        if length is None:
            length = shape[0]
        elif shape[0] != length:
            raise ValueError(
                f"{path}: '{group_name}/{name}' length {shape[0]} does not "
                f"match {length}."
            )
        columns.append(dataset)

    if length == 0:
        raise ValueError(f"{path}: '{group_name}' is empty.")
    return columns, int(length)


def _read_state_row(columns: list[h5py.Dataset], index: int) -> np.ndarray:
    return np.concatenate([np.asarray(column[index]).reshape(-1) for column in columns])


def _read_action_range(columns: list[h5py.Dataset], start: int) -> np.ndarray:
    return np.concatenate([np.asarray(column[start:]) for column in columns], axis=1)


def _read_full(columns: list[h5py.Dataset]) -> np.ndarray:
    # Upstream conversion writes both qpos and actions as float32 before stats.
    return np.concatenate([np.asarray(column[:]) for column in columns], axis=1).astype(np.float32, copy=False)


def _pad_last_frame(data: np.ndarray, length: int) -> np.ndarray:
    if len(data) == length:
        return data
    pad = np.repeat(data[-1:], length - len(data), axis=0)
    return np.concatenate([data, pad], axis=0)


def _read_image(
    handle: h5py.File, camera: str, index: int, path: Path
) -> np.ndarray:
    from XPolicyLab.utils.process_data import decode_image_bit

    vision = handle.get("vision")
    if vision is None or camera not in vision:
        raise KeyError(f"{path}: missing 'vision/{camera}'.")
    frames = vision[camera]
    if "colors" not in frames:
        raise KeyError(f"{path}: missing 'vision/{camera}/colors'.")

    image = np.asarray(decode_image_bit(frames["colors"][index]))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"{path}: decoded '{camera}' frame {index} must be HxWx3, "
            f"got shape {image.shape}."
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"{path}: decoded '{camera}' frame {index} must be uint8, "
            f"got {image.dtype}."
        )
    return cv2.resize(image, IMAGE_SIZE)
