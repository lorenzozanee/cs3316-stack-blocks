"""Episode-local action windows for visual behavior cloning."""

from bisect import bisect_right
from itertools import accumulate

import cv2
import h5py
import numpy as np

from starter.data.demos import pack_ee_group
from starter.interfaces import ACTION_DIM, HEAD_CAMERA


def preprocess_image(rgb, image_size):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 RGB image, got {rgb.shape} {rgb.dtype}.")
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized.transpose(2, 0, 1))


class ActionWindows:
    def __init__(self, paths, horizon, image_size, decode_image):
        self.horizon = horizon
        self.episodes = []
        self.images = []
        camera_key = f"vision/{HEAD_CAMERA}/colors"
        for path in paths:
            with h5py.File(path, "r") as handle:
                state = np.asarray(pack_ee_group(handle["state"]), dtype=np.float32)
                action = np.asarray(pack_ee_group(handle["action"]), dtype=np.float32)
                camera = handle[camera_key]
                frames = len(camera)
                if frames == 0 or state.shape != (frames, ACTION_DIM) or action.shape != state.shape:
                    raise ValueError(f"{path}: empty or mismatched image, state or action shapes.")
                if not (np.isfinite(state).all() and np.isfinite(action).all()):
                    raise ValueError(f"{path}: non-finite state or action values.")
                images = np.empty((frames, 3, image_size, image_size), dtype=np.uint8)
                for t, encoded in enumerate(camera):
                    images[t] = preprocess_image(decode_image(encoded), image_size)
            self.images.append(images)
            self.episodes.append((path, state, action))

        self.ends = list(accumulate(len(action) for _, _, action in self.episodes))
        if not self.ends:
            raise ValueError("No episodes in this split.")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        episode = bisect_right(self.ends, index)
        t = index - (self.ends[episode - 1] if episode else 0)
        _, state, action = self.episodes[episode]
        valid = min(self.horizon, len(action) - t)
        steps = np.arange(t, t + self.horizon).clip(max=len(action) - 1)
        return self.images[episode][t], state[t], action[steps], valid

    def bounds(self):
        stats = {}
        for key, column in (("state", 1), ("action", 2)):
            values = [episode[column] for episode in self.episodes]
            stats[f"{key}_min"] = np.minimum.reduce([v.min(axis=0) for v in values])
            stats[f"{key}_max"] = np.maximum.reduce([v.max(axis=0) for v in values])
        return stats
