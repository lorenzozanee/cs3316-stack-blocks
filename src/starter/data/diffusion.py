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
        self.image_size = image_size
        self.decode_image = decode_image
        self.camera_key = f"vision/{HEAD_CAMERA}/colors"
        self.episodes = []
        for path in paths:
            with h5py.File(path, "r") as handle:
                state = np.asarray(pack_ee_group(handle["state"]), dtype=np.float32)
                action = np.asarray(pack_ee_group(handle["action"]), dtype=np.float32)
                frames = len(handle[self.camera_key])
            if state.shape != (frames, ACTION_DIM) or action.shape != state.shape:
                raise ValueError(f"{path}: mismatched image, state or action shapes.")
            if not (np.isfinite(state).all() and np.isfinite(action).all()):
                raise ValueError(f"{path}: non-finite state or action values.")
            if frames >= horizon:
                self.episodes.append((path, state, action))

        self.ends = list(accumulate(len(action) - horizon + 1 for _, _, action in self.episodes))
        if not self.ends:
            raise ValueError(f"No full action windows of length {horizon} in this split.")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        episode = bisect_right(self.ends, index)
        t = index - (self.ends[episode - 1] if episode else 0)
        path, state, action = self.episodes[episode]
        with h5py.File(path, "r") as handle:
            rgb = self.decode_image(handle[self.camera_key][t])
        image = preprocess_image(rgb, self.image_size)
        return image, state[t], action[t : t + self.horizon]

    def bounds(self):
        stats = {}
        for key, column in (("state", 1), ("action", 2)):
            values = [episode[column] for episode in self.episodes]
            stats[f"{key}_min"] = np.minimum.reduce([v.min(axis=0) for v in values])
            stats[f"{key}_max"] = np.maximum.reduce([v.max(axis=0) for v in values])
        return stats
