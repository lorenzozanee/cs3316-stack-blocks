"""Evaluate starter ACT checkpoints through the XPolicyLab websocket server."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)
from starter.interfaces import (
    ACTION_DIM,
    BENCH_NAME,
    ENV_CFG,
    EVAL_TASK,
    GRIPPER_KEYS,
    HEAD_CAMERA,
)
from starter.policies.act_inference import block_pose_features, compose_residual_action, load_act_runner


def _block_poses(obs: Mapping, env_index: int) -> dict:
    """Use the same ordered pose validation as the online training runner."""
    try:
        blocks = obs["privileged"]["blocks"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Residual env {env_index} requires privileged block poses.") from error
    block_pose_features(blocks)
    return blocks


class Model(ModelTemplate):
    """Frozen ACT sees only head RGB and 16-D proprio.

    Online residual checkpoints additionally read privileged block poses from
    the flat observation's ``blocks`` entry; those poses never reach the
    frozen base network.
    """

    def __init__(self, model_cfg):
        for key, expected in (("bench_name", BENCH_NAME), ("task_name", EVAL_TASK),
                              ("env_cfg_type", ENV_CFG), ("action_type", "ee")):
            if model_cfg[key] != expected:
                raise ValueError(f"ACT evaluation requires {key}={expected!r}.")
        policy_dir = Path(__file__).resolve().parent
        checkpoint = resolve_checkpoint_root(
            model_cfg, policy_dir / "checkpoints", policy_dir=policy_dir,
        )
        if not checkpoint.is_file():
            raise ValueError("Pass the path of an ACT .ckpt file, not a run directory.")
        self.device = torch.device(model_cfg["device"])
        temporal_agg = bool(model_cfg["temporal_agg"])
        self.runner, self.residual, self.residual_scale = load_act_runner(
            checkpoint, self.device, temporal_agg=temporal_agg,
        )
        if self.runner.policy.model.camera_names != [HEAD_CAMERA]:
            raise ValueError("ACT evaluation requires the cam_head-only checkpoint.")
        self.robot_dims = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        self.temporal_agg = temporal_agg
        self.reset()

    def reset(self):
        self.runner.reset()
        self.observations = {}

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        observations = {}
        for obs in obs_list:
            # The shared server has already decoded this image to RGB.
            color = np.asarray(obs["vision"][HEAD_CAMERA]["color"])
            if color.ndim != 3 or color.shape[2] != 3:
                raise ValueError("cam_head must be a decoded HWC RGB image.")
            qpos = pack_robot_state(obs, "ee", self.robot_dims, source_type="obs")
            qpos = np.asarray(qpos, dtype=np.float32)
            if qpos.shape != (ACTION_DIM,):
                raise ValueError(f"ACT proprio must have shape ({ACTION_DIM},).")
            if not np.isfinite(qpos).all():
                raise ValueError("ACT proprio contains a non-finite value.")
            observation = {
                "image": color,
                "proprio": qpos.tolist(),
                "camera": HEAD_CAMERA,
            }
            if self.residual is not None:
                observation["blocks"] = _block_poses(obs, int(obs.get("env_idx", 0)))
            observations[int(obs.get("env_idx", 0))] = observation
        self.observations = observations

    def _unpack_chunk(self, chunk):
        unpacked = unpack_robot_state(chunk, "ee", self.robot_dims, source_type="obs")
        for action in unpacked:
            for key in GRIPPER_KEYS:
                action[key] = np.clip(action[key], 0.0, 1.0)
        return unpacked

    def get_action(self):
        return self.get_action_batch([0])[0]

    @torch.no_grad()
    def get_action_batch(self, env_idx_list=None):
        indices = list(self.observations) if env_idx_list is None else [int(index) for index in env_idx_list]
        if not indices:
            raise RuntimeError("Update observations before requesting ACT actions.")
        obs_list = [self.observations[index] for index in indices]
        chunks, features = self.runner.predict_base(
            obs_list, indices, with_features=self.residual is not None,
        )
        if self.residual is None:
            return [self._unpack_chunk(chunk) for chunk in chunks]

        # Online: add the deterministic residual mean to the selected base action.
        base_actions = np.stack([chunk[-1] for chunk in chunks]).astype(np.float32)
        raw_actions, _, _ = self.residual.act(features, deterministic=True)
        composed = compose_residual_action(
            base_actions, raw_actions, self.runner.stats["action_std"], self.residual_scale,
            position_residual_multiplier=self.runner.position_residual_multiplier,
        )
        return [self._unpack_chunk(composed[index : index + 1]) for index in range(len(indices))]
