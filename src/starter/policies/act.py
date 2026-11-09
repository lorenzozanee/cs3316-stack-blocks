"""ACT pipeline. Model, training, and checkpoint format are yours."""

from __future__ import annotations


def train_offline(config: dict) -> None:
    """Behavior-clone a deterministic policy on the standard stack_blocks demos.

    Use the head camera and the 16-D end-effector state. The result has to be
    freezable: the online stage must not change these weights.
    """

    del config
    raise NotImplementedError("实现 ACT 的离线行为克隆，并保存可冻结的基座。")


def train_online(config: dict) -> None:
    """Learn a residual on the frozen policy in ``stack_blocks``.

    ``scripts/train_act_residual.sh`` has already started Isaac in the simulator
    image. Connect with ``starter.envs.remote.RemoteEnv``. The action is the
    frozen action plus a residual. Keep the environment score, and do not feed
    privileged block poses to the policy.
    """

    del config
    raise NotImplementedError("实现冻结基座上的残差在线训练。")
