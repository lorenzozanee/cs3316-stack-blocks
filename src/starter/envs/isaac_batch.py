"""Step several RoboDojo scenes in one Isaac process.

Eval already does this in ``take_action_batch``. The online server cannot use
``EvalEnv`` itself: that class opens a policy websocket. This attaches the same
end-effector control and camera read to a plain task object.
"""

from __future__ import annotations

from copy import deepcopy


def install_parallel_control(task, observation_cfg: dict, task_name: str) -> None:
    """Add ``get_obs_batch`` and ``take_action_batch`` to ``task``."""

    import numpy as np

    from env.observation_manager.obs_manager import ObsManager

    num_envs = int(task.num_envs)
    seeds = list(getattr(task, "env_seed_list", None) or [0] * num_envs)
    obs = ObsManager(
        obs_config=dict(observation_cfg),
        num_envs=num_envs,
        dt=float(task.dt),
        task_name=task_name,
        description_cfg={},
        seeds_per_env=seeds,
    )
    obs.initialize(task)
    task.online_obs = obs
    _install_capture_lifecycle(task.capture_manager)

    def get_obs_batch(env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(num_envs))
        task.online_obs.render_for_capture()
        data = task.online_obs.get_obs(env_idx_list=list(env_idx_list))
        return [data[index] for index in env_idx_list]

    def take_action_batch(actions_list, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(num_envs))
        env_idx_list = list(env_idx_list)
        sequences = []
        for action, env_idx in zip(actions_list, env_idx_list):
            sequences.append(_control_sequence(task, action, int(env_idx), np))
        if not sequences:
            return
        task.robot_manager.control_manager.push(env_idx_list, sequences)
        while _controls_remain(task, env_idx_list):
            meta = task.robot_manager.control_manager.pop(env_idx_list)
            task.robot_manager.control_robot(meta_control_list=meta)
            task.sim_step(render=False)
        if hasattr(task.reward_manager, "step"):
            task.reward_manager.step(env_idx_list=env_idx_list)

    task.get_obs_batch = get_obs_batch
    task.take_action_batch = take_action_batch


def _install_capture_lifecycle(capture) -> None:
    """Reuse capture resources until CameraManager replaces its camera handles.

    The pinned TiledCaptureManager.reset appends new render products and
    annotators on every episode. Its CameraView cleanup also references unused
    attributes. Manage their lifetime here without modifying the simulator image.
    """
    previous_cameras = ()

    def release_resources():
        for view in capture.tiled_cameras:
            for annotator in view._annotators.values():
                annotator.detach([view._render_product_path])
            view._annotators.clear()
        for product in capture.tiled_render_products:
            product.destroy()
        capture.tiled_cameras.clear()
        capture.tiled_render_products.clear()
        capture.annotator.clear()
        capture.annotator_type.clear()
        capture.annotator_device.clear()
        capture.camera_prim_paths.clear()
        capture._output_buffers.clear()

    def reset():
        nonlocal previous_cameras
        cameras = tuple(camera for row in capture.camera_manager.cameras for camera in row)
        unchanged = len(cameras) == len(previous_cameras) and all(
            current is previous for current, previous in zip(cameras, previous_cameras)
        )
        if unchanged and capture.tiled_cameras:
            print(f"[env] capture reused: render_products={len(capture.tiled_render_products)}", flush=True)
            return
        release_resources()
        capture.cameras = capture.camera_manager.cameras
        capture.camera_names = capture.camera_manager.camera_names
        capture.num_cams = len(capture.cameras[0])
        capture.init_cameras()
        previous_cameras = cameras
        print(f"[env] capture initialized: render_products={len(capture.tiled_render_products)}", flush=True)

    def destroy():
        nonlocal previous_cameras
        release_resources()
        previous_cameras = ()
        # Do not clear the aliased CameraManager lists: it owns those handles.
        capture.cameras = []
        capture.camera_names = []
        capture.sim = None

    capture.reset = reset
    capture.destroy = destroy


def bind_eval_layouts(task, task_name: str, config_name: str, seed: int = 0):
    """Point each scene at an official eval layout before ``reset``.

    Training seeds are layout indices modulo the number of saved layouts.
    """

    from env.seed_manager.seed_manager import SeedManager

    manager = SeedManager(
        {
            "num_envs": int(task.num_envs),
            "task_name": task_name,
            "config_name": config_name,
            "seed": int(seed),
        }
    )
    manager.init_eval()
    layouts = list(manager.seed_list)
    if not layouts:
        raise RuntimeError(f"No eval layouts for {task_name} / {config_name}.")
    original = task.reset

    def reset(seed=None, options=None):
        seeds = list(seed) if isinstance(seed, (list, tuple)) else [seed]
        if len(seeds) != int(task.num_envs):
            raise ValueError(f"Expected {task.num_envs} seeds, got {len(seeds)}.")
        for index, item in enumerate(seeds):
            layout = layouts[int(item) % len(layouts)]
            task.scene_manager.layout_manager.set_saved_layout(
                index, manager.get_seed_scene_info(layout)
            )
        result = original(seed=list(seeds), options=options)
        task.online_obs.reset()
        # CUDA reset parks pooled rigid bodies and the table offscreen. Restore
        # the selected layout before the adapter initializes reward state.
        task.scene_manager.apply_saved_poses(env_idx_list=list(range(task.num_envs)))
        # Reuse EvalEnv.setup_scene's validity and 300-step stability check;
        # evaluating a different settled scene is not an ACT/PPO comparison.
        layout_manager = task.scene_manager.layout_manager
        task.success = [bool(valid) for valid in layout_manager.layout_valid]
        task.end_flag = [not valid for valid in task.success]
        invalid = [i for i, valid in enumerate(task.success) if not valid]
        if invalid:
            raise RuntimeError(f"Invalid training layouts in scenes {invalid}, "
                               f"layout ids {[seeds[i] for i in invalid]}.")
        any_usable, unstable = layout_manager.check_layout_stability(task)
        failed = sorted(set(unstable) | {i for i, valid in enumerate(task.success) if not valid})
        # The upstream boolean means ANY scene survived. Eval drops failed
        # scenes; this fixed-width training batch must fail explicitly rather
        # than learn from an invalid scene. RemoteEnv propagates this error.
        if not any_usable or failed:
            raise RuntimeError(f"Unstable training layouts in scenes {failed}, "
                               f"layout ids {[seeds[i] for i in failed]}.")
        # Match EvalEnv.setup_scene: let materials and camera exposure settle
        # before ACT sees its first frame or rewards capture their initial state.
        for _ in range(10):
            task.render()
        for step in range(200):
            task.sim_step()
            if step % 5 == 0:
                task.online_obs.render_for_capture()
                task.online_obs.get_obs()
        return result

    task.reset = reset
    return len(layouts)


def _controls_remain(task, env_idx_list) -> bool:
    empty = task.robot_manager.control_manager.get_empty(env_idx_list=env_idx_list)
    return len(empty) == 0


def _control_sequence(task, action: dict, env_idx: int, np):
    interval = int(task.online_obs.collect_interval)
    control = {}
    for robot in task.robot_manager.robot_list:
        if robot.type != "target":
            continue
        name = robot.arm_name.split("_")[0]
        pose_key = f"{name}_ee_pose"
        target_pose = action[pose_key]
        ik_result = task.robot_manager.solve_ik(target_pose=target_pose, env_idx=env_idx, robot=robot)
        obs_name = task.robot_manager.process_name(robot.arm_name)
        if ik_result["status"] == "Success":
            control[obs_name] = {"position": ik_result["joint_value"]}
        gripper_key = task.robot_manager.process_name(robot.gripper_name)
        if robot.ee_type == "gripper":
            val = float(np.clip(action[gripper_key][0], 0, 1))
            if robot.gripper_move["sign"] == 1:
                val = val * (robot.gripper_scale[1] - robot.gripper_scale[0]) + robot.gripper_scale[0]
            else:
                val = (1 - val) * (robot.gripper_scale[1] - robot.gripper_scale[0]) + robot.gripper_scale[0]
            control[gripper_key] = {
                "position": [
                    val,
                    val * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                ]
            }
    if interval <= 0:
        return [deepcopy(control)]
    frames = [deepcopy(control) for _ in range(interval)]
    _interpolate(task, control, frames, env_idx, interval, np)
    return frames


def _interpolate(task, control, frames, env_idx: int, interval: int, np) -> None:
    interp_count = int(np.floor(interval * 0.8))
    for robot in task.robot_manager.robot_list:
        if robot.type != "target":
            continue
        arm_key = task.robot_manager.process_name(robot.arm_name)
        if arm_key in control:
            current = task.robot_manager.get_joint(robot, env_idx_list=[env_idx])[env_idx]
            target = np.array(control[arm_key]["position"])
            if current is not None:
                current_arr = np.array(current)
                for frame in range(interp_count):
                    alpha = (frame + 1) / (interp_count + 1)
                    frames[frame][arm_key]["position"] = ((1 - alpha) * current_arr + alpha * target).tolist()
                for frame in range(interp_count, interval):
                    frames[frame][arm_key]["position"] = target.tolist()
        gripper_key = task.robot_manager.process_name(robot.gripper_name)
        if gripper_key in control and robot.ee_type == "gripper":
            target_pos = control[gripper_key]["position"][0]
            current = task.robot_manager.get_end_effector_real_val(robot, env_idx_list=[env_idx])[env_idx][0]
            scale = robot.gripper_scale
            if current is not None:
                for frame in range(interp_count):
                    alpha = (frame + 1) / (interp_count + 1)
                    interp_pos = float(np.clip((1 - alpha) * current + alpha * target_pos, scale[0], scale[1]))
                    frames[frame][gripper_key]["position"] = [
                        interp_pos,
                        interp_pos * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                    ]
                held = float(np.clip(target_pos, scale[0], scale[1]))
                held_vals = [held, held * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2]]
                for frame in range(interp_count, interval):
                    frames[frame][gripper_key]["position"] = list(held_vals)
