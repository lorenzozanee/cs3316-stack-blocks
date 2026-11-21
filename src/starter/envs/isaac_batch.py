"""Step several RoboDojo scenes in one Isaac process.

Eval already does this in ``take_action_batch``. The online server cannot use
``EvalEnv`` itself: that class opens a policy websocket. This attaches the same
end-effector control and camera read to a plain task object.
"""

from __future__ import annotations


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
    batch_ik = _build_batch_ik_solvers(task, num_envs)
    task._batch_ik_solvers = batch_ik

    def get_obs_batch(env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(num_envs))
        task.online_obs.render_for_capture()
        data = task.online_obs.get_obs(env_idx_list=list(env_idx_list))
        return [data[index] for index in env_idx_list]

    def take_action_batch(actions_list, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(num_envs))
        env_idx_list = [int(idx) for idx in env_idx_list]
        if not env_idx_list:
            return
        sequences = _control_sequences_batch(task, actions_list, env_idx_list, np, batch_ik, num_envs)
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


def install_room_fixture_cleanup(scene_manager) -> None:
    """Match the pinned layout manager's plural USD path for room fixtures."""
    from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid

    delete_fixture = scene_manager.delete_fixture_prims

    def delete_fixture_prims(env_id: int, types: str) -> None:
        if types == "Room":
            path = f"{scene_manager.env_roots[env_id]}/Rooms"
            if is_prim_path_valid(path):
                delete_prim(path)
        else:
            delete_fixture(env_id, types)

    scene_manager.delete_fixture_prims = delete_fixture_prims


def bind_eval_layouts(task, task_name: str, config_name: str, seed: int = 0, *, sample_reset_motion=None):
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
        warmup_steps = 200
        if sample_reset_motion is not None:
            interval = max(1, int(task.online_obs.collect_interval))
            warmup_steps = max(warmup_steps, interval)
        for step in range(warmup_steps):
            if sample_reset_motion is not None and step == warmup_steps - interval:
                sample_reset_motion()
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


def _build_batch_ik_solvers(task, num_envs: int) -> dict:
    if num_envs <= 1:
        return {}
    ik_map = getattr(getattr(task, "robot_manager", None), "ik_solver", None)
    if not isinstance(ik_map, dict) or not ik_map:
        return {}
    planners = {
        name: planner for name, planner in ik_map.items()
        if getattr(planner, "ik_solver", None) is not None
        and hasattr(planner, "robot_cfg")
        and hasattr(planner, "scene_model")
    }
    if not planners:
        return {}
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import GoalToolPose, JointState

    solvers = {}
    for name, planner in planners.items():
        cfg = InverseKinematicsCfg.create(
            robot=planner.robot_cfg,
            scene_model=planner.scene_model,
            device_cfg=planner.device_cfg,
            num_seeds=32,
            seed_solver_num_seeds=32,
            position_tolerance=0.001,
            orientation_tolerance=0.02,
            self_collision_check=True,
            use_cuda_graph=bool(getattr(planner, "use_cuda_graph", True)),
            max_batch_size=num_envs,
            multi_env=False,
            max_goalset=1,
        )
        solver = InverseKinematics(cfg)
        retract = solver.default_joint_state.clone()
        retract_pos = retract.position.view(1, -1).repeat(num_envs, 1)
        retract_js = JointState.from_position(retract_pos, joint_names=list(solver.joint_names))
        kin = solver.compute_kinematics(retract_js)
        goal = GoalToolPose.from_poses(
            kin.tool_poses.to_dict(),
            ordered_tool_frames=list(solver.tool_frames),
            num_goalset=1,
        )
        solver.solve_pose(goal_tool_poses=goal, current_state=retract_js, return_seeds=1)
        solvers[name] = (planner, solver)
    print(f"[env] batched cuRobo IK ready: robots={sorted(solvers)} batch={num_envs}", flush=True)
    return solvers


def _extract_active_ik_batch(planner, js_solution, batch_size: int, np):
    arr = np.asarray(js_solution.position.detach().cpu())
    while arr.ndim > 2:
        arr = arr[:, 0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] < batch_size:
        raise ValueError(f"Batched IK returned {arr.shape[0]} rows for batch_size {batch_size}.")
    dof = arr.shape[-1]
    n_active = len(planner.active_joints_name)
    n_cspace = len(planner.cspace_joint_names)
    js_names = list(getattr(js_solution, "joint_names", None) or [])
    if len(js_names) == dof:
        indices = [js_names.index(joint) for joint in planner.active_joints_name]
        return arr[:, indices]
    if dof == n_cspace:
        return arr[:, planner._active_to_cspace_idx]
    if dof == n_active:
        return arr
    raise ValueError(
        f"Unexpected batched IK DOF {dof}: expected {n_active} active or {n_cspace} cspace joints."
    )


def _solve_ik_batch(planner, solver, batch_size: int, robot, target_poses, current_joints, np) -> list[dict]:
    import torch
    from curobo.types import GoalToolPose, JointState, Pose
    from utils.transformer import calculate_target_pose

    actual = len(target_poses)
    if actual == 0:
        return []
    if actual > batch_size:
        raise ValueError(f"Got {actual} IK targets for max_batch_size {batch_size}.")
    world_base_pose = np.asarray(planner.robot_origin_pose, dtype=np.float32)
    frame_bias = np.asarray(planner.frame_bias, dtype=np.float32)
    positions = []
    quaternions = []
    cspaces = []
    for target_pose, curr_qpos in zip(target_poses, current_joints):
        transformed = calculate_target_pose(robot.entity_origin_pose, planner.robot_origin_pose, list(target_pose))
        pose_p, pose_q = planner._trans_from_world_to_base(world_base_pose, np.asarray(transformed, dtype=np.float32))
        positions.append(np.asarray(pose_p, dtype=np.float32) + frame_bias)
        quaternions.append(np.asarray(pose_q, dtype=np.float32))
        cspaces.append(planner._build_cspace_joint_values(curr_qpos))
    # Pad active subsets to the captured CUDA-graph batch size so cuRobo never resets graphs mid-wave.
    while len(positions) < batch_size:
        positions.append(positions[0])
        quaternions.append(quaternions[0])
        cspaces.append(cspaces[0])

    device = planner.device_cfg.device
    pos_tensor = torch.as_tensor(np.asarray(positions, dtype=np.float32), dtype=torch.float32, device=device)
    quat_tensor = torch.as_tensor(np.asarray(quaternions, dtype=np.float32), dtype=torch.float32, device=device)
    goal = GoalToolPose.from_poses(
        {planner.ee_link: Pose(position=pos_tensor, quaternion=quat_tensor)},
        ordered_tool_frames=planner.tool_frames,
        num_goalset=1,
    )
    q_tensor = torch.as_tensor(np.asarray(cspaces, dtype=np.float32), dtype=torch.float32, device=device)
    current_state = JointState.from_position(q_tensor, joint_names=planner.cspace_joint_names)
    ik_result = solver.solve_pose(goal_tool_poses=goal, current_state=current_state, return_seeds=1)
    success = planner._success_to_bool_array(ik_result.success, batch_size)[:actual]
    solutions = _extract_active_ik_batch(planner, ik_result.js_solution, batch_size, np) if bool(success.any()) else None
    return [
        {"status": "Success", "joint_value": solutions[index]} if success[index] else {"status": "Fail"}
        for index in range(actual)
    ]


def _control_sequences_batch(task, actions_list, env_idx_list: list[int], np, batch_ik: dict, batch_size: int):
    interval = int(task.online_obs.collect_interval)
    frame_count = interval if interval > 0 else 1
    interp_count = int(np.floor(interval * 0.8)) if interval > 0 else 0
    sequences = [[{} for _ in range(frame_count)] for _ in env_idx_list]

    for robot in task.robot_manager.robot_list:
        if robot.type != "target":
            continue
        name = robot.arm_name.split("_")[0]
        pose_key = f"{name}_ee_pose"
        arm_key = task.robot_manager.process_name(robot.arm_name)
        gripper_key = task.robot_manager.process_name(robot.gripper_name)
        joints_map = task.robot_manager.get_joint(robot, env_idx_list=env_idx_list)
        target_poses = [action[pose_key] for action in actions_list]

        robot_name = getattr(robot, "robot_name", None)
        if robot_name in batch_ik:
            planner, solver = batch_ik[robot_name]
            ik_results = _solve_ik_batch(
                planner,
                solver,
                batch_size,
                robot,
                target_poses,
                [joints_map[env_idx] for env_idx in env_idx_list],
                np,
            )
        else:
            ik_results = [
                task.robot_manager.solve_ik(target_pose=pose, env_idx=env_idx, robot=robot)
                for pose, env_idx in zip(target_poses, env_idx_list)
            ]

        gripper_map = (
            task.robot_manager.get_end_effector_real_val(robot, env_idx_list=env_idx_list)
            if robot.ee_type == "gripper"
            else None
        )

        for seq_idx, (action, env_idx, ik_result) in enumerate(zip(actions_list, env_idx_list, ik_results)):
            frames = sequences[seq_idx]
            if ik_result["status"] == "Success":
                target = np.asarray(ik_result["joint_value"])
                target_list = target.tolist()
                current = joints_map[env_idx]
                if interval > 0 and current is not None:
                    current_arr = np.asarray(current)
                    for frame in range(interp_count):
                        alpha = (frame + 1) / (interp_count + 1)
                        frames[frame][arm_key] = {"position": ((1 - alpha) * current_arr + alpha * target).tolist()}
                    for frame in range(interp_count, frame_count):
                        frames[frame][arm_key] = {"position": list(target_list)}
                else:
                    for frame in range(frame_count):
                        frames[frame][arm_key] = {"position": list(target_list)}

            if robot.ee_type == "gripper":
                val = float(np.clip(action[gripper_key][0], 0, 1))
                scale = robot.gripper_scale
                if robot.gripper_move["sign"] == 1:
                    val = val * (scale[1] - scale[0]) + scale[0]
                else:
                    val = (1 - val) * (scale[1] - scale[0]) + scale[0]
                mimic_a = robot.gripper_move["mimic"][1]
                mimic_b = robot.gripper_move["mimic"][2]
                current_g = gripper_map[env_idx]
                if interval > 0 and current_g is not None:
                    current_pos = current_g[0]
                    for frame in range(interp_count):
                        alpha = (frame + 1) / (interp_count + 1)
                        interp_pos = float(np.clip((1 - alpha) * current_pos + alpha * val, scale[0], scale[1]))
                        frames[frame][gripper_key] = {"position": [interp_pos, interp_pos * mimic_a + mimic_b]}
                    held = float(np.clip(val, scale[0], scale[1]))
                    held_vals = [held, held * mimic_a + mimic_b]
                    for frame in range(interp_count, frame_count):
                        frames[frame][gripper_key] = {"position": list(held_vals)}
                else:
                    raw_vals = [val, val * mimic_a + mimic_b]
                    for frame in range(frame_count):
                        frames[frame][gripper_key] = {"position": list(raw_vals)}

    return sequences
