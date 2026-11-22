"""Launch one native Isaac/ACT round and preserve an archive on Google Drive."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import math
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


ROOT = Path(__file__).resolve().parents[1]
DOJO = ROOT / "third_party/RoboDojo"


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _drive_credentials(path: Path) -> dict:
    credentials = json.loads(path.read_text(encoding="utf-8"))
    if "refresh_token" in credentials or "client_secret" in credentials:
        raise ValueError("Keep permanent OAuth credentials on the host; use a short-lived Drive access token.")
    if not credentials.get("access_token"):
        raise ValueError("Drive credentials require an access token.")
    return credentials


def upload(archive: Path, credentials_path: Path, *, max_attempts: int = 3) -> str:
    size = str(archive.stat().st_size)
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "md5").hexdigest()
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            credentials = _drive_credentials(credentials_path)
            metadata = {"name": archive.name}
            if credentials.get("folder_id"):
                metadata["parents"] = [credentials["folder_id"]]
            headers = {"Authorization": f"Bearer {credentials['access_token']}"}
            if credentials.get("quota_project"):
                headers["x-goog-user-project"] = credentials["quota_project"]
            request = Request(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id,size,md5Checksum",
                data=json.dumps(metadata).encode("utf-8"),
                headers={
                    **headers,
                    "Content-Type": "application/json",
                    "X-Upload-Content-Type": "application/gzip",
                    "X-Upload-Content-Length": size,
                },
                method="POST",
            )
            with urlopen(request, timeout=30) as response:
                location = response.headers["Location"]
            if not location or not location.startswith("https://www.googleapis.com/"):
                raise RuntimeError("Drive returned an unexpected upload endpoint.")
            with archive.open("rb") as stream:
                put_request = Request(
                    location,
                    data=stream,
                    method="PUT",
                    headers={
                        **headers,
                        "Content-Type": "application/gzip",
                        "Content-Length": size,
                    },
                )
                with urlopen(put_request, timeout=300) as response:
                    result = json.load(response)
            if result.get("size") != size or result.get("md5Checksum") != checksum or not result.get("id"):
                raise RuntimeError("Drive upload did not confirm the file ID, size and checksum.")
            return result["id"]
        except HTTPError as error:
            last_error = RuntimeError(f"Drive upload failed with HTTP {error.code}.")
            if error.code not in (401, 408, 429, 500, 502, 503, 504) or attempt == max_attempts:
                raise last_error from None
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == max_attempts:
                raise RuntimeError(f"Drive upload failed after {max_attempts} attempts: {error}") from None
        time.sleep(min(2.0 ** attempt, 10.0))
    raise RuntimeError(f"Drive upload failed: {last_error}")


def checkpoint_archive(source: Path, destination: Path, snapshot: Path, run_stamp: str) -> int:
    # Keep the exact checkpoint inode while the learner atomically publishes a newer one.
    snapshot.mkdir(parents=True, exist_ok=True)
    link = snapshot / "policy_last.ckpt"
    link.unlink(missing_ok=True)
    try:
        os.link(source / link.name, link)
        inode = link.stat().st_ino
        with tarfile.open(destination, "x:gz", compresslevel=1) as archive:
            archive.add(link, arcname=f"{run_stamp}/{link.name}")
            archive.add(source / "config.json", arcname=f"{run_stamp}/config.json")
        return inode
    finally:
        link.unlink(missing_ok=True)
        shutil.rmtree(snapshot, ignore_errors=True)


def _backup_checkpoint(
    source: Path, destination: Path, snapshot: Path, run_stamp: str, credentials_path: Path,
) -> tuple[int, str, str]:
    try:
        inode = checkpoint_archive(source, destination, snapshot, run_stamp)
        return inode, upload(destination, credentials_path), sha256_file(destination)
    finally:
        destination.unlink(missing_ok=True)


def write_configs(args, run_stamp: str) -> tuple[Path, Path, Path]:
    env = yaml.safe_load((ROOT / "configs/env.yaml").read_text(encoding="utf-8"))
    online = yaml.safe_load((ROOT / "configs/act_online.yaml").read_text(encoding="utf-8"))
    env.update(
        num_envs=args.num_envs,
        env_workers=args.env_workers,
        env_port=args.port,
        min_reset_free_vram_mib=args.min_reset_free_vram_mib,
    )
    if args.reset_deadline_seconds is not None:
        env["reset_deadline_seconds"] = args.reset_deadline_seconds
    budget = args.max_train_seconds if args.max_train_seconds is not None else (3600 if args.round == 25 else 7200)
    episodes = (args.finetune_episodes if args.finetune_episodes is not None else
                args.num_envs if args.measure_updates else
                math.ceil(online["finetune_episodes"] / args.num_envs) * args.num_envs)
    online.update(
        finetune_updates=args.measure_updates or sys.maxsize,
        finetune_episodes=episodes,
        max_train_seconds=budget,
        run_stamp=run_stamp,
    )
    if args.act_inference_batch_size is not None:
        online["act_inference_batch_size"] = args.act_inference_batch_size
    env_config, online_config = (args.logs / name for name in ("env.yaml", "act_online.yaml"))
    for path, data in ((env_config, env), (online_config, online)):
        # The repo's simple YAML reader requires indentation before list items.
        from starter.config import load_config

        class IndentedDumper(yaml.SafeDumper):
            def increase_indent(self, flow=False, indentless=False):
                return super().increase_indent(flow, False)

        path.write_text(yaml.dump(data, Dumper=IndentedDumper, sort_keys=False), encoding="utf-8")
        load_config(path)
    from starter.paths import OUTPUTS_ROOT, run_name

    return env_config, online_config, OUTPUTS_ROOT / "act_residual" / run_name(env) / run_stamp


def build_runtime_env(environment: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    source = dict(os.environ if base_env is None else base_env)
    system_libs = environment / "system/usr/lib/x86_64-linux-gnu"
    ld_parts = [str(system_libs)]
    if Path("/usr/lib64-nvidia").is_dir():
        ld_parts.append("/usr/lib64-nvidia")
    if source.get("LD_LIBRARY_PATH"):
        ld_parts.append(source["LD_LIBRARY_PATH"])
    cuda_home = environment / "system/usr/local/cuda-12.8"
    env = dict(
        source,
        ACCEPT_EULA="Y",
        PRIVACY_CONSENT="N",
        OMNI_KIT_ACCEPT_EULA="YES",
        OMNI_CRASHREPORTER_ENABLED="0",
        NVIDIA_DRIVER_CAPABILITIES="all",
        TORCH_CUDA_ARCH_LIST="12.0",
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="0.8.0",
        CUDA_VISIBLE_DEVICES="0",
        MPLBACKEND="Agg",
        PYTHONPATH=f"{ROOT / 'src'}:{DOJO}",
        STACK_BLOCKS_ROOT=str(ROOT),
        VK_DRIVER_FILES=str(environment / "nvidia_icd.json"),
        VK_ICD_FILENAMES=str(environment / "nvidia_icd.json"),
        LD_LIBRARY_PATH=os.pathsep.join(ld_parts),
        CUDA_HOME=str(cuda_home),
    )
    env["PATH"] = os.pathsep.join((str(cuda_home / "bin"), env.get("PATH", "")))
    return env


def build_sim_commands(
    args, run_stamp: str = "run",
) -> list[tuple[list[str | Path], Path, dict[str, str], Path | None]]:
    per_worker = args.num_envs // args.env_workers
    kit_parts = [
        "--enable isaacsim.replicator.behavior",
        "--enable isaacsim.sensors.camera",
        "--/app/hangDetector/enabled=false",
    ]
    if args.physx_threads is not None:
        kit_parts.append(f"--/persistent/physics/numThreads={args.physx_threads}")
    if args.tasking_threads is not None:
        kit_parts.append(f"--/plugins/carb.tasking.plugin/threadCount={args.tasking_threads}")
        kit_parts.append(f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={args.tasking_threads}")
    base_kit_args = " ".join(kit_parts)
    commands: list[tuple[list[str | Path], Path, dict[str, str], Path | None]] = []
    for index in range(args.env_workers):
        extra_env: dict[str, str] = {}
        cache_root: Path | None = None
        kit_args = base_kit_args
        cmd: list[str | Path] = [
            args.environment / "sim/bin/python", "-u", ROOT / "scripts/isaac_env_server.py",
            "--task", "stack_blocks", "--env-cfg", "arx_x5", "--port", str(args.port + index),
            "--num-envs", str(per_worker), "--headless", "--enable_cameras", "--device", "cuda:0",
            "--min-reset-free-vram-mib", str(args.min_reset_free_vram_mib),
        ]
        if args.reset_deadline_seconds is not None:
            cmd.extend(["--reset-deadline-seconds", str(args.reset_deadline_seconds)])
        if args.env_workers > 1:
            cache_root = args.logs.parent / f".{run_stamp}-worker-{index}-cache"
            extra_env = {"XDG_CACHE_HOME": str(cache_root / "xdg")}
            kit_args = (
                f"{base_kit_args}"
                f" --/app/tokens/cache={cache_root / 'cache'}"
                f" --/app/tokens/data={cache_root / 'data'}"
                f" --/app/tokens/logs={cache_root / 'logs'}"
            )
            cmd.extend(["--coordinated-recycle", "--portable-root", str(cache_root / "portable")])
        cmd.extend(["--kit_args", kit_args])
        log_name = "isaac_server.log" if args.env_workers == 1 else f"isaac_server_w{index}.log"
        commands.append((cmd, args.logs / log_name, extra_env, cache_root))
    return commands


def _stop_processes(processes) -> None:
    alive = [proc for proc in processes if proc is not None and proc.poll() is None]
    for proc in alive:
        proc.terminate()
    deadline = time.monotonic() + 15
    for proc in alive:
        try:
            proc.wait(timeout=max(deadline - time.monotonic(), 0.0))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--num-envs", required=True, type=int)
    parser.add_argument("--env-workers", type=int, default=1)
    parser.add_argument("--min-reset-free-vram-mib", required=True, type=int)
    parser.add_argument("--reset-deadline-seconds", type=float,
                        help="Restart a simulator if its reset exceeds this measured deadline.")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--worker-stagger-seconds", type=float, default=3.0)
    parser.add_argument("--tasking-threads", type=int)
    parser.add_argument("--physx-threads", type=int)
    parser.add_argument("--finetune-episodes", type=int)
    parser.add_argument("--act-inference-batch-size", type=int)
    parser.add_argument("--max-train-seconds", type=int)
    parser.add_argument("--drive-credentials-file", type=Path)
    parser.add_argument("--measure-updates", type=int)
    args = parser.parse_args()
    if (args.round < 25 or args.num_envs < 1 or args.env_workers < 1
            or args.min_reset_free_vram_mib < 1
            or args.worker_stagger_seconds < 0
            or (args.tasking_threads is not None and args.tasking_threads < 1)
            or (args.physx_threads is not None and args.physx_threads < 1)
            or (args.measure_updates is not None and args.measure_updates < 1)
            or (args.finetune_episodes is not None and args.finetune_episodes < 1)
            or (args.max_train_seconds is not None and args.max_train_seconds < 1)):
        parser.error("Require round >= 25 and positive scene, worker, episode, budget and headroom counts.")
    if args.reset_deadline_seconds is not None and (
        not math.isfinite(args.reset_deadline_seconds) or args.reset_deadline_seconds <= 0
    ):
        parser.error("--reset-deadline-seconds must be positive and finite.")
    if args.num_envs % args.env_workers != 0:
        parser.error(f"--num-envs ({args.num_envs}) must be divisible by --env-workers ({args.env_workers}).")
    if args.port < 1 or args.port + args.env_workers - 1 > 65535:
        parser.error("Port range for --env-workers must lie in 1..65535.")
    if args.act_inference_batch_size is not None and args.act_inference_batch_size < 1:
        parser.error("--act-inference-batch-size must be positive.")
    if not args.measure_updates:
        if args.drive_credentials_file is None or not args.drive_credentials_file.is_file():
            parser.error("Formal training requires a readable private Drive credentials file.")
    if args.drive_credentials_file:
        args.drive_credentials_file = args.drive_credentials_file.resolve(strict=True)
        _drive_credentials(args.drive_credentials_file)
    args.environment = args.environment.resolve()
    args.checkpoint = args.checkpoint.resolve(strict=True)
    config = json.loads(args.checkpoint.with_name("config.json").read_text(encoding="utf-8"))
    is_residual = config.get("policy_type") == "act_residual_ppo"
    if not args.measure_updates and is_residual != (args.round > 25):
        parser.error("Round 25 starts from offline ACT; later rounds resume ACT residual PPO.")
    if not is_residual and not args.checkpoint.with_name("dataset_stats.pkl").is_file():
        parser.error("Offline ACT requires dataset_stats.pkl next to the checkpoint.")
    args.logs = args.logs.resolve()
    if not (args.environment / "ready").is_file():
        parser.error("Run setup_act_colab.sh before starting training.")
    args.logs.mkdir(parents=True, exist_ok=False)
    run_stamp = time.strftime("%Y.%m.%d-%H.%M.%S")
    sys.path.insert(0, str(ROOT / "src"))
    env_cfg, train_cfg, output = write_configs(args, run_stamp)
    if output.exists():
        raise FileExistsError(output)

    env = build_runtime_env(args.environment)
    sim_specs = build_sim_commands(args, run_stamp)
    for _, _, extra_env, cache_root in sim_specs:
        if cache_root is not None:
            for sub in ("cache", "data", "logs", "portable", "xdg"):
                (cache_root / sub).mkdir(parents=True, exist_ok=True)
    learner_cmd = [
        args.environment / "learner/bin/python", "-u", "-m", "starter.pipelines.act", "online",
        "--config", env_cfg, "--online-config", train_cfg, "--checkpoint", args.checkpoint,
    ]
    gpu_query = [
        "nvidia-smi", "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,memory.free",
        "--format=csv,noheader,nounits", "-i", "0",
    ]
    gpu = subprocess.check_output(gpu_query, text=True).strip()
    if "RTX PRO 6000" not in gpu or int(gpu.split(",")[3]) < 95000:
        raise RuntimeError(f"Expected a 96 GB G4 runtime, got {gpu}.")
    sims: list[subprocess.Popen | None] = [None] * len(sim_specs)
    restart_at = [0.0] * len(sim_specs)
    worker_restarts = [0] * len(sim_specs)
    worker_exits: list[int | None] = [None] * len(sim_specs)
    learner = None
    checkpoint = output / "policy_last.ckpt"
    archived_inode = None
    last_backup = float("-inf")
    backup_job = None
    status = {
        "run_stamp": run_stamp,
        "num_envs": args.num_envs,
        "env_workers": args.env_workers,
        "per_worker_envs": args.num_envs // args.env_workers,
        "output": str(output),
    }
    status_path = args.logs / "status.json"

    def write_status(**fields):
        status.update(fields)
        pending = status_path.with_suffix(".tmp")
        pending.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        pending.replace(status_path)

    def collect_backup(wait: bool) -> None:
        nonlocal archived_inode, backup_job, last_backup
        if backup_job is None or (not wait and not backup_job.done()):
            return
        try:
            archived_inode, backup_id, backup_hash = backup_job.result()
            write_status(backup_drive_id=backup_id, backup_sha256=backup_hash, backup_error=None)
        except Exception as error:
            write_status(backup_error=str(error))
        backup_job = None
        last_backup = time.monotonic()

    try:
        with ExitStack() as stack:
            sim_logs = [stack.enter_context(log_path.open("ab")) for _, log_path, _, _ in sim_specs]
            learner_log = stack.enter_context((args.logs / "learner.log").open("ab"))
            resources = stack.enter_context((args.logs / "resources.csv").open("a", buffering=1))
            uploader = stack.enter_context(ThreadPoolExecutor(max_workers=1))
            learner = subprocess.Popen(
                learner_cmd, stdout=learner_log, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
            )
            while learner.poll() is None:
                spawned_this_tick = False
                for index, (sim_cmd, _log_path, extra_env, _cache_root) in enumerate(sim_specs):
                    proc = sims[index]
                    now = time.monotonic()
                    if (proc is None or proc.poll() is not None) and now >= restart_at[index]:
                        if proc is not None:
                            worker_exits[index] = proc.returncode
                            worker_restarts[index] += 1
                            write_status(
                                simulator_exit=proc.returncode,
                                simulator_exits=list(worker_exits),
                                worker_restarts=list(worker_restarts),
                            )
                        if spawned_this_tick and args.worker_stagger_seconds > 0:
                            time.sleep(args.worker_stagger_seconds)
                        worker_env = {**env, **extra_env} if extra_env else env
                        sims[index] = subprocess.Popen(
                            sim_cmd, stdout=sim_logs[index], stderr=subprocess.STDOUT, cwd=DOJO, env=worker_env,
                        )
                        restart_at[index] = time.monotonic() + 15
                        spawned_this_tick = True
                sim_pids = [proc.pid if proc is not None else None for proc in sims]
                try:
                    use = subprocess.check_output(gpu_query, text=True, timeout=10).strip()
                    resources.write(use + "\n")
                    write_status(
                        learner_pid=learner.pid,
                        simulator_pid=sim_pids[0],
                        simulator_pids=sim_pids,
                        heartbeat=time.time(),
                        gpu=use,
                        gpu_query_error=None,
                        state="running",
                    )
                except (subprocess.SubprocessError, OSError) as error:
                    write_status(
                        learner_pid=learner.pid,
                        simulator_pid=sim_pids[0],
                        simulator_pids=sim_pids,
                        heartbeat=time.time(),
                        gpu_query_error=str(error),
                        state="running",
                    )
                collect_backup(wait=False)
                if (backup_job is None and args.drive_credentials_file and checkpoint.is_file()
                        and checkpoint.stat().st_ino != archived_inode
                        and time.monotonic() - last_backup >= 900):
                    backup = args.logs.parent / f"{run_stamp}-checkpoint-{time.time_ns()}.tar.gz"
                    snapshot = args.logs.parent / f".{run_stamp}-snapshot"
                    backup_job = uploader.submit(
                        _backup_checkpoint, output, backup, snapshot, run_stamp, args.drive_credentials_file,
                    )
                time.sleep(1)
            collect_backup(wait=True)
            write_status(learner_exit=learner.returncode, state="packaging")
    except (Exception, KeyboardInterrupt) as error:
        # A failed supervisor cannot restart Isaac or back up a live learner.
        _stop_processes([learner, *sims])
        write_status(supervisor_error=f"{type(error).__name__}: {error}",
                     learner_exit=learner.returncode if learner is not None else None, state="packaging")
        raise
    finally:
        collect_backup(wait=True)
        _stop_processes(sims)
        for _, _, _, cache_root in sim_specs:
            if cache_root is not None:
                shutil.rmtree(cache_root, ignore_errors=True)
        if learner is not None:
            archive = args.logs.parent / f"{run_stamp}.tar.gz"
            with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
                if output.is_dir():
                    bundle.add(output, arcname=run_stamp)
                bundle.add(args.logs, arcname=f"{run_stamp}/logs")
            archive_sha256 = sha256_file(archive)
            drive_id = None
            upload_error = None
            if args.drive_credentials_file:
                try:
                    drive_id = upload(archive, args.drive_credentials_file)
                except Exception as error:
                    upload_error = str(error)
            write_status(
                state="finished",
                archive=str(archive),
                archive_sha256=archive_sha256,
                archive_drive_id=drive_id,
                archive_upload_error=upload_error,
            )
    return learner.returncode


if __name__ == "__main__":
    raise SystemExit(main())
