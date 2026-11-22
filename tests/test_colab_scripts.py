from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_act_colab.py"


def _load_colab_module():
    spec = importlib.util.spec_from_file_location("train_act_colab", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ColabScriptsTests(unittest.TestCase):
    def test_setup_script_has_valid_bash_syntax(self):
        script = REPO_ROOT / "scripts" / "setup_act_colab.sh"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        body = script.read_text(encoding="utf-8")
        for extension in ("isaaclab", "isaaclab_assets", "isaaclab_contrib", "isaaclab_rl", "isaaclab_tasks"):
            self.assertIn(f"IsaacLab/source/{extension}", body)
        self.assertIn("opencv-python-headless transforms3d", body)
        self.assertIn('--extra-index-url https://download.pytorch.org/whl/cu128 -c "${constraints}"', body)

    def test_write_configs_round25_and_round26_defaults(self):
        mod = _load_colab_module()
        sys.path.insert(0, str(REPO_ROOT / "src"))
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            args = types.SimpleNamespace(
                num_envs=16,
                env_workers=1,
                port=5555,
                min_reset_free_vram_mib=2560,
                reset_deadline_seconds=None,
                round=25,
                measure_updates=None,
                finetune_episodes=None,
                act_inference_batch_size=None,
                max_train_seconds=None,
                logs=logs,
            )
            env_cfg, train_cfg, output = mod.write_configs(args, "2026.11.21-10.00.00")
            from starter.config import load_config

            env_data = load_config(env_cfg)
            train_data = load_config(train_cfg)
            self.assertEqual(env_data["num_envs"], 16)
            self.assertEqual(env_data["env_workers"], 1)
            self.assertEqual(env_data["min_reset_free_vram_mib"], 2560)
            self.assertEqual(train_data["max_train_seconds"], 3600)
            self.assertEqual(train_data["finetune_episodes"], 16)
            self.assertEqual(train_data["finetune_updates"], sys.maxsize)
            self.assertEqual(output.name, "2026.11.21-10.00.00")

            self.assertNotIn("act_inference_batch_size", train_data)
            args.round = 26
            args.env_workers = 4
            args.reset_deadline_seconds = 210
            args.act_inference_batch_size = 8
            env_cfg2, train_cfg2, _ = mod.write_configs(args, "2026.11.21-12.00.00")
            self.assertEqual(load_config(env_cfg2)["env_workers"], 4)
            self.assertEqual(load_config(env_cfg2)["reset_deadline_seconds"], 210)
            train_data2 = load_config(train_cfg2)
            self.assertEqual(train_data2["max_train_seconds"], 7200)
            self.assertEqual(train_data2["act_inference_batch_size"], 8)
            self.assertEqual(train_data2["rollout_layout_ids"], list(range(55)))

            args.round = 27
            args.num_envs = 12
            args.max_train_seconds = 3600
            _, train_cfg3, _ = mod.write_configs(args, "2026.11.21-13.00.00")
            train_data3 = load_config(train_cfg3)
            self.assertEqual(train_data3["finetune_episodes"], 24)
            self.assertEqual(train_data3["max_train_seconds"], 3600)
            self.assertEqual(train_data3["rollout_layout_ids"], list(range(55)))
            args.max_train_seconds = None

            args.measure_updates = 1
            _, train_cfg4, _ = mod.write_configs(args, "2026.11.21-14.00.00")
            self.assertEqual(load_config(train_cfg4)["finetune_episodes"], 12)

    def test_drive_credentials_rejects_permanent_oauth_secrets(self):
        mod = _load_colab_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "creds.json"
            for payload in (
                {"access_token": "tok", "refresh_token": "rt"},
                {"access_token": "tok", "client_secret": "cs"},
                {"quota_project": "proj"},
            ):
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    mod._drive_credentials(path)

    def test_backup_checkpoint_unlinks_archive_after_upload_and_failure(self):
        mod = _load_colab_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "output"
            out_dir.mkdir()
            (out_dir / "policy_last.ckpt").write_bytes(b"weights-v1")
            (out_dir / "config.json").write_text('{"ok": true}\n', encoding="utf-8")
            dest = root / "backup.tar.gz"
            snap = root / ".snapshot"
            creds = root / "creds.json"
            creds.write_text('{"access_token": "tok"}', encoding="utf-8")
            with patch.object(mod, "upload", return_value="drive-1"):
                inode, drive_id, digest = mod._backup_checkpoint(out_dir, dest, snap, "stamp", creds)
            self.assertEqual(inode, (out_dir / "policy_last.ckpt").stat().st_ino)
            self.assertEqual(drive_id, "drive-1")
            self.assertEqual(len(digest), 64)
            self.assertFalse(dest.exists())
            self.assertFalse(snap.exists())
            with patch.object(mod, "upload", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    mod._backup_checkpoint(out_dir, dest, snap, "stamp", creds)
            self.assertFalse(dest.exists())
            self.assertFalse(snap.exists())

    def test_checkpoint_archive_uses_external_snapshot_and_leaves_output_clean(self):
        mod = _load_colab_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "output"
            out_dir.mkdir()
            (out_dir / "policy_last.ckpt").write_bytes(b"weights-v1")
            (out_dir / "config.json").write_text('{"ok": true}\n', encoding="utf-8")
            dest = root / "backup.tar.gz"
            snap = root / ".snapshot"
            mod.checkpoint_archive(out_dir, dest, snap, "stamp")
            self.assertFalse(snap.exists())
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["config.json", "policy_last.ckpt"])
            with tarfile.open(dest, "r:gz") as tar:
                self.assertEqual(sorted(tar.getnames()), ["stamp/config.json", "stamp/policy_last.ckpt"])

    def test_checkpoint_archive_recovers_from_a_stale_snapshot_directory(self):
        mod = _load_colab_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "output"
            out_dir.mkdir()
            (out_dir / "policy_last.ckpt").write_bytes(b"weights-v2")
            (out_dir / "config.json").write_text('{"ok": true}\n', encoding="utf-8")
            snap = root / ".snapshot"
            snap.mkdir()
            (snap / "policy_last.ckpt").write_bytes(b"stale")
            dest = root / "backup.tar.gz"
            mod.checkpoint_archive(out_dir, dest, snap, "stamp")
            self.assertFalse(snap.exists())
            with tarfile.open(dest, "r:gz") as tar:
                self.assertEqual(tar.extractfile("stamp/policy_last.ckpt").read(), b"weights-v2")

    def test_upload_rereads_rotated_host_token_after_transient_error(self):
        mod = _load_colab_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "run.tar.gz"
            archive.write_bytes(b"archive-bytes")
            creds = root / "creds.json"
            initial_token = "tok-initial"
            creds.write_text(json.dumps({
                "access_token": initial_token,
                "quota_project": "proj-1",
                "folder_id": "folder-9",
            }), encoding="utf-8")

            calls = []

            class _Resp:
                def __init__(self, body: bytes, headers=None):
                    self._body = io.BytesIO(body)
                    self.headers = headers or {}

                def read(self, *a, **kw):
                    return self._body.read(*a, **kw)

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

            def fake_urlopen(req, timeout=0):
                calls.append((req.get_method(), req.full_url, dict(req.header_items())))
                if req.get_method() == "POST":
                    if len(calls) == 1:
                        config = json.loads(creds.read_text())
                        config["access_token"] = "tok-new"
                        creds.write_text(json.dumps(config))
                        raise HTTPError(req.full_url, 401, "expired", {}, None)
                    return _Resp(b"", headers={"Location": "https://www.googleapis.com/upload/session/1"})
                return _Resp(json.dumps({"id": "file-123", "size": str(archive.stat().st_size),
                                         "md5Checksum": hashlib.md5(archive.read_bytes()).hexdigest()}).encode())

            with patch.object(mod, "urlopen", side_effect=fake_urlopen), patch.object(mod.time, "sleep"):
                file_id = mod.upload(archive, creds, max_attempts=3)

            self.assertEqual(file_id, "file-123")
            self.assertEqual([call[0] for call in calls], ["POST", "POST", "PUT"])
            self.assertEqual([call[2]["Authorization"] for call in calls],
                             [f"Bearer {initial_token}", "Bearer tok-new", "Bearer tok-new"])
            self.assertEqual(calls[-1][2].get("X-goog-user-project"), "proj-1")

    def test_build_runtime_env_includes_blackwell_and_egl_vars(self):
        mod = _load_colab_module()
        env = mod.build_runtime_env(Path("/content/env"), base_env={"PATH": "/usr/bin"})
        self.assertEqual(env["NVIDIA_DRIVER_CAPABILITIES"], "all")
        self.assertEqual(env["TORCH_CUDA_ARCH_LIST"], "12.0")
        self.assertEqual(env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO"], "0.8.0")
        self.assertEqual(env["OMNI_CRASHREPORTER_ENABLED"], "0")
        self.assertEqual(env["VK_ICD_FILENAMES"], "/content/env/nvidia_icd.json")

    def test_build_sim_commands_single_and_multi_worker(self):
        mod = _load_colab_module()
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            args = types.SimpleNamespace(
                environment=Path("/content/env"),
                logs=logs,
                num_envs=64,
                env_workers=1,
                port=5555,
                min_reset_free_vram_mib=2560,
                reset_deadline_seconds=None,
                tasking_threads=None,
                physx_threads=None,
            )
            single = mod.build_sim_commands(args, "stamp1")
            self.assertEqual(len(single), 1)
            cmd, log_path, extra_env, cache_root = single[0]
            self.assertEqual(log_path, logs / "isaac_server.log")
            self.assertEqual(extra_env, {})
            self.assertIsNone(cache_root)
            self.assertNotIn("--coordinated-recycle", cmd)
            self.assertNotIn("--reset-deadline-seconds", cmd)
            self.assertEqual(cmd[cmd.index("--num-envs") + 1], "64")

            args.env_workers = 4
            args.reset_deadline_seconds = 210
            args.tasking_threads = 8
            args.physx_threads = 4
            multi = mod.build_sim_commands(args, "stamp2")
            self.assertEqual(len(multi), 4)
            ports = []
            cache_roots = set()
            for idx, (cmd, log_path, extra_env, cache_root) in enumerate(multi):
                self.assertEqual(log_path, logs / f"isaac_server_w{idx}.log")
                self.assertEqual(cmd[cmd.index("--num-envs") + 1], "16")
                ports.append(int(cmd[cmd.index("--port") + 1]))
                self.assertIn("--coordinated-recycle", cmd)
                self.assertEqual(cmd[cmd.index("--reset-deadline-seconds") + 1], "210")
                self.assertIn("--portable-root", cmd)
                kit_args = cmd[cmd.index("--kit_args") + 1]
                self.assertNotIn("ujitsoCollisionCooking", kit_args)
                self.assertIn("--/plugins/carb.tasking.plugin/threadCount=8", kit_args)
                self.assertIn("--/persistent/physics/numThreads=4", kit_args)
                self.assertIsNotNone(cache_root)
                cache_roots.add(cache_root)
                self.assertEqual(extra_env, {"XDG_CACHE_HOME": str(cache_root / "xdg")})
                self.assertEqual(cmd[cmd.index("--portable-root") + 1], str(cache_root / "portable"))
            self.assertEqual(ports, [5555, 5556, 5557, 5558])
            self.assertEqual(len(cache_roots), 4)

    def test_launcher_stops_learner_only_on_supervisor_failure(self):
        for fail in (False, True):
            with self.subTest(supervisor_failure=fail), tempfile.TemporaryDirectory() as tmp:
                mod = _load_colab_module()
                root = Path(tmp)
                (root / "ready").touch()
                (root / "config.json").write_text('{"policy_type": "act_residual_ppo"}')
                (root / "base.ckpt").touch()
                logs = root / "logs"
                argv = [str(SCRIPT_PATH), "--environment", str(root), "--checkpoint", str(root / "base.ckpt"),
                        "--logs", str(logs), "--round", "26", "--num-envs", "4", "--env-workers", "2",
                        "--min-reset-free-vram-mib", "2560", "--measure-updates", "1"]
                learner, sim0, sim1 = (Mock(pid=i) for i in range(3))
                for proc in (learner, sim0, sim1):
                    proc.returncode = None
                    proc.poll.side_effect = lambda proc=proc: proc.returncode
                    proc.terminate.side_effect = lambda proc=proc: setattr(proc, "returncode", -15)
                procs = [learner, sim0, OSError("cannot launch worker 1") if fail else sim1]
                gpu = "2026/11/21 20:42:30, NVIDIA RTX PRO 6000, 100, 97887, 0, 97151"

                def learner_finishes(seconds):
                    if seconds == 1:
                        learner.returncode = 0

                with (patch.object(sys, "argv", argv), patch.object(sys, "path", list(sys.path)),
                      patch.object(mod, "write_configs", return_value=(logs / "env.yaml", logs / "online.yaml", root / "out")),
                      patch.object(mod.subprocess, "Popen", side_effect=procs),
                      patch.object(mod.subprocess, "check_output", return_value=gpu),
                      patch.object(mod.time, "sleep", side_effect=learner_finishes)):
                    if fail:
                        with self.assertRaisesRegex(OSError, "cannot launch worker 1"):
                            mod.main()
                        learner.terminate.assert_called_once_with()
                    else:
                        self.assertEqual(mod.main(), 0)
                        learner.terminate.assert_not_called()
                        sim1.terminate.assert_called_once_with()
                sim0.terminate.assert_called_once_with()
                status = json.loads((logs / "status.json").read_text())
                self.assertEqual(status["state"], "finished")
                self.assertEqual(status["learner_exit"], -15 if fail else 0)
                self.assertEqual("supervisor_error" in status, fail)
                self.assertTrue(Path(status["archive"]).is_file())
                self.assertEqual(list(root.glob(".*-worker-*-cache")), [])


if __name__ == "__main__":
    unittest.main()
