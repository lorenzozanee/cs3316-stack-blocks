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
from unittest.mock import patch
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
                port=5555,
                min_reset_free_vram_mib=2560,
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
            self.assertEqual(env_data["min_reset_free_vram_mib"], 2560)
            self.assertEqual(train_data["max_train_seconds"], 3600)
            self.assertEqual(train_data["finetune_episodes"], 16)
            self.assertEqual(train_data["finetune_updates"], sys.maxsize)
            self.assertEqual(output.name, "2026.11.21-10.00.00")

            self.assertNotIn("act_inference_batch_size", train_data)
            args.round = 26
            args.act_inference_batch_size = 8
            _, train_cfg2, _ = mod.write_configs(args, "2026.11.21-12.00.00")
            train_data2 = load_config(train_cfg2)
            self.assertEqual(train_data2["max_train_seconds"], 7200)
            self.assertEqual(train_data2["act_inference_batch_size"], 8)

            args.num_envs = 12
            _, train_cfg3, _ = mod.write_configs(args, "2026.11.21-13.00.00")
            self.assertEqual(load_config(train_cfg3)["finetune_episodes"], 24)

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


if __name__ == "__main__":
    unittest.main()
