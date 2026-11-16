"""Planner configs use mounted container paths, never stale host asset paths."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from starter.envs.assets import prepare_robot_configs


class RobotAssetConfigTest(unittest.TestCase):
    def _fixture(self, root):
        configs = root / "env_cfg"
        (configs / "robot").mkdir(parents=True)
        (configs / "arx_x5.yml").write_text("config:\n  robot: dual_x5\n")
        (configs / "robot/dual_x5.yml").write_text(
            "robots:\n  - robot_name: x5\n  - robot_name: x5\n"
        )
        assets = root / "shared_assets"
        robot = assets / "Robots/x5"
        robot.mkdir(parents=True)
        (robot / "X5A.urdf").write_text('<robot name="x5"/>')
        (robot / "curobo.yml").write_text("stale generated configuration; keep unchanged\n")
        (robot / "curobo_tmp.yml").write_text(
            "robot_cfg:\n  kinematics:\n"
            "    asset_root_path: ${ASSETS_PATH}/Assets/Robots/x5\n"
            "    urdf_path: $ASSETS_PATH/Assets/Robots/x5/X5A.urdf\n"
        )
        return configs / "arx_x5.yml", assets, root / "generated", Path("/sim/RoboDojo")

    def test_render_deduplicates_robots_and_preserves_shared_asset_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            source = args[1] / "Robots/x5/curobo.yml"
            before = source.read_bytes()
            mounts = prepare_robot_configs(*args)
            self.assertEqual(len(mounts), 1)
            generated, target = mounts[0]
            self.assertEqual(target, Path("/sim/RoboDojo/Assets/Robots/x5/curobo.yml"))
            self.assertEqual(generated, args[2] / "x5/curobo.yml")
            values = yaml.safe_load(generated.read_text())["robot_cfg"]["kinematics"]
            self.assertEqual(values["asset_root_path"], "/sim/RoboDojo/Assets/Robots/x5")
            self.assertEqual(values["urdf_path"], "/sim/RoboDojo/Assets/Robots/x5/X5A.urdf")
            self.assertNotIn("ASSETS_PATH", generated.read_text())
            self.assertEqual(source.read_bytes(), before)
            # Regenerating at startup replaces an obsolete generated output.
            generated.write_text("obsolete output")
            self.assertEqual(prepare_robot_configs(*args), mounts)
            self.assertEqual(yaml.safe_load(generated.read_text())["robot_cfg"]["kinematics"], values)

    def test_missing_urdf_fails_before_generating_a_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            (args[1] / "Robots/x5/X5A.urdf").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_robot_configs(*args)
            self.assertFalse(args[2].exists())

    def test_stale_template_path_fails_instead_of_using_generated_host_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            template = args[1] / "Robots/x5/curobo_tmp.yml"
            template.write_text(template.read_text().replace("${ASSETS_PATH}", "/obsolete_checkout"))
            with self.assertRaisesRegex(ValueError, "container asset tree"):
                prepare_robot_configs(*args)


if __name__ == "__main__":
    unittest.main()
