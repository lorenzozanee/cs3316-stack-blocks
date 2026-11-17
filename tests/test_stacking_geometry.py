"""Geometry helper tests; no Isaac Sim is started."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from starter.envs.stacking_geometry import stacking_geometry
from starter.interfaces import BLOCK_LABELS

ASSET_BLOCK_DIR = (
    Path(__file__).resolve().parents[1]
    / "third_party/RoboDojo/Assets/Object/RoboDojo/Rigid/block"
)


class _FakeObject:
    def __init__(self, scale=(1.0, 1.0, 1.0)) -> None:
        self.scale = scale


class _FakeLayoutManager:
    """Mirrors the real call signatures used by the helper."""

    def __init__(self, table_info, metadata, scales=None) -> None:
        self.table_info = table_info
        self._metadata = dict(metadata)
        self._scales = dict(scales or {})
        self.metadata_calls = []
        self.object_calls = []

    def get_instance_metadata(self, env_idx, inst_name=None, label=None):
        self.metadata_calls.append((env_idx, inst_name, label))
        return self._metadata.get((env_idx, label))

    def get_instance_name(self, env_idx, label):
        return f"{label}_env{env_idx}"

    def get_scene_object(self, env_idx, inst_name):
        self.object_calls.append((env_idx, inst_name))
        label = inst_name.split("_env", 1)[0]
        return _FakeObject(self._scales.get((env_idx, label), (1.0, 1.0, 1.0)))


def _task(layout_manager):
    return SimpleNamespace(scene_manager=SimpleNamespace(layout_manager=layout_manager))


def _table(height=0.765, pos=(0.0, -0.05, 0.74), size=(-0.7, -0.55, 0.7, 0.55)):
    return {"height": height, "pos": list(pos), "size": list(size)}


def _metadata(extents):
    return {"geometry": {"oriented_bbox": {"extents": list(extents)}}}


def _metadata_for(table_info, metadata_by_env, scales=None):
    return _FakeLayoutManager(table_info, metadata_by_env, scales)


def _default_metadata(env_count, extents=(0.03, 0.03, 0.03)):
    return {(env_idx, label): _metadata(extents) for env_idx in range(env_count) for label in BLOCK_LABELS}


class StackingGeometryTest(unittest.TestCase):
    def test_reads_per_scene_table_and_extents_for_two_origins(self):
        table_info = [
            _table(pos=(0.0, -0.05, 0.74)),
            _table(height=1.0, pos=(2.0, 3.0, 0.98), size=(-0.3, -0.2, 0.4, 0.6)),
        ]
        metadata = _default_metadata(2)
        metadata[(0, "block_1")] = _metadata((0.035, 0.04, 0.045))
        metadata[(1, "block_2")] = _metadata((0.05, 0.06, 0.07))
        manager = _metadata_for(table_info, metadata)

        first = stacking_geometry(_task(manager), 0)
        second = stacking_geometry(_task(manager), 1)

        self.assertEqual(
            list(first["table"]), ["top_z", "x_min", "x_max", "y_min", "y_max"]
        )
        for key, expected in {
            "top_z": 0.765,
            "x_min": -0.7,
            "x_max": 0.7,
            "y_min": -0.6,
            "y_max": 0.5,
        }.items():
            self.assertAlmostEqual(first["table"][key], expected, places=12)
        for key, expected in {
            "top_z": 1.0,
            "x_min": 1.7,
            "x_max": 2.4,
            "y_min": 2.8,
            "y_max": 3.6,
        }.items():
            self.assertAlmostEqual(second["table"][key], expected, places=12)
        self.assertEqual(first["block_extents"]["block_0"], [0.03, 0.03, 0.03])
        self.assertEqual(first["block_extents"]["block_1"], [0.035, 0.04, 0.045])
        self.assertEqual(second["block_extents"]["block_2"], [0.05, 0.06, 0.07])
        self.assertEqual(manager.metadata_calls[0], (0, None, "block_0"))
        self.assertEqual(manager.object_calls[0], (0, "block_0_env0"))

    def test_extents_are_multiplied_by_instance_scale(self):
        manager = _metadata_for(
            [_table()],
            _default_metadata(1),
            scales={(0, "block_0"): (2.0, 3.0, 4.0)},
        )
        result = stacking_geometry(_task(manager), 0)
        self.assertEqual(result["block_extents"]["block_0"], [0.06, 0.09, 0.12])
        self.assertEqual(result["block_extents"]["block_1"], [0.03, 0.03, 0.03])

    def test_output_is_plain_float_list_data(self):
        manager = _metadata_for([_table()], _default_metadata(1))
        result = stacking_geometry(_task(manager), 0)
        self.assertIs(type(result["table"]["top_z"]), float)
        self.assertIs(type(result["table"]["x_min"]), float)
        self.assertIs(type(result["block_extents"]["block_0"]), list)
        self.assertTrue(all(type(value) is float for value in result["block_extents"]["block_0"]))
        for label in BLOCK_LABELS:
            extents = result["block_extents"][label]
            self.assertEqual(len(extents), 3)
            self.assertTrue(all(math.isfinite(value) and value > 0.0 for value in extents))

    def test_missing_layout_manager_apis_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, "task.scene_manager"):
            stacking_geometry(SimpleNamespace(), 0)
        task = SimpleNamespace(scene_manager=SimpleNamespace(layout_manager=SimpleNamespace()))
        with self.assertRaisesRegex(ValueError, "table_info"):
            stacking_geometry(task, 0)

        class _NoMetadata:
            table_info = [_table()]

        task = _task(_NoMetadata())
        with self.assertRaisesRegex(ValueError, "get_instance_metadata"):
            stacking_geometry(task, 0)

        class _NoScaleApi:
            table_info = [_table()]

            def get_instance_metadata(self, env_idx, inst_name=None, label=None):
                del env_idx, inst_name, label
                return _metadata((0.03, 0.03, 0.03))

        task = _task(_NoScaleApi())
        with self.assertRaisesRegex(ValueError, "get_instance_name"):
            stacking_geometry(task, 0)

    def test_missing_or_bad_table_info_raises(self):
        cases = [
            ([], 0, "outside"),
            ([{}], 0, "empty"),
            ([{"pos": [0, 0, 0], "size": [-1, -1, 1, 1]}], 0, "missing"),
            ([_table(height=float("nan"))], 0, "finite"),
            ([_table(pos=(0.0, 0.0))], 0, "length 3"),
            ([_table(size=(-1.0, -1.0, 1.0))], 0, "length 4"),
            ([_table(size=(-0.7, -0.55, -0.8, 0.55))], 0, "empty x"),
            ([_table(size=(-0.7, -0.55, 0.7, -0.6))], 0, "empty y"),
        ]
        for table_info, env_index, message in cases:
            manager = _metadata_for(table_info, _default_metadata(1))
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    stacking_geometry(_task(manager), env_index)

    def test_missing_or_bad_metadata_raises(self):
        bad = {
            "missing": None,
            "geometry": {"oriented_bbox": {"extents": [0.03, 0.03, 0.03]}},
            "oriented": {"geometry": {"oriented_bbox": {}}},
            "length2": _metadata((0.03, 0.03)),
            "length4": _metadata((0.03, 0.03, 0.03, 0.03)),
            "nan": _metadata((0.03, float("nan"), 0.03)),
            "zero": _metadata((0.0, 0.03, 0.03)),
            "negative": _metadata((0.03, -0.03, 0.03)),
        }
        for name, metadata in bad.items():
            with self.subTest(name=name):
                data = _default_metadata(1)
                data[(0, "block_2")] = metadata
                manager = _metadata_for([_table()], data)
                with self.assertRaises(ValueError):
                    stacking_geometry(_task(manager), 0)

    def test_missing_or_bad_instance_scale_raises(self):
        class _NoScale(_FakeObject):
            def __init__(self):
                pass

        manager = _metadata_for([_table()], _default_metadata(1))
        manager.get_scene_object = lambda env_idx, inst_name: _NoScale()
        with self.assertRaisesRegex(ValueError, "no scale"):
            stacking_geometry(_task(manager), 0)

        for scale in [(1.0, 1.0), (1.0, 0.0, 1.0), (1.0, float("inf"), 1.0)]:
            with self.subTest(scale=scale):
                manager = _metadata_for(
                    [_table()],
                    _default_metadata(1),
                    scales={(0, "block_1"): scale},
                )
                with self.assertRaises(ValueError):
                    stacking_geometry(_task(manager), 0)

    def test_missing_scene_instance_raises(self):
        manager = _metadata_for([_table()], _default_metadata(1))
        manager.get_instance_name = lambda env_idx, label: None
        with self.assertRaisesRegex(ValueError, "No scene instance"):
            stacking_geometry(_task(manager), 0)

        manager = _metadata_for([_table()], _default_metadata(1))
        manager.get_scene_object = lambda env_idx, inst_name: None
        with self.assertRaisesRegex(ValueError, "missing"):
            stacking_geometry(_task(manager), 0)

    @unittest.skipUnless(ASSET_BLOCK_DIR.is_dir(), "RoboDojo block assets are not mounted")
    def test_actual_asset_metadata_extents(self):
        # Layouts 0-54 use block category indices 5-19. Indices 5/6/7 are the
        # size0/size1/size2 assets; the helper must report their metadata
        # extents, which is what the stack-blocks task uses for stacking checks.
        expected = {5: 0.03, 6: 0.035, 7: 0.04}
        metadata = {}
        for env_idx in range(2):
            for label, category_idx in zip(BLOCK_LABELS, (6, 5, 7)):
                path = ASSET_BLOCK_DIR / f"{category_idx:05d}" / "metadata.json"
                with path.open(encoding="utf-8") as handle:
                    metadata[(env_idx, label)] = json.load(handle)
        manager = _metadata_for([_table(), _table()], metadata)
        for env_index in (0, 1):
            result = stacking_geometry(_task(manager), env_index)
            self.assertEqual(result["block_extents"]["block_0"], [expected[6]] * 3)
            self.assertEqual(result["block_extents"]["block_1"], [expected[5]] * 3)
            self.assertEqual(result["block_extents"]["block_2"], [expected[7]] * 3)


if __name__ == "__main__":
    unittest.main()
