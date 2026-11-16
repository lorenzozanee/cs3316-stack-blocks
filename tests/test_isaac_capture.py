"""Episode resets retain one capture graph and release it when cameras change."""
from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace

from starter.envs.isaac_batch import _install_capture_lifecycle


class FakeCapture:
    """Model the pinned manager's append-on-initialize resource behavior."""

    def __init__(self):
        self.camera_manager = SimpleNamespace(
            cameras=[[object(), object()]], camera_names=[["head", "wrist"]],
        )
        self.cameras = self.camera_manager.cameras
        self.camera_names = self.camera_manager.camera_names
        self.num_cams = 2
        self.sim = object()
        self.tiled_cameras = []
        self.tiled_render_products = []
        self.annotator = []
        self.annotator_type = []
        self.annotator_device = []
        self.camera_prim_paths = []
        self._output_buffers = {}
        self.initializations = 0
        self.events = []

    def init_cameras(self):
        self.initializations += 1
        self.events.append("initialize")
        for index in range(self.num_cams):
            path = f"product_{self.initializations}_{index}"
            self.tiled_cameras.append(SimpleNamespace(
                _render_product_path=path,
                _annotators={"rgba": SimpleNamespace(
                    detach=lambda paths: self.events.append(("detach", paths[0])),
                )},
            ))
            self.tiled_render_products.append(SimpleNamespace(
                destroy=lambda p=path: self.events.append(("destroy", p)),
            ))
            self.annotator.append([])
            self.annotator_device.append([])
            self.annotator_type.append(["rgb"])
            self._output_buffers[index] = object()
        self.camera_prim_paths.append(["head", "wrist"])


class CaptureLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.capture = FakeCapture()
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        _install_capture_lifecycle(self.capture)

    def test_repeated_episode_resets_reuse_products_annotators_and_buffers(self):
        capture = self.capture
        capture.reset()
        products = tuple(capture.tiled_render_products)
        buffers = dict(capture._output_buffers)
        for _ in range(100):
            capture.reset()
        self.assertEqual(capture.initializations, 1)
        self.assertEqual(tuple(capture.tiled_render_products), products)
        self.assertEqual(capture._output_buffers, buffers)
        self.assertEqual(capture.annotator_type, [["rgb"], ["rgb"]])
        self.assertEqual(len(capture.tiled_cameras), 2)
        self.assertEqual(capture.events, ["initialize"])

    def test_rebuilt_camera_handles_release_old_resources_before_initializing(self):
        capture = self.capture
        capture.reset()
        # The paths/names can remain identical while handles are replaced.
        capture.camera_manager.cameras = [[object(), object()]]
        capture.reset()
        self.assertEqual(capture.events, [
            "initialize", ("detach", "product_1_0"), ("detach", "product_1_1"),
            ("destroy", "product_1_0"), ("destroy", "product_1_1"), "initialize",
        ])
        self.assertEqual(capture.initializations, 2)
        self.assertEqual(len(capture.tiled_render_products), 2)
        self.assertEqual(capture.annotator_type, [["rgb"], ["rgb"]])
        self.assertIs(capture.cameras, capture.camera_manager.cameras)
        capture.reset()
        self.assertEqual(capture.initializations, 2)

    def test_destroy_detaches_and_releases_once_without_clearing_camera_owner(self):
        capture = self.capture
        capture.reset()
        owned_cameras = capture.camera_manager.cameras
        owned_names = capture.camera_manager.camera_names
        capture.destroy()
        events = list(capture.events)
        capture.destroy()
        self.assertEqual(capture.events, events)
        self.assertEqual(len(owned_cameras[0]), 2)
        self.assertEqual(owned_names[0], ["head", "wrist"])
        for name in ("tiled_cameras", "tiled_render_products", "annotator", "annotator_type",
                     "annotator_device", "camera_prim_paths", "_output_buffers"):
            self.assertFalse(getattr(capture, name), name)
        self.assertIsNone(capture.sim)


if __name__ == "__main__":
    unittest.main()
