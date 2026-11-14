"""CUDA selection checks for ACT training device resolution."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from starter.policies.act import _training_device


class TrainingDeviceTest(unittest.TestCase):
    def test_cpu_selection_does_not_probe_cuda(self):
        with mock.patch("torch.cuda.is_available") as available, \
                mock.patch("torch.cuda.synchronize") as synchronize:
            device = _training_device({"device": "cpu", "policy_gpu": 0})
        self.assertEqual(device, torch.device("cpu"))
        available.assert_not_called()
        synchronize.assert_not_called()

    def test_unavailable_cuda_device_is_rejected(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "cuda:0.*unavailable"):
                _training_device({"device": "cuda", "policy_gpu": 0})

    def test_usable_cuda_device_is_returned_after_probe(self):
        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.device_count", return_value=1), \
                mock.patch("torch.cuda.set_device") as set_device, \
                mock.patch("torch.arange", return_value=torch.ones(4)), \
                mock.patch("torch.cuda.synchronize") as synchronize:
            device = _training_device({"device": "cuda", "policy_gpu": 0})
        self.assertEqual(device, torch.device("cuda", 0))
        set_device.assert_called_once_with(device)
        synchronize.assert_called_once_with(device)

    def test_failed_kernel_probe_reports_torch_build_and_cause(self):
        kernel_error = RuntimeError("CUDA error: no kernel image is available for execution on the device")
        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.device_count", return_value=1), \
                mock.patch("torch.cuda.set_device"), \
                mock.patch("torch.arange", side_effect=kernel_error), \
                mock.patch("torch.cuda.synchronize"):
            with self.assertRaises(RuntimeError) as raised:
                _training_device({"device": "cuda", "policy_gpu": 0})
        message = str(raised.exception)
        self.assertIn(str(torch.__version__), message)
        self.assertIn(str(torch.version.cuda or "unknown CUDA build"), message)
        self.assertIn("cuda:0", message)
        self.assertIn("README.md", message)
        self.assertIs(raised.exception.__cause__, kernel_error)


if __name__ == "__main__":
    unittest.main()
