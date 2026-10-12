#!/usr/bin/env python3
"""Check that GPU measurements cannot silently skip a required device
and that a report can be saved without overwriting another machine's.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("ml_bench", Path(__file__).with_name("ml-bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkReporting(unittest.TestCase):
    def test_required_device_cannot_be_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(bench, "WORK", Path(directory)), patch.object(bench, "devices", return_value=[]), \
                    patch.object(bench.subprocess, "run"), patch.object(bench, "write_results") as write, \
                    patch("sys.argv", ["ml-bench.py", "--no-facade", "--require-device"]):
                with self.assertRaisesRegex(SystemExit, "refuses to skip GPU measurements"):
                    bench.main()
                write.assert_not_called()

    def test_report_preserves_other_machine(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "mac.md"
            original.write_text("Existing Mac measurements\n")
            destination = Path(directory) / "arch.md"
            row = {"name": "dense", "what": "a product", "result": "1.0", "xetal_ms": 2, "cpu_ms": 1,
                   "opencl:0": {"diff": 0, "kernels": 0.01, "launch": 0.1, "total": 3, "process": 4},
                   "facade": [1, 2, 3, 4]}
            model = dict(row, name="world-model", result="1897 and 348")
            with patch.object(bench, "RESULTS", original), \
                    patch.object(bench, "device_names", return_value={"opencl:0": "RTX 5060 Ti"}):
                bench.write_results([row], [model], "pinned ML", ["opencl:0"], "ml-pin", "xetal-pin", True, destination)
            self.assertEqual(original.read_text(), "Existing Mac measurements\n")
            report = destination.read_text()
            self.assertIn("RTX 5060 Ti", report)
            self.assertIn("| dense | 0 | 0.010 | 0.10 | 3 | 4 | 3 | 6 |", report)
            self.assertIn("## The models", report)
            self.assertIn("1897 and 348", report)


if __name__ == "__main__":
    unittest.main()
