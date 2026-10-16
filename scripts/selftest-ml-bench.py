#!/usr/bin/env python3
"""Check that GPU measurements cannot silently skip a required device,
that a report can be saved without overwriting another machine's, and
that the macro route's agreement rule reads a program's result lines.
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
            with patch.object(bench, "WORK", Path(directory)), patch.object(bench, "CACHE", Path(directory) / "none"), \
                    patch.object(bench, "devices", return_value=[]), \
                    patch.object(bench.subprocess, "run"), patch.object(bench, "write_results") as write, \
                    patch("sys.argv", ["ml-bench.py", "--no-macro", "--require-device"]):
                with self.assertRaisesRegex(SystemExit, "refuses to skip GPU measurements"):
                    bench.main()
                write.assert_not_called()

    def test_report_preserves_other_machine(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "mac.md"
            original.write_text("Existing Mac measurements\n")
            destination = Path(directory) / "arch.md"
            row = {"name": "dense", "what": "a product", "result": "1.0", "xetal_ms": 2, "cpu_ms": 1, "lower_s": 0.3, "host": 0,
                   "opencl:0": {"diff": 0, "kernels": 0.01, "launch": 0.1, "total": 3, "process": 4}, "macro": [10, 5]}
            model = dict(row, name="world-model", result="1897 and 348", host=1)
            with patch.object(bench, "RESULTS", original), \
                    patch.object(bench, "device_names", return_value={"opencl:0": "RTX 5060 Ti"}):
                bench.write_results([row, model], ["opencl:0"], "ml-pin", "xetal-pin", True, destination)
            self.assertEqual(original.read_text(), "Existing Mac measurements\n")
            report = destination.read_text()
            self.assertIn("RTX 5060 Ti", report)
            self.assertIn("| dense | 0 | 0.010 | 0.10 | 3 | 4 | 10 | 5 |", report)
            self.assertIn("| world-model | a product | 1897 and 348 | 2 | 1 | 0.3 | 1 |", report)
            self.assertIn("1897 and 348", report)

    def test_macro_agreement_reads_result_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            want = Path(directory) / "want.out"
            want.write_text("1\n1897\n348\n....\n")
            self.assertTrue(bench.macro_agrees(want, (2, 3), ["1897", "348"])[0])
            self.assertFalse(bench.macro_agrees(want, (2, 3), ["1897", "347"])[0])
            want.write_text("40394.999999997555\n")
            self.assertTrue(bench.macro_agrees(want, (1,), ["40395.0"])[0])
            self.assertFalse(bench.macro_agrees(want, (1,), ["40396.0"])[0])
            self.assertEqual(bench.result_summary("1\n1897\n348\n....\n", (2, 3)), "1897 and 348")
            self.assertEqual(bench.result_summary("40395.0\n", (1,)), "40395.0")


if __name__ == "__main__":
    unittest.main()
