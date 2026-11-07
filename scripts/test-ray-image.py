#!/usr/bin/env python3
"""Independent baseline TIFF tags/payload checks and invalid pixel checks."""
import importlib.util
from pathlib import Path
import struct
import shutil
import tempfile
import unittest
import zlib

spec = importlib.util.spec_from_file_location("ray_image", Path(__file__).with_name("ray-render-cpu.py"))
image = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image)

spec = importlib.util.spec_from_file_location("ray_regions", Path(__file__).with_name("ray-render-regions.py"))
regions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regions)


class ImageTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ppm2tiff"), "libtiff ppm2tiff not installed")
    def test_known_two_pixel_tiff(self):
        pixels = image.rgb8("1 0 0 0 .5 1", 2, 1)
        self.assertEqual(pixels, bytes([255, 0, 0, 0, 128, 255]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "known.tif"
            image.write_tiff(path, 2, 1, pixels)
            data = path.read_bytes()
        self.assertEqual(data[:4], b"II*\0")
        directory = struct.unpack_from("<I", data, 4)[0]
        count = struct.unpack_from("<H", data, directory)[0]
        tags = {}
        for i in range(count):
            tag, kind, length, value = struct.unpack_from("<HHII", data, directory + 2 + i * 12)
            tags[tag] = (kind, length, value)
        self.assertEqual(tags[256][1:], (1, 2))
        self.assertEqual(tags[257][1:], (1, 1))
        self.assertEqual(tags[259], (3, 1, 1))  # TIFF Compression=None
        self.assertEqual(tags[262], (3, 1, 2))  # RGB photometric interpretation
        self.assertEqual(tags[277], (3, 1, 3))
        self.assertEqual(struct.unpack_from("<HHH", data, tags[258][2]), (8, 8, 8))
        self.assertEqual(data[tags[273][2]:tags[273][2] + tags[279][2]], pixels)

    def test_png_pixel_rows_and_crc(self):
        pixels = bytes([255, 0, 0, 0, 128, 255])
        data = image.png(1, 2, pixels)
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        pos = 8
        compressed = b""
        while pos < len(data):
            size = struct.unpack_from(">I", data, pos)[0]
            kind = data[pos + 4:pos + 8]
            payload = data[pos + 8:pos + 8 + size]
            self.assertEqual(struct.unpack_from(">I", data, pos + 8 + size)[0], zlib.crc32(kind + payload))
            if kind == b"IDAT":
                compressed += payload
            pos += 12 + size
        self.assertEqual(zlib.decompress(compressed), b"\0\xff\0\0\0\0\x80\xff")

    def test_linear_hdr_is_retained_and_display_is_validated_separately(self):
        linear, display = regions.split_outputs("4.0 .5 0.0 1.0 .735 0.0", 1, 1)
        self.assertEqual([float(x) for x in linear.split()], [4.0, .5, 0.0])
        self.assertEqual(image.rgb8(display, 1, 1), bytes([255, 187, 0]))
        for text in ["1 0 0", "nan 0 0 1 0 0", "1 0 0 2 0 0"]:
            with self.assertRaises(ValueError):
                regions.split_outputs(text, 1, 1)

    def test_raw_comparison_precedes_rgb8_rounding(self):
        actual = (".5 0 0", ".500001 0 0")
        expected = (".5 0 0", ".499999 0 0")
        compared = regions.compare_outputs(actual, expected, 1, 1)
        self.assertEqual(compared["rgb8_different_channels"], 1)
        self.assertEqual(compared["rgb8_max_absolute_difference"], 1)
        with self.assertRaises(ValueError):
            regions.compare_outputs((".6 0 0", ".5 0 0"), expected, 1, 1)
        with self.assertRaises(ValueError):
            regions.compare_outputs((".5 0", ".5 0 0"), expected, 1, 1)

    def test_bad_rgb_is_refused(self):
        for text in ["1 0", "nan 0 0", "inf 0 0", "-.01 0 0", "1.01 0 0"]:
            with self.assertRaises(ValueError):
                image.rgb8(text, 1, 1)


if __name__ == "__main__":
    unittest.main()
