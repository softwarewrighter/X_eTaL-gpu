#!/usr/bin/env python3
"""Verify curated frames, lossless master pixels, provenance and video decoding."""
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
import struct
import subprocess
import zipfile
import zlib

ROOT = Path(__file__).resolve().parent.parent


def main():
    assets = ROOT / "site-assets/ray/logo"
    meta = json.loads((assets / "manifest.json").read_text())
    assert meta["backend"] == "X_eTaL CPU evaluator"
    assert meta["scene_sha256"] == hashlib.sha256((assets / "scene.xtl").read_bytes()).hexdigest()
    assert all(re.fullmatch("[0-9a-f]{40}", meta[key]) for key in ["xetal", "libraries"])
    width, height, count = meta["width"], meta["height"], meta["frame_count"]
    assert len(meta["frames"]) == count and count >= 4
    hashes = set()
    with zipfile.ZipFile(assets / "masters.zip") as archive:
        assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
        for index, frame in enumerate(meta["frames"]):
            assert frame["index"] == index
            assert abs(frame["angle_radians"] - 2 * math.pi * index / count) < 1e-12
            name = f"frame-{index:03d}"
            assert hashlib.sha256(archive.read(name + ".xtl")).hexdigest() == frame["source_sha256"]
            recorded = json.loads(archive.read(name + ".json"))
            assert all(recorded[k] == meta[k] for k in ["xetal", "libraries", "scene_sha256"])
            png = (assets / "frames" / (name + ".png")).read_bytes()
            assert png[:8] == b"\x89PNG\r\n\x1a\n"
            assert struct.unpack_from(">II", png, 16) == (width, height)
            pos, compressed = 8, b""
            while pos < len(png):
                size = struct.unpack_from(">I", png, pos)[0]
                kind, payload = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + size]
                assert struct.unpack_from(">I", png, pos + 8 + size)[0] == zlib.crc32(kind + payload)
                if kind == b"IDAT":
                    compressed += payload
                pos += size + 12
            rows = zlib.decompress(compressed)
            stride = width * 3 + 1
            assert len(rows) == stride * height
            assert all(rows[y * stride] == 0 for y in range(height))
            pixels = b"".join(rows[y * stride + 1:(y + 1) * stride] for y in range(height))
            hashes.add(hashlib.sha256(pixels).hexdigest())
            master = archive.read(name + ".tif")
            assert master[:4] == b"II*\0"
            directory = struct.unpack_from("<I", master, 4)[0]
            tags = {}
            for i in range(struct.unpack_from("<H", master, directory)[0]):
                tag, kind, length, value = struct.unpack_from("<HHII", master, directory + 2 + i * 12)
                tags[tag] = (kind, length, value)
            assert tags[256][2] == width and tags[257][2] == height
            assert tags[259][2] == 1 and tags[262][2] == 2 and tags[277][2] == 3
            assert tags[273][1] == tags[279][1] == 1
            assert master[tags[273][2]:tags[273][2] + tags[279][2]] == pixels
    assert len(hashes) >= count // 2, "frames do not show a changing scene"
    if shutil.which("ffprobe"):
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "quiet", "-count_frames", "-select_streams", "v:0",
                         "-show_entries", "stream=codec_name,width,height,nb_read_frames", "-of", "json", str(assets / "logo.webm")], text=True))
        video = probe["streams"][0]
        assert video["codec_name"] == "vp9"
        assert (video["width"], video["height"], int(video["nb_read_frames"])) == (width, height, count)
    else:
        print("skipped: WebM decode check (ffprobe not installed)")
    print(f"ray media: {count} rendered frames, exact PNG/TIFF pixels, pins/source pass; WebM checked when ffprobe is installed")


if __name__ == "__main__":
    main()
