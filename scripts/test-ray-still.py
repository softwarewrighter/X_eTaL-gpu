#!/usr/bin/env python3
"""Verify the native-resolution still, exact master pixels and archived provenance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("still_media", ROOT / "scripts/test-ray-media.py")
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)


def main():
    assets = ROOT / "site-assets/ray/logo-still"
    m = json.loads((assets / "manifest.json").read_text())
    assert m["backend"] == "X_eTaL CPU evaluator"
    assert m["width"] >= 1024 and m["height"] > 0
    assert m["samples_per_pixel"] == m["samples_axis"] ** 2
    assert all(re.fullmatch("[0-9a-f]{40}", m[key]) for key in ["xetal", "libraries"])
    assert m["source_sha256"] == hashlib.sha256((assets / "scene.xtl").read_bytes()).hexdigest()
    assert media.png_pixels((assets / "still.png").read_bytes(), m["width"], m["height"]) == media.tiff_pixels((assets / "still.tif").read_bytes(), m["width"], m["height"])
    row = 0
    with zipfile.ZipFile(assets / "masters.zip") as archive:
        assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
        for file in ["still.tif", "scene.xtl", "manifest.json"]:
            assert archive.read(file) == (assets / file).read_bytes()
        for strip in m["strips"]:
            assert strip["row"] == row and 0 < strip["rows"] <= m["strip_rows"]
            row += strip["rows"]
            assert hashlib.sha256(archive.read(strip["source"])).hexdigest() == strip["source_sha256"]
            data = json.loads(archive.read(str(Path(strip["source"]).with_suffix(".json"))))
            assert all(data[k] == m[k] for k in ["xetal", "libraries", "scene_sha256", "samples_axis", "samples_per_pixel"])
            assert data["source_sha256"] == strip["source_sha256"] and data["row_start"] == strip["row"]
    assert row == m["height"]
    print(f"native CPU still: {m['width']}x{m['height']}, exact PNG/TIFF pixels, complete strips and committed source pins")


if __name__ == "__main__":
    main()
