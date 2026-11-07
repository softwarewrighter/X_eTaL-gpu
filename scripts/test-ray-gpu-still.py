#!/usr/bin/env python3
"""Validate the frozen GPU image, exact source and small matched-source evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ray_media_check", ROOT / "scripts/test-ray-media.py")
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)
p = ROOT / "site-assets/ray/logo-gpu"
m = json.loads((p / "manifest.json").read_text())
w, h = m["width"], m["height"]
assert w >= 1024 and h >= 512 and m["samples_per_pixel"] == 1
assert codec.png_pixels((p / "still.png").read_bytes(), w, h) == codec.tiff_pixels((p / "still.tif").read_bytes(), w, h)
assert hashlib.sha256((p / "scene.xtl").read_bytes()).hexdigest() == m["source_sha256"]
for key in ["libraries", "xetal", "backend_git_base"]:
    assert len(m[key]) == 40 and all(c in "0123456789abcdef" for c in m[key])
assert hashlib.sha256((p / "validation.xtl").read_bytes()).hexdigest() == m["validation_source_sha256"]
a = [float(x) for x in (p / "validation-gpu.txt").read_text().split()]
b = [float(x) for x in (p / "validation-cpu.txt").read_text().split()]
assert len(a) == len(b) == 8*4*3
assert all(abs(x-y) <= 1e-5*(1+abs(y)) for x,y in zip(a,b))
assert abs(max(abs(x-y) for x,y in zip(a,b)) - m["small_same_source_validation"]["max_absolute_rgb_error"]) < 1e-15
with zipfile.ZipFile(p / "masters.zip") as z:
    assert all(info.compress_type == zipfile.ZIP_STORED for info in z.infolist())
    for name in ["still.tif", "scene.xtl", "manifest.json", "validation.xtl", "validation-gpu.txt", "validation-cpu.txt"]:
        assert z.read(name) == (p / name).read_bytes()
print("GPU still: native PNG/TIFF pixels, source hash, uncompressed masters and matched small CPU/GPU outputs pass")
