#!/usr/bin/env python3
"""Independently verify frozen paired masters against both returned Float arrays."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import zipfile
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ray_pair_media", ROOT / "scripts/test-ray-media.py")
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)
def verify(p, minimum=(1, 1)):
    m = json.loads((p / "manifest.json").read_text())
    w, h = m["width"], m["height"]
    assert w >= minimum[0] and h >= minimum[1] and m["cpu_comparison"]["same_source"]
    assert hashlib.sha256((p / "scene.xtl").read_bytes()).hexdigest() == m["source_sha256"]
    for name, sha in m["artifact_sha256"].items():
        assert hashlib.sha256((p/name).read_bytes()).hexdigest() == sha
    for key in ["xetal", "libraries", "renderer_git_base"]:
        assert len(m[key]) == 40 and all(c in "0123456789abcdef" for c in m[key])
    arrays = {}
    pixels = {}
    for backend in ["cpu", "gpu"]:
        for kind in ["linear", "display"]:
            values = [float(x) for x in (p/f"{kind}-{backend}.txt").read_text().split()]
            assert len(values) == w*h*3 and all(math.isfinite(x) for x in values)
            arrays[kind, backend] = values
        display = arrays["display", backend]
        assert all(0 <= x <= 1 for x in display)
        expected = bytes(int(x*255 + .5) for x in display)
        pixels[backend] = expected
        assert media.tiff_pixels((p/f"{backend}.tif").read_bytes(), w, h) == expected
        assert media.png_pixels((p/f"{backend}.png").read_bytes(), w, h) == expected
    for kind in ["linear", "display"]:
        cpu, gpu = arrays[kind, "cpu"], arrays[kind, "gpu"]
        assert all(abs(a-b) <= (1e-12 if m.get("float_width") == "f64" else 1e-5)*(1+abs(a)) for a,b in zip(cpu,gpu))
        observed = max(abs(a-b) for a,b in zip(cpu,gpu))
        assert observed == m["cpu_comparison"][kind+"_max_absolute_error"]
    assert sum(a != b for a,b in zip(pixels["cpu"],pixels["gpu"])) == m["cpu_comparison"]["rgb8_different_channels"]
    assert max(abs(a-b) for a,b in zip(pixels["cpu"],pixels["gpu"])) == m["cpu_comparison"]["rgb8_max_absolute_difference"]
    with zipfile.ZipFile(p / "masters.zip") as z:
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
        for name in [*m["artifact_sha256"], "manifest.json"]:
            assert z.read(name) == (p/name).read_bytes()
    print(f"paired images: {w}x{h}, CPU/GPU linear and display Floats agree; RGB8 PNG/TIFF tags and exact pixels pass")


verify(ROOT / "site-assets/ray/image-pair", (16, 8))
for cap in [2, 4]:
    verify(ROOT / f"site-assets/ray/glass-transport/cap{cap}")
a, b = [(ROOT / f"site-assets/ray/glass-transport/cap{cap}/linear-gpu.txt").read_text() for cap in [2, 4]]
assert all(float(x) == 0 for x in a.split())
assert any(float(x) > 0 for x in b.split())
