#!/usr/bin/env python3
"""Verify glass master pixels, raw radiance and small CPU/GPU evidence."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("glass_media", ROOT / "scripts/test-ray-media.py")
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)


def floats(data, count):
    values = [float(x) for x in data.split()]
    assert len(values) == count and all(math.isfinite(x) for x in values)
    return values


def main():
    p = ROOT / "site-assets/ray/glass-rtx"
    m = json.loads((p / "manifest.json").read_text())
    w, h = m["width"], m["height"]
    assert w >= 1024 and h >= 512 and m["scene"] == "glass"
    assert m["backend"].startswith("opencl:") and m["float_width"] == "f64"
    assert m["samples_per_pixel"] == 1 and m["interaction_cap"] >= 4
    with zipfile.ZipFile(p / "masters.zip") as z:
        native = {name: z.read(name) for name in ["linear.txt", "display.txt"]}
    for key in ["xetal", "libraries", "renderer_commit"]:
        assert len(m[key]) == 40 and all(c in "0123456789abcdef" for c in m[key])
    for name, digest in m["artifact_sha256"].items():
        data = native[name] if name in native else (p/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
    assert hashlib.sha256((p / "scene.xtl").read_bytes()).hexdigest() == m["source_sha256"]
    linear = floats(native["linear.txt"], w*h*3)
    assert min(linear) >= 0 and max(linear) > .1
    display = floats(native["display.txt"], w*h*3)
    assert all(0 <= x <= 1 for x in display)
    expected = bytes(int(x*255+.5) for x in display)
    assert len(set(expected)) > 32
    assert media.tiff_pixels((p / "still.tif").read_bytes(), w, h) == expected
    assert media.png_pixels((p / "still.png").read_bytes(), w, h) == expected
    v = json.loads((p / "validation.json").read_text())
    assert v["cpu_comparison"]["same_source"]
    assert hashlib.sha256((p / "validation.xtl").read_bytes()).hexdigest() == v["source_sha256"]
    for key in ["xetal", "libraries", "renderer_binary_sha256", "float_width", "interaction_cap", "seed"]:
        assert m[key] == v[key]
    for kind in ["linear", "display"]:
        a, b = [floats((p / f"validation-{kind}-{backend}.txt").read_bytes(), v["width"]*v["height"]*3)
                for backend in ["cpu", "gpu"]]
        assert all(abs(x-y) <= 1e-12*(1+abs(x)) for x, y in zip(a, b))
        assert max(abs(x-y) for x, y in zip(a, b)) == v["cpu_comparison"][kind+"_max_absolute_error"]
    with zipfile.ZipFile(p / "masters.zip") as z:
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
        for name in [*m["artifact_sha256"], "manifest.json"]:
            expected_member = native[name] if name in native else (p/name).read_bytes()
            assert z.read(name) == expected_member, name
    print(f"glass still: native {w}x{h}, exact TIFF/PNG/Float pixels, stored masters and matched small CPU/GPU evidence pass")


if __name__ == "__main__":
    main()
