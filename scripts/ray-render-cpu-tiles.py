#!/usr/bin/env python3
"""Render independent full-camera CPU strips, then encode their exact pixels."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ray_image", ROOT / "scripts/ray-render-cpu.py")
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--strip-rows", type=int, default=16)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--samples-axis", type=int, default=1)
    p.add_argument("--angle", type=float, default=0.0)
    p.add_argument("--libraries-ref", required=True, help="full committed CPU preview snapshot")
    p.add_argument("--output", type=Path, default=ROOT / "work/ray-hires/logo-native")
    a = p.parse_args()
    if min(a.width, a.height, a.strip_rows, a.jobs, a.samples_axis) < 1 or a.samples_axis > 4:
        p.error("positive dimensions/jobs and sample axis 1..4 required")
    subprocess.run([str(ROOT / "scripts/xetal-libraries.sh"), "--snapshot", a.libraries_ref], check=True, stdout=subprocess.DEVNULL)
    out = a.output.resolve()
    strips = out.parent / (out.name + "-strips")
    strips.mkdir(parents=True, exist_ok=True)
    template = ROOT / "examples/ray/logo.xtl"
    scene_hash = hashlib.sha256(template.read_bytes()).hexdigest()
    xetal = (ROOT / "XETAL_COMMIT").read_text().strip()

    def render(y):
        rows = min(a.strip_rows, a.height-y)
        prefix = strips / f"row-{y:04d}"
        meta_file = prefix.with_suffix(".json")
        if meta_file.is_file() and all(prefix.with_suffix(ext).is_file() for ext in [".ppm", ".tif", ".png", ".xtl"]):
            m = json.loads(meta_file.read_text())
            if (m.get("scene_sha256") == scene_hash and m.get("libraries") == a.libraries_ref and m.get("xetal") == xetal
                    and m.get("width") == a.width and m.get("frame_height") == a.height and m.get("height") == rows
                    and m.get("row_start") == y and m.get("samples_axis") == a.samples_axis and m.get("angle_radians") == a.angle
                    and m.get("source_sha256") == hashlib.sha256(prefix.with_suffix(".xtl").read_bytes()).hexdigest()
                    and m.get("source_sha256") == hashlib.sha256(codec.prepare_source(template.read_text(), a.width, a.height, a.angle, y, rows, a.samples_axis, "logo").encode()).hexdigest()):
                return y, prefix, m, True
        args = [sys.executable, str(ROOT / "scripts/ray-render-cpu.py"), "--width", str(a.width), "--height", str(a.height),
                "--row-start", str(y), "--rows", str(rows), "--samples-axis", str(a.samples_axis), "--angle", str(a.angle),
                "--libraries-ref", a.libraries_ref, "--output", str(prefix)]
        run = subprocess.run(args, text=True, capture_output=True)
        if run.returncode:
            raise RuntimeError(run.stdout + run.stderr)
        return y, prefix, json.loads(meta_file.read_text()), False

    start = time.monotonic()
    completed = []
    print(f"CPU native {a.width}x{a.height}, {a.samples_axis**2} samples/pixel, {a.jobs} workers; full-camera strips", flush=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(render, y) for y in range(0, a.height, a.strip_rows)]
        for f in as_completed(futures):
            y, prefix, m, cached = f.result()
            completed.append((y, prefix, m, cached))
            print(f"{len(completed)}/{len(futures)} strips: row {y}, {m['height']} rows, {'cached' if cached else 'rendered'} {m['render_seconds']:.1f}s", flush=True)
    completed.sort(key=lambda item: item[0])
    pixels = bytearray()
    for _, prefix, m, _ in completed:
        header, payload = prefix.with_suffix(".ppm").read_bytes().split(b"\n255\n", 1)
        assert header == f"P6\n{a.width} {m['height']}".encode()
        assert len(payload) == a.width*m["height"]*3
        pixels.extend(payload)
    assert len(pixels) == a.width*a.height*3
    codec.write_tiff(out.with_suffix(".tif"), a.width, a.height, pixels)
    out.with_suffix(".png").write_bytes(codec.png(a.width, a.height, pixels))
    out.with_suffix(".xtl").write_text(codec.prepare_source(template.read_text(), a.width, a.height, a.angle, 0, a.height, a.samples_axis, "logo"))
    manifest = dict(backend="X_eTaL CPU evaluator", width=a.width, height=a.height, samples_axis=a.samples_axis,
                    samples_per_pixel=a.samples_axis**2, angle_radians=a.angle, scene_sha256=scene_hash,
                    full_frame_source_sha256=hashlib.sha256(out.with_suffix(".xtl").read_bytes()).hexdigest(),
                    execution="Parallel strips; full-frame source is the equivalent single-process program; exact executed strip sources are recorded",
                    xetal=xetal, libraries=a.libraries_ref, library_snapshot=True, compression="None (TIFF)",
                    render_wall_seconds=time.monotonic()-start, workers=a.jobs, strip_rows=a.strip_rows,
                    cached_strips=sum(int(item[3]) for item in completed),
                    total_strip_seconds=sum(item[2]["render_seconds"] for item in completed),
                    timing_scope="Sequence wall time includes evaluator setup/text output and encoding; not a paired benchmark or measured serial baseline",
                    paired_gpu_validation="not established for this CPU snapshot", sample_pattern="regular centered subpixel grid; linear RGB averaged in X_eTaL",
                    strips=[dict(row=y, rows=m["height"], source_sha256=m["source_sha256"], render_seconds=m["render_seconds"], source=str(prefix.with_suffix(".xtl"))) for y,prefix,m,_ in completed])
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Native still ready: {out}.png / .tif ({manifest['render_wall_seconds']:.1f}s wall)", flush=True)


if __name__ == "__main__":
    main()
