#!/usr/bin/env python3
"""Render the shared logo through structured OpenCL regions; preserve TIFF and timings."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ray_codec", ROOT / "scripts/ray-render-cpu.py")
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


def prepare(width, height, angle):
    scene = (ROOT / "examples/ray/logo.xtl").read_text()
    scene = codec.prepare_source(scene, width, height, angle, 0, height, 1, "logo")
    head = scene[:scene.index("## Direct diffuse light")]
    adapter = (ROOT / "examples/ray/region-shade.xtl").read_text().replace('"rt:" u_se< "RayTrace"\n', "")
    return head + adapter + f"""
## A device-computed vector selects structured scalar each lowering.
indices := f_loor f_loat r_ange {width * height * 3}
channels := ((((((((clip u:s_hadeChannels triangles)_ materials)_ logomaterials)_ logolights)_ environment)_ origins)_ directions)_ indices)
linear := {width * height} 3 r_eshape channels
1.0 rt:s_rgb linear
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--angle", type=float, default=0.0)
    p.add_argument("--device", default="opencl:0")
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--compare-cpu", action="store_true", help="also run the exact generated source in the pinned evaluator")
    p.add_argument("--output", type=Path, default=ROOT / "work/ray-regions/logo")
    a = p.parse_args()
    if min(a.width, a.height, a.repeat) < 1 or not math.isfinite(a.angle):
        p.error("positive dimensions/repeats and finite angle required")
    if not a.device.startswith("opencl:"):
        p.error("this prototype requires opencl:N")
    libraries = Path(subprocess.check_output([str(ROOT / "scripts/xetal-libraries.sh")], text=True).strip())
    env = dict(os.environ, XETAL_PATH=":".join(str(p) for p in sorted(libraries.glob("libs/*/src"))))
    out = a.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    source = prepare(a.width, a.height, a.angle)
    src = out.with_suffix(".xtl")
    src.write_text(source)
    tool = ROOT / "target/release/xetal-gpu"
    checked = subprocess.run([str(tool), "lower", str(src), "--regions"], env=env, text=True, capture_output=True, check=True)
    out.with_suffix(".check").write_text(checked.stdout + checked.stderr)
    fallback = re.findall(r"evaluated by xetal: ([^:]+):", checked.stderr)
    allowed = {"shaft", "head", "post", "transforms", "triangles", "base", "groups", "materials"}
    if any(name not in allowed for name in fallback):
        raise RuntimeError("non-geometry host fallback: " + repr(fallback))
    explain = subprocess.run([str(tool), "explain", str(src), "--regions"], env=env, text=True, capture_output=True, check=True)
    out.with_suffix(".plan").write_text(explain.stdout)
    memory = subprocess.run([str(tool), "memory", str(src), "--regions"], env=env, text=True, capture_output=True, check=True)
    metrics = {key:int(value) for key,value in (line.split() for line in memory.stdout.splitlines())}
    out.with_suffix(".memory").write_text(memory.stdout)
    print(f"RTX region render {a.width}x{a.height}: starting {a.repeat} resident runs", flush=True)
    start = time.monotonic()
    run = subprocess.run([str(tool), "run", str(src), "--regions", "--device", a.device, "--time", "--repeat", str(a.repeat)], env=env, text=True, capture_output=True, check=True)
    elapsed = time.monotonic() - start
    out.with_suffix(".out").write_text(run.stdout)
    out.with_suffix(".timings").write_text(run.stderr)
    pixels = codec.rgb8(run.stdout, a.width, a.height)
    codec.write_tiff(out.with_suffix(".tif"), a.width, a.height, pixels)
    out.with_suffix(".png").write_bytes(codec.png(a.width, a.height, pixels))
    comparison = None
    if a.compare_cpu:
        cpu = ROOT / "bin/xetal"
        start = time.monotonic()
        reference = subprocess.run([str(cpu), "run", "--seed", "1", "--ascii", str(src)], env=env, text=True, capture_output=True, check=True)
        cpu_seconds = time.monotonic() - start
        out.with_suffix(".cpu.out").write_text(reference.stdout)
        actual = [float(x) for x in run.stdout.split()]
        expected = [float(x) for x in reference.stdout.split()]
        assert len(actual) == len(expected) == a.width*a.height*3
        worst = max(abs(x-y) for x,y in zip(actual, expected))
        assert all(abs(x-y) <= 1e-5*(1+abs(y)) for x,y in zip(actual, expected)), f"CPU/GPU mismatch: {worst}"
        comparison = dict(cpu_process_seconds=cpu_seconds, max_absolute_rgb_error=worst, tolerance="1e-5 * (1 + abs(reference))", same_source=True)
    meta = dict(width=a.width, height=a.height, angle_radians=a.angle, backend=a.device, repeats=a.repeat,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(), xetal=(ROOT/"XETAL_COMMIT").read_text().strip(),
                libraries=(ROOT/"XETAL_LIBRARIES_COMMIT").read_text().strip(), process_seconds=elapsed,
                timings=run.stderr.splitlines(), execution_metrics=metrics, cpu_comparison=comparison, samples_per_pixel=1,
                compression="None (TIFF)", host_geometry_fallbacks=fallback, algorithm="Shared RayTrace direct diffuse and hard shadows; scalar RGB channel adapter traces each pixel three times",
                timing_scope="CLI process includes load/setup/ASCII; resident run phase times recorded separately; CPU evaluator process timing includes load/ASCII")
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    print(run.stderr, end="")
    print(f"Ready: {out}.png / .tif; CPU comparison: {comparison}", flush=True)


if __name__ == "__main__":
    main()
