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


def prepare(width, height, angle, scene="logo", cap=4, fov=None):
    if scene == "glass":
        source = (ROOT / "examples/ray/glass-transport.xtl").read_text()
        if fov is not None:
            source=source.replace("fov .6",f"fov {fov}")
        for name, value in [("width",width),("height",height),("cap",cap)]:
            source="\n".join(f"{name} := {value}" if line.startswith(name+" := ") else line for line in source.splitlines())+"\n"
        return source
    scene = (ROOT / "examples/ray/logo.xtl").read_text()
    scene = codec.prepare_source(scene, width, height, angle, 0, height, 1, "logo")
    head = scene[:scene.index("## Direct diffuse light")]
    adapter = (ROOT / "examples/ray/region-shade.xtl").read_text().replace('"rt:" u_se< "RayTrace"\n', "")
    return head + adapter + f"""
## A device-computed vector selects structured scalar each lowering.
indices := f_loor f_loat r_ange {width * height * 3}
channels := ((((((((clip u:s_hadeChannels triangles)_ materials)_ logomaterials)_ logolights)_ environment)_ origins)_ directions)_ indices)
linear := {width * height} 3 r_eshape channels
display := 1.0 rt:s_rgb linear
r_avel linear
r_avel display
"""


def split_outputs(text, width, height):
    """Separate returned linear/display arrays without doing rendering math."""
    count = width * height * 3
    tokens = text.split()
    if len(tokens) != count * 2:
        raise ValueError(f"expected two {count}-item RGB arrays, got {len(tokens)} values")
    linear = [float(token) for token in tokens[:count]]
    if any(not math.isfinite(value) for value in linear):
        raise ValueError("linear RGB must be finite; HDR values are preserved")
    display = " ".join(tokens[count:]) + "\n"
    codec.rgb8(display, width, height)
    return " ".join(tokens[:count]) + "\n", display


def compare_outputs(actual, expected, width, height, tolerance=1e-5):
    """Compare raw Float arrays first; report quantization separately."""
    result = {}
    for kind, a_text, e_text in zip(["linear", "display"], actual, expected):
        a, e = [float(x) for x in a_text.split()], [float(x) for x in e_text.split()]
        if len(a) != width*height*3 or len(e) != width*height*3:
            raise ValueError("CPU/GPU comparison arrays must match image dimensions")
        worst = max(abs(x-y) for x,y in zip(a,e))
        if not all(abs(x-y) <= tolerance*(1+abs(y)) for x,y in zip(a,e)):
            raise ValueError(f"CPU/GPU {kind} mismatch: maximum absolute error {worst}")
        result[kind + "_max_absolute_error"] = worst
    a, e = codec.rgb8(actual[1], width, height), codec.rgb8(expected[1], width, height)
    result.update(rgb8_different_channels=sum(x != y for x,y in zip(a,e)),
                  rgb8_max_absolute_difference=max(abs(x-y) for x,y in zip(a,e)))
    return result


def write_images(prefix, text, width, height):
    pixels = codec.rgb8(text, width, height)
    codec.write_tiff(Path(str(prefix) + ".tif"), width, height, pixels)
    Path(str(prefix) + ".png").write_bytes(codec.png(width, height, pixels))


def invoke(command, env):
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{command[1]} failed:\n{result.stderr}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",choices=["logo","glass"],default="logo")
    p.add_argument("--fov",type=float,help="glass validation camera vertical FOV in radians")
    p.add_argument("--cap",type=int,default=4)
    p.add_argument("--float",choices=["f32","f64"],default=None)
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
    if a.fov is not None and not 0 < a.fov < math.pi:
        p.error("FOV must be between zero and pi")
    if not 0 <= a.cap <= 64:
        p.error("interaction cap must be in 0..64")
    precision = a.float or ("f64" if a.scene == "glass" else "f32")
    libraries = Path(subprocess.check_output([str(ROOT / "scripts/xetal-libraries.sh")], text=True).strip())
    env = dict(os.environ, XETAL_PATH=":".join(str(p) for p in sorted(libraries.glob("libs/*/src"))))
    out = a.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    source = prepare(a.width, a.height, a.angle, a.scene, a.cap, a.fov)
    src = out.with_suffix(".xtl")
    src.write_text(source)
    tool = ROOT / "target/release/xetal-gpu"
    checked = invoke([str(tool), "lower", str(src), "--regions"], env)
    out.with_suffix(".check").write_text(checked.stdout + checked.stderr)
    fallback = re.findall(r"evaluated by xetal: ([^:]+):", checked.stderr)
    allowed = {"shaft", "head", "post", "transforms", "triangles", "base", "groups", "materials"}
    if any(name not in allowed for name in fallback):
        raise RuntimeError("non-geometry host fallback: " + repr(fallback))
    explain = invoke([str(tool), "explain", str(src), "--regions", "--float", precision], env)
    if not re.search(r"region_[^:]+: \d+: (channels|linear) :=", explain.stdout):
        raise RuntimeError("no mapped tracing kernel found; refusing a host-rendered image")
    out.with_suffix(".plan").write_text(explain.stdout)
    memory = invoke([str(tool), "memory", str(src), "--regions", "--float", precision], env)
    metrics = {key:int(value) for key,value in (line.split() for line in memory.stdout.splitlines())}
    out.with_suffix(".memory").write_text(memory.stdout)
    print(f"OpenCL region render {a.width}x{a.height}: starting {a.repeat} resident runs", flush=True)
    start = time.monotonic()
    run = invoke([str(tool), "run", str(src), "--regions", "--device", a.device, "--time", "--repeat", str(a.repeat), "--float", precision], env)
    elapsed = time.monotonic() - start
    out.with_suffix(".all.out").write_text(run.stdout)
    actual = split_outputs(run.stdout, a.width, a.height)
    out.with_suffix(".linear.out").write_text(actual[0])
    out.with_suffix(".out").write_text(actual[1])
    out.with_suffix(".timings").write_text(run.stderr)
    write_images(out.with_suffix(""), actual[1], a.width, a.height)
    comparison = None
    if a.compare_cpu:
        cpu = ROOT / "bin/xetal"
        start = time.monotonic()
        reference = invoke([str(cpu), "run", "--seed", "1", "--ascii", str(src)], env)
        cpu_seconds = time.monotonic() - start
        expected = split_outputs(reference.stdout, a.width, a.height)
        out.with_suffix(".cpu.all.out").write_text(reference.stdout)
        out.with_suffix(".cpu.linear.out").write_text(expected[0])
        out.with_suffix(".cpu.out").write_text(expected[1])
        # Path.with_suffix replaces an extension, so preserve the .cpu stem.
        cpu_prefix = out.with_suffix("").with_name(out.stem + "-cpu")
        write_images(cpu_prefix, expected[1], a.width, a.height)
        comparison = dict(cpu_process_seconds=cpu_seconds, **compare_outputs(actual, expected, a.width, a.height, 1e-12 if precision=="f64" else 1e-5),
                          tolerance=("1e-12" if precision=="f64" else "1e-5") + " * (1 + abs(reference))", same_source=True,
                          cpu_tiff=Path(str(cpu_prefix) + ".tif").name,
                          cpu_png=Path(str(cpu_prefix) + ".png").name)
    meta = dict(camera_fov=a.fov if a.fov is not None else (0.6 if a.scene == "glass" else 0.30), renderer_binary_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(), scene=a.scene, float_width=precision, interaction_cap=a.cap if a.scene == "glass" else None, seed=123 if a.scene == "glass" else None, width=a.width, height=a.height, angle_radians=a.angle, backend=a.device, repeats=a.repeat,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(), xetal=(ROOT/"XETAL_COMMIT").read_text().strip(),
                libraries=(ROOT/"XETAL_LIBRARIES_COMMIT").read_text().strip(), process_seconds=elapsed,
                timings=run.stderr.splitlines(), execution_metrics=metrics, cpu_comparison=comparison, samples_per_pixel=1,
                compression="None (TIFF)", host_geometry_fallbacks=fallback, algorithm=("Shared Glass dielectric sampled transport; conservative opaque point-light shadows; no caustics/DOF" if a.scene == "glass" else "Shared RayTrace direct diffuse and hard shadows") + "; scalar RGB channel adapter traces each pixel three times",
                outputs="linear RGB Float array followed by shared X_eTaL sRGB Float array; TIFF/PNG encode display only",
                display_transfer="Shared RayTrace s_rgb, exposure 1; host RGB8 quantization only",
                timing_scope="CLI process includes load/setup/ASCII; resident run phase times recorded separately; CPU evaluator process timing includes load/ASCII")
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    print(run.stderr, end="")
    print(f"Ready: {out}.png / .tif; CPU comparison: {comparison}", flush=True)


if __name__ == "__main__":
    main()
