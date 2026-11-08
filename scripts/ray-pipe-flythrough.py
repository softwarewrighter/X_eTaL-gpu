#!/usr/bin/env python3
"""Render periodic pipe frames in resident GPU batches; encode actual traced pixels."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pipe_codec", ROOT / "scripts/ray-render-regions.py")
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


def captured(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--compare-cpu", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "work/ray-pipe-rtx")
    a = p.parse_args()
    if min(a.width, a.height, a.frames, a.batch, a.fps) < 1 or a.frames % a.batch:
        p.error("positive dimensions required; frames must be divisible by batch")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    libraries = Path(subprocess.check_output([str(ROOT/"scripts/xetal-libraries.sh")], text=True).strip())
    env = dict(os.environ, XETAL_PATH=":".join(str(p) for p in libraries.glob("libs/*/src")))
    template = (ROOT / "examples/ray/pipe-flythrough.xtl").read_text()
    geom = template.split("width :=", 1)[0] + "\nr_avel triangles\nr_avel materialids\n"
    (out / "geometry.xtl").write_text(geom)
    run = captured([str(ROOT/"bin/xetal"), "run", "--ascii", str(out/"geometry.xtl")], env=env, cwd=out)
    mesh, ids = run.stdout.strip().splitlines()
    triangles = len(mesh.split()) // 9
    assert len(mesh.split()) == triangles*9 and len(ids.split()) == triangles
    xyz = list(map(float, mesh.split()))
    mids = list(map(int, ids.split()))
    for k in range(triangles):
        points = [xyz[k*9+i*3:k*9+i*3+3] for i in range(3)]
        u = [points[1][j]-points[0][j] for j in range(3)]
        v = [points[2][j]-points[0][j] for j in range(3)]
        cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        assert sum(x*x for x in cross)>1e-16, f"degenerate triangle {k}"
    sphere_points = [xyz[k*9+j*3:k*9+j*3+3] for k in range(triangles) if mids[k]==4 for j in range(3)]
    assert abs(min(p[0] for p in sphere_points) - (-3.6-1.15)) < 1e-9
    assert abs(max(p[0] for p in sphere_points) - (-3.6+1.15)) < 1e-9
    (out / "triangles.txt").write_text(mesh+"\n")
    (out / "materialids.txt").write_text(ids+"\n")
    materials = next(line for line in template.splitlines() if line.startswith("materials :="))
    tail = "width :=" + template.split("width :=", 1)[1]
    source = f'''## Immutable mesh was evaluated once by the pinned X_eTaL evaluator; see geometry.xtl.
"rt:" u_se< "RayTrace"
triangles := {triangles} 3 3 r_eshape n_umbers []N_GET "triangles.txt"
materialids := f_loor n_umbers []N_GET "materialids.txt"
{materials}
''' + tail
    for name, value in [("width", a.width), ("height", a.height), ("framecount", a.frames), ("batchframes", a.batch)]:
        source = "\n".join(f"{name} := {value}" if line.startswith(name+" :=") else line for line in source.splitlines())+"\n"
    (out / "scene.xtl").write_text(source)
    body = "\n".join(line for line in source.splitlines() if not line.startswith(('"rt:" u_se', 'frameStart :=', 'r_avel linear', 'r_avel 1.0 rt:s_rgb linear')))
    function = "{ frameStart ->\n" + body.replace("u:b_atch", "batch_") + "\n(r_avel linear) c_at r_avel 1.0 rt:s_rgb linear\n}\n"
    (out / "function.xtl").write_text(function)
    (out / "frame-start.txt").write_text("0\n")
    tool = ROOT / "target/release/xetal-gpu"
    lowered = captured([str(tool), "lower-function", str(out/"function.xtl"), "--param", "frameStart:i64", "--uses", '\"rt:\" u_se< \"RayTrace\"', "--regions"],env=env,cwd=out)
    (out/"scene.xir").write_text(lowered.stdout)
    cmd = [str(tool), "run", str(out/"scene.xir"), "--bind", "frameStart=0", "--regions", "--float", "f32", "--device", "opencl:0", "--time", "--repeat", str(a.frames//a.batch+1), "--advance", f"frameStart={a.batch}", "--save-repeats", str(out/"batches")]
    print(f"{a.frames} frames, {a.width}x{a.height}, {a.batch} frames per resident GPU batch; {triangles} triangles", flush=True)
    start = time.monotonic()
    with (out/"last.out").open("w") as stdout, (out/"timings.txt").open("w") as stderr:
        subprocess.run(cmd, env=env, cwd=out, stdout=stdout, stderr=stderr, check=True)
    elapsed = time.monotonic()-start
    count = a.width*a.height*3
    for batch, file in enumerate(sorted((out/"batches").glob("batch-*.out"))):
        raw, display = codec.split_outputs(file.read_text(), a.width, a.height*a.batch)
        pixels = codec.codec.rgb8(display, a.width, a.height*a.batch)
        for local in range(a.batch):
            frame = batch*a.batch+local
            if frame > a.frames:
                continue
            rgb = pixels[local*count:(local+1)*count]
            prefix = out/f"frame-{frame:04d}"
            codec.codec.write_tiff(prefix.with_suffix(".tif"), a.width, a.height, rgb)
            prefix.with_suffix(".png").write_bytes(codec.codec.png(a.width,a.height,rgb))
    assert (out/"frame-0000.png").read_bytes() == (out/f"frame-{a.frames:04d}.png").read_bytes(), "GPU loop seam differs"
    comparison = None
    if a.compare_cpu:
        (out/"validation.xtl").write_text('"rt:" u_se< "RayTrace"\nu:f_rame := '+function+'\nu:f_rame 0\n')
        reference = captured([str(ROOT/"bin/xetal"), "run", "--ascii", str(out/"validation.xtl")], env=env, cwd=out)
        (out/"validation-cpu.out").write_text(reference.stdout)
        actual = codec.split_outputs((out/"batches/batch-0000.out").read_text(), a.width, a.height*a.batch)
        expected = codec.split_outputs(reference.stdout, a.width, a.height*a.batch)
        comparison = codec.compare_outputs(actual, expected, a.width, a.height*a.batch)
    manifest = dict(width=a.width,height=a.height,frames=a.frames,batch_frames=a.batch,fps=a.fps,triangles=triangles,
        xetal=(ROOT/"XETAL_COMMIT").read_text().strip(),libraries=(ROOT/"XETAL_LIBRARIES_COMMIT").read_text().strip(),
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),geometry_sha256=hashlib.sha256(geom.encode()).hexdigest(),
        renderer_binary_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),function_sha256=hashlib.sha256(function.encode()).hexdigest(),ir_sha256=hashlib.sha256(lowered.stdout.encode()).hexdigest(),backend="opencl:0",float_width="f32",render_process_seconds=elapsed,
        cpu_comparison=comparison,period_units=36,pipe_starts=[0,18],pipe_length=8,inner_radius=1.9,outer_radius=2.2,camera_height=2.15,
        sphere_right=[-3.6,1.15,14],pyramid_left=[3.8,0,32],resident_scene=True,
        algorithm="Shared RayTrace direct diffuse and hard shadows, flat triangles; each pixel traced three times by scalar RGB adapter",
        outputs="Two Float arrays per batch: unclamped linear then shared sRGB; host quantization and encoding only")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Rendered {a.frames} native frames in {elapsed:.2f}s; CPU check: {comparison}",flush=True)


if __name__ == "__main__":
    main()
