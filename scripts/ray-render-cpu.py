#!/usr/bin/env python3
"""Run an X_eTaL scene on CPU; encode its pixels as uncompressed TIFF and PNG."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import shutil
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parent.parent


def rgb8(text, width, height):
    """Validate display RGB and quantize it; scene/display math stays in X_eTaL."""
    values = [float(token) for token in text.split()]
    if len(values) != width * height * 3:
        raise ValueError(f"expected {width * height * 3} RGB samples, got {len(values)}")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("display RGB must be finite and within 0..1")
    return bytes(int(v * 255 + 0.5) for v in values)


def write_tiff(path, width, height, pixels):
    """Adapt raw RGB to installed libtiff; keep its encoder out of this repo."""
    ppm = path.with_suffix(".ppm")
    ppm.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pixels)
    subprocess.run(["ppm2tiff", "-c", "none", "-r", str(height), str(ppm), str(path)], check=True)


def png(width, height, pixels):
    """A convenient RGB preview; the primary TIFF stays uncompressed."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + pixels[y * width * 3:(y + 1) * width * 3] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def prepare_source(source, width, height, angle, row_start, rows, samples_axis, scene):
    """Keep full camera/sample coordinates while selecting independent output rows."""
    for name, value in [("width", width * samples_axis), ("height", height * samples_axis), ("angle", angle)]:
        source = "\n".join(f"{name} := {value}" if line.startswith(f"{name} := ") else line for line in source.splitlines()) + "\n"
    if rows != height or row_start != 0 or samples_axis != 1:
        film_width = width * samples_axis
        sample_count = film_width * rows * samples_axis
        sample_offset = row_start * samples_axis * film_width
        camera = "logocamera" if scene == "logo" else "studiocamera"
        lines = source.splitlines()
        lines = [f"origins := {camera} rt:o_rigins {sample_count}" if line.startswith("origins :=") else line for line in lines]
        lines = [f"pixels := {sample_offset} + r_ange {sample_count}" if line.startswith("pixels :=") else line for line in lines]
        lines = [f"directions := (({camera} rt:r_ays width c_at height)_ {sample_offset} + r_ange {sample_count})" if line.startswith("directions :=") else line for line in lines]
        lines = [f"sampleids := {sample_offset} + (r_ange {sample_count}) - 1" if line.startswith("sampleids :=") else line for line in lines]
        source = "\n".join(lines) + "\n"
        if samples_axis > 1:
            source = source.replace("1.0 rt:s_rgb linear", f"averaged := ('+ r_/_24 {rows} {samples_axis} {width} {samples_axis} 3 r_eshape linear) / {samples_axis ** 2}.0\n1.0 rt:s_rgb {rows * width} 3 r_eshape averaged")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["logo", "glass"], default="logo")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=40)
    parser.add_argument("--samples-axis", type=int, default=1, help="regular subpixel samples along each axis, averaged in X_eTaL")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--rows", type=int, help="output rows in this full-image strip")
    parser.add_argument("--libraries-ref", help="isolated committed library snapshot; does not move the GPU pin")
    parser.add_argument("--angle", type=float, default=0.35, help="vane angle in radians")
    parser.add_argument("--output", type=Path, default=ROOT / "work/ray-integration/logo-cpu")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or not math.isfinite(args.angle):
        parser.error("dimensions must be positive and angle finite")
    rows = args.height if args.rows is None else args.rows
    if args.samples_axis < 1 or args.samples_axis > 4 or rows < 1 or args.row_start < 0 or args.row_start + rows > args.height:
        parser.error("sample axis must be 1..4 and rows must lie within the full image")
    if args.libraries_ref and (len(args.libraries_ref) != 40 or any(c not in "0123456789abcdef" for c in args.libraries_ref)):
        parser.error("libraries-ref must be a full committed SHA")
    if shutil.which("ppm2tiff") is None:
        parser.error("libtiff ppm2tiff is required for uncompressed TIFF output")
    library_command = [str(ROOT / "scripts/xetal-libraries.sh")]
    if args.libraries_ref:
        library_command += ["--snapshot", args.libraries_ref]
    libraries = Path(subprocess.check_output(library_command, text=True).strip())
    library_commit = subprocess.check_output(["git", "-C", str(libraries), "rev-parse", "HEAD"], text=True).strip()
    xetal = subprocess.check_output([str(ROOT / "scripts/xetal.sh")], text=True).strip()
    import os
    env = dict(os.environ, XETAL_PATH=":".join(str(p) for p in sorted(libraries.glob("libs/*/src"))))
    source = (ROOT / "examples/ray" / (args.scene + ".xtl")).read_text()
    scene_sha256 = hashlib.sha256(source.encode()).hexdigest()
    source = prepare_source(source, args.width, args.height, args.angle, args.row_start, rows, args.samples_axis, args.scene)
    prefix = args.output.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(str(prefix) + ".xtl")
    source_path.write_text(source)
    print(f"CPU evaluator: {args.width} x {args.height}, scene {args.scene}; rendering...", flush=True)
    start = time.monotonic()
    result = subprocess.run([xetal, "run", "--seed", "1", "--ascii", str(source_path)], env=env, text=True, capture_output=True)
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    elapsed = time.monotonic() - start
    pixels = rgb8(result.stdout, args.width, rows)
    Path(str(prefix) + ".out").write_text(result.stdout)
    write_tiff(Path(str(prefix) + ".tif"), args.width, rows, pixels)
    Path(str(prefix) + ".png").write_bytes(png(args.width, rows, pixels))
    metadata = dict(backend="X_eTaL CPU evaluator", scene=args.scene, scene_sha256=scene_sha256,
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(), width=args.width, height=rows,
                    frame_width=args.width, frame_height=args.height, row_start=args.row_start,
                    samples_axis=args.samples_axis, samples_per_pixel=args.samples_axis ** 2,
                    sample_pattern="regular centered subpixel grid; linear RGB averaged in X_eTaL",
                    render_seconds=elapsed, compression="None (TIFF)",
                    xetal=(ROOT / "XETAL_COMMIT").read_text().strip(),
                    libraries=library_commit, library_snapshot=bool(args.libraries_ref),
                    paired_gpu_validation="not established for this output",
                    timing_scope="CPU evaluator process wall time, including load and ASCII output; encoding excluded")
    if args.scene == "logo":
        metadata["angle_radians"] = args.angle
    if args.scene == "glass":
        metadata.update(stage="Opaque/mirror composition diagnostic; no dielectric transport", samples_per_pixel=1, seed=123, interaction_cap=4, camera="pinhole", exposure=1.0, display_transfer="sRGB; exposed linear clamped to 0..1", point_light_shadows="opaque", effects="No depth of field, caustics, dispersion or bloom")
    Path(str(prefix) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Rendered in {elapsed:.2f} s: {prefix}.tif and {prefix}.png", flush=True)


if __name__ == "__main__":
    main()
