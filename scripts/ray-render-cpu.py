#!/usr/bin/env python3
"""Run the X_eTaL logo on CPU; encode its pixels as uncompressed TIFF and PNG."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=40)
    parser.add_argument("--angle", type=float, default=0.35, help="vane angle in radians")
    parser.add_argument("--output", type=Path, default=ROOT / "work/ray-integration/logo-cpu")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or not math.isfinite(args.angle):
        parser.error("dimensions must be positive and angle finite")
    if shutil.which("ppm2tiff") is None:
        parser.error("libtiff ppm2tiff is required for uncompressed TIFF output")
    libraries = Path(subprocess.check_output([str(ROOT / "scripts/xetal-libraries.sh")], text=True).strip())
    xetal = subprocess.check_output([str(ROOT / "scripts/xetal.sh")], text=True).strip()
    import os
    env = dict(os.environ, XETAL_PATH=":".join(str(p) for p in sorted(libraries.glob("libs/*/src"))))
    source = (ROOT / "examples/ray/logo.xtl").read_text()
    scene_sha256 = hashlib.sha256(source.encode()).hexdigest()
    for name, value in [("width", args.width), ("height", args.height), ("angle", args.angle)]:
        source = "\n".join(f"{name} := {value}" if line.startswith(f"{name} := ") else line for line in source.splitlines()) + "\n"
    prefix = args.output.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(str(prefix) + ".xtl")
    source_path.write_text(source)
    print(f"CPU evaluator: {args.width} x {args.height}, vane {args.angle} radians; rendering...", flush=True)
    start = time.monotonic()
    result = subprocess.run([xetal, "run", "--seed", "1", "--ascii", str(source_path)], env=env, text=True, capture_output=True)
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    elapsed = time.monotonic() - start
    pixels = rgb8(result.stdout, args.width, args.height)
    Path(str(prefix) + ".out").write_text(result.stdout)
    write_tiff(Path(str(prefix) + ".tif"), args.width, args.height, pixels)
    Path(str(prefix) + ".png").write_bytes(png(args.width, args.height, pixels))
    metadata = dict(backend="X_eTaL CPU evaluator", scene_sha256=scene_sha256,
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(), width=args.width, height=args.height,
                    angle_radians=args.angle, render_seconds=elapsed, compression="None (TIFF)",
                    xetal=(ROOT / "XETAL_COMMIT").read_text().strip(),
                    libraries=(ROOT / "XETAL_LIBRARIES_COMMIT").read_text().strip())
    Path(str(prefix) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Rendered in {elapsed:.2f} s: {prefix}.tif and {prefix}.png", flush=True)


if __name__ == "__main__":
    main()
