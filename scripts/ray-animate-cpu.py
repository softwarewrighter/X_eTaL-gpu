#!/usr/bin/env python3
"""Render independent X_eTaL CPU frames, retaining TIFF masters and web media."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "work/ray-live/logo")
    args = parser.parse_args()
    if min(args.frames, args.fps, args.width, args.height, args.jobs) <= 0 or args.frames < 4:
        parser.error("positive dimensions/fps/jobs and at least four frames required")
    if args.width % 2 or args.height % 2:
        parser.error("WebM recording dimensions must be even")
    if not shutil.which("ffmpeg") or not shutil.which("ppm2tiff"):
        parser.error("ffmpeg and libtiff ppm2tiff are required")
    subprocess.run([str(ROOT / "scripts/xetal-libraries.sh")], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([str(ROOT / "scripts/xetal.sh")], check=True, stdout=subprocess.DEVNULL)
    out = args.output.resolve()
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    scene_hash = hashlib.sha256((ROOT / "examples/ray/logo.xtl").read_bytes()).hexdigest()
    pins = {"xetal": (ROOT / "XETAL_COMMIT").read_text().strip(),
            "libraries": (ROOT / "XETAL_LIBRARIES_COMMIT").read_text().strip()}

    def render(index):
        angle = 2 * math.pi * index / args.frames
        prefix = frames / f"frame-{index:03d}"
        meta_path = prefix.with_suffix(".json")
        if meta_path.exists() and all(prefix.with_suffix(ext).is_file() for ext in [".png", ".tif", ".xtl", ".out"]):
            meta = json.loads(meta_path.read_text())
            if (meta.get("scene_sha256") == scene_hash and meta.get("width") == args.width
                    and meta.get("height") == args.height and meta.get("angle_radians") == angle
                    and all(meta.get(k) == v for k, v in pins.items())):
                return index, meta, True
        result = subprocess.run([sys.executable, str(ROOT / "scripts/ray-render-cpu.py"),
                                 "--width", str(args.width), "--height", str(args.height),
                                 "--angle", repr(angle), "--output", str(prefix)],
                                text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"frame {index}: {result.stdout}\n{result.stderr}")
        return index, json.loads(meta_path.read_text()), False

    start = time.monotonic()
    print(f"Rendering {args.frames} real CPU frames, {args.width}x{args.height}, {args.jobs} independent jobs", flush=True)
    metadata = [None] * args.frames
    cached_frames = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(render, i) for i in range(args.frames)]
        for future in as_completed(futures):
            index, meta, cached = future.result()
            metadata[index] = meta
            cached_frames += int(cached)
            print(f"frame {index + 1}/{args.frames}: {'cached' if cached else 'rendered'} ({meta['render_seconds']:.1f} s)", flush=True)
    video = out / "logo.webm"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(args.fps),
                    "-i", str(frames / "frame-%03d.png"), "-frames:v", str(args.frames),
                    "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "18", "-pix_fmt", "yuv420p", str(video)], check=True)
    manifest = dict(backend="X_eTaL CPU evaluator", width=args.width, height=args.height,
                    frame_count=args.frames, fps=args.fps, rotation_axis=[0, 1, 0], pivot=[0, 0.5, 0],
                    scene_sha256=scene_hash, **pins, lighting="Point light, diffuse surfaces, hard shadows; no reflections",
                    master_format="RGB8 TIFF, Compression=None", video_format="WebM VP9 (lossy web derivative)",
                    render_wall_seconds=time.monotonic() - start, cached_frames=cached_frames,
                    total_frame_seconds=sum(m["render_seconds"] for m in metadata),
                    frames=[dict(index=i, angle_radians=m["angle_radians"], render_seconds=m["render_seconds"],
                                 source_sha256=m["source_sha256"]) for i, m in enumerate(metadata)])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Animation ready: {video}; uncompressed masters in {frames}", flush=True)


if __name__ == "__main__":
    main()
