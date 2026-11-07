#!/usr/bin/env python3
"""Curate a completed CPU/GPU image pair, preserving raw arrays and exact source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "work/ray-pair/logo")
    p.add_argument("--output", type=Path, default=ROOT / "site-assets/ray/image-pair")
    a = p.parse_args()
    src = a.input.resolve()
    dst = a.output.resolve()
    m = json.loads(src.with_suffix(".json").read_text())
    assert m["cpu_comparison"]["same_source"] and m["outputs"].startswith("linear RGB")
    assert hashlib.sha256(src.with_suffix(".xtl").read_bytes()).hexdigest() == m["source_sha256"]
    dst.mkdir(parents=True, exist_ok=True)
    files = {
        "scene.xtl": src.with_suffix(".xtl"), "gpu.tif": src.with_suffix(".tif"), "gpu.png": src.with_suffix(".png"),
        "cpu.tif": src.parent / m["cpu_comparison"]["cpu_tiff"], "cpu.png": src.parent / m["cpu_comparison"]["cpu_png"],
        "linear-gpu.txt": src.with_suffix(".linear.out"), "linear-cpu.txt": src.with_suffix(".cpu.linear.out"),
        "display-gpu.txt": src.with_suffix(".out"), "display-cpu.txt": src.with_suffix(".cpu.out"),
        "timings.txt": src.with_suffix(".timings"),
    }
    for name, file in files.items():
        shutil.copyfile(file, dst / name)
    # Frozen observations keep their original pins even after future repins.
    m.update(role="Small correctness fixture, not a resolution or performance showcase",
             gpu_tiff="gpu.tif", cpu_tiff="cpu.tif", gpu_png="gpu.png", cpu_png="cpu.png",
             float_outputs="linear-*.txt are pre-display radiance; display-*.txt are shared sRGB before RGB8 quantization",
             artifact_sha256={name: hashlib.sha256((dst/name).read_bytes()).hexdigest() for name in files},
             renderer_worktree_status=subprocess.check_output(["git", "-C", str(ROOT), "status", "--short", "--untracked-files=no"], text=True).strip(),
             renderer_git_base=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
             renderer_binary_sha256=hashlib.sha256((ROOT/"target/release/xetal-gpu").read_bytes()).hexdigest())
    (dst / "manifest.json").write_text(json.dumps(m, indent=2)+"\n")
    with zipfile.ZipFile(dst / "masters.zip", "w", compression=zipfile.ZIP_STORED) as z:
        for name in [*files, "manifest.json"]:
            z.write(dst/name, name)
        z.write(ROOT/"LICENSE", "LICENSE")
    print(f"staged matched CPU/GPU {m['width']}x{m['height']} TIFF/PNG and pre-display Float arrays")


if __name__ == "__main__":
    main()
