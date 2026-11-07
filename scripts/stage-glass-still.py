#!/usr/bin/env python3
"""Freeze an executed glass still and independent matched-source validation."""
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
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--device-name", nargs="+", required=True)
    p.add_argument("--driver", required=True)
    p.add_argument("--output", type=Path, default=ROOT / "site-assets/ray/glass-rtx")
    a = p.parse_args()
    source, validation = a.input.resolve(), a.validation.resolve()
    meta = json.loads(source.with_suffix(".json").read_text())
    vmeta = json.loads(validation.with_suffix(".json").read_text())
    assert meta["scene"] == vmeta["scene"] == "glass"
    assert meta["width"] >= 1024 and meta["height"] >= 512
    assert vmeta["cpu_comparison"]["same_source"]
    for key in ["xetal", "libraries", "renderer_binary_sha256", "float_width", "interaction_cap", "seed"]:
        assert meta[key] == vmeta[key], key
    assert hashlib.sha256(source.with_suffix(".xtl").read_bytes()).hexdigest() == meta["source_sha256"]
    assert not subprocess.check_output(["git", "diff", "HEAD", "--", "components"], cwd=ROOT)
    dst = a.output.resolve()
    dst.mkdir(parents=True, exist_ok=True)
    files = {
        "still.png": source.with_suffix(".png"), "still.tif": source.with_suffix(".tif"),
        "scene.xtl": source.with_suffix(".xtl"), "linear.txt": source.with_suffix(".linear.out"),
        "display.txt": source.with_suffix(".out"), "timings.txt": source.with_suffix(".timings"),
        "validation.xtl": validation.with_suffix(".xtl"), "validation.json": validation.with_suffix(".json"),
        "validation-linear-cpu.txt": validation.with_suffix(".cpu.linear.out"),
        "validation-linear-gpu.txt": validation.with_suffix(".linear.out"),
        "validation-display-cpu.txt": validation.with_suffix(".cpu.out"),
        "validation-display-gpu.txt": validation.with_suffix(".out"),
    }
    for name, file in files.items():
        shutil.copyfile(file, dst / name)
    meta.update(device_name=" ".join(a.device_name), driver=a.driver,
                renderer_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                role="First native glass transport still; one sample per pixel, bounded continuation preview",
                visual_limitations="Visible structured sampling patterns and black capped paths; not final glass quality",
                raw_float_storage="Native linear.txt and display.txt are stored in masters.zip, without duplicate standalone copies",
                validation_scope="Small independent image from identical CPU/GPU source; full-resolution CPU equivalence not measured",
                timing_context="Preview and validation shader compilation overlapped this job; observations are not an isolated benchmark",
                omissions=["antialiasing", "soft shadows", "depth of field", "dispersion", "caustics", "transparent shadows"],
                artifact_sha256={n: hashlib.sha256((dst/n).read_bytes()).hexdigest() for n in files})
    (dst / "manifest.json").write_text(json.dumps(meta, indent=2)+"\n")
    with zipfile.ZipFile(dst / "masters.zip", "w", compression=zipfile.ZIP_STORED) as z:
        for name in [*files, "manifest.json"]:
            z.write(dst/name, name)
        z.write(ROOT / "LICENSE", "LICENSE")
    # The native Float streams are large. Keep their exact bytes once in
    # the stored archive; the manifest hashes also cover these members.
    for name in ["linear.txt", "display.txt"]:
        (dst/name).unlink()
    print(f"staged native {meta['width']}x{meta['height']} glass still and raw Float evidence")


if __name__ == "__main__":
    main()
