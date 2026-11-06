#!/usr/bin/env python3
"""Curate verified rendered frames and uncompressed masters as site inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "work/ray-live/logo")
    args = parser.parse_args()
    source = args.input.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    scene = ROOT / "examples/ray/logo.xtl"
    assert manifest["scene_sha256"] == hashlib.sha256(scene.read_bytes()).hexdigest(), "scene changed after rendering"
    for pin, filename in [("xetal", "XETAL_COMMIT"), ("libraries", "XETAL_LIBRARIES_COMMIT")]:
        assert manifest[pin] == (ROOT / filename).read_text().strip(), "render pin does not match repository"
    for frame in manifest["frames"]:
        name = f"frame-{frame['index']:03d}"
        prefix = source / "frames" / name
        assert hashlib.sha256(prefix.with_suffix(".xtl").read_bytes()).hexdigest() == frame["source_sha256"]
        frame_meta = json.loads(prefix.with_suffix(".json").read_text())
        assert all(frame_meta[k] == manifest[k] for k in ["xetal", "libraries", "scene_sha256"])
        info = subprocess.run(["tiffinfo", "-D", str(prefix.with_suffix(".tif"))], capture_output=True, text=True, check=True)
        assert "Compression Scheme: None" in info.stdout, "master must be uncompressed"
    dest = ROOT / "site-assets/ray/logo"
    (dest / "frames").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scene, dest / "scene.xtl")
    for frame in manifest["frames"]:
        name = f"frame-{frame['index']:03d}"
        shutil.copyfile(source / "frames" / f"{name}.png", dest / "frames" / f"{name}.png")
    poster = source / "poster"
    assert poster.with_suffix(".json").exists(), "render the 320x128 poster before staging media"
    if poster.with_suffix(".json").exists():
        poster_meta = json.loads(poster.with_suffix(".json").read_text())
        assert poster_meta["scene_sha256"] == manifest["scene_sha256"]
        assert all(poster_meta[k] == manifest[k] for k in ["xetal", "libraries"])
        assert poster_meta["angle_radians"] == 0.0
        info = subprocess.run(["tiffinfo", "-D", str(poster.with_suffix(".tif"))], capture_output=True, text=True, check=True)
        assert "Compression Scheme: None" in info.stdout
        manifest["poster"] = {k: poster_meta[k] for k in ["width", "height", "source_sha256", "render_seconds"]}
        shutil.copyfile(poster.with_suffix(".png"), dest / "poster.png")
    else:
        shutil.copyfile(source / "frames/frame-000.png", dest / "poster.png")
    shutil.copyfile(source / "logo.webm", dest / "logo.webm")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(dest / "masters.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        archive.write(ROOT / "LICENSE", "LICENSE")
        if "poster" in manifest:
            for suffix in [".tif", ".xtl", ".json"]:
                archive.write(poster.with_suffix(suffix), "poster" + suffix)
        for frame in manifest["frames"]:
            name = f"frame-{frame['index']:03d}"
            for suffix in [".tif", ".xtl", ".json"]:
                archive.write(source / "frames" / (name + suffix), name + suffix)
    print(f"Staged {manifest['frame_count']} verified frames, WebM and uncompressed master archive in {dest}")


if __name__ == "__main__":
    main()
