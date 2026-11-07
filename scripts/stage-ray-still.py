#!/usr/bin/env python3
"""Curate a completed native CPU still and its exact strip sources for the gallery."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent.parent

def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main():
    codec = module("still_codec", "ray-render-cpu.py")
    check = module("still_check", "test-ray-media.py")
    prefix = ROOT / "work/ray-hires/logo-native"
    meta = json.loads(prefix.with_suffix(".json").read_text())
    assert meta["backend"] == "X_eTaL CPU evaluator" and meta["width"] >= 1024
    w, h, samples = meta["width"], meta["height"], meta["samples_axis"]
    pixels = check.tiff_pixels(prefix.with_suffix(".tif").read_bytes(), w, h)
    assert check.png_pixels(prefix.with_suffix(".png").read_bytes(), w, h) == pixels
    template = ROOT / "examples/ray/logo.xtl"
    assert hashlib.sha256(template.read_bytes()).hexdigest() == meta["scene_sha256"]
    source = codec.prepare_source(template.read_text(), w, h, meta["angle_radians"], 0, h, samples, "logo")
    public = dict(meta, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  source_role="Equivalent full-frame X_eTaL program; exact executed strips in masters.zip",
                  display_transfer="sRGB after linear RGB averaging, exposure 1, clamped to 0..1; RGB8 quantized by encoder")
    public.pop("full_frame_source_sha256", None)
    entries = []
    assembled = bytearray()
    next_row = 0
    for strip in meta["strips"]:
        src = Path(strip["source"])
        recorded = json.loads(src.with_suffix(".json").read_text())
        assert strip["row"] == next_row and strip["rows"] == recorded["height"]
        assert recorded["row_start"] == next_row and recorded["frame_height"] == h and recorded["width"] == w
        assert all(recorded[k] == meta[k] for k in ["xetal", "libraries", "scene_sha256", "samples_axis", "samples_per_pixel"])
        assert hashlib.sha256(src.read_bytes()).hexdigest() == strip["source_sha256"] == recorded["source_sha256"]
        assembled.extend(check.tiff_pixels(src.with_suffix(".tif").read_bytes(), w, recorded["height"]))
        next_row += strip["rows"]
        item = dict(strip, source="strips/"+src.name)
        entries.append((item, src))
    assert next_row == h and assembled == pixels
    public["strips"] = [item for item,_ in entries]
    assets = ROOT / "site-assets/ray/logo-still"
    assets.mkdir(parents=True, exist_ok=True)
    for ext in [".png", ".tif"]:
        shutil.copyfile(prefix.with_suffix(ext), assets / ("still"+ext))
    (assets / "scene.xtl").write_text(source)
    (assets / "manifest.json").write_text(json.dumps(public, indent=2)+"\n")
    with zipfile.ZipFile(assets / "masters.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for name in ["still.tif", "scene.xtl", "manifest.json"]:
            archive.write(assets / name, name)
        archive.write(ROOT / "LICENSE", "LICENSE")
        for item, src in entries:
            archive.write(src, item["source"])
            archive.write(src.with_suffix(".json"), str(Path(item["source"]).with_suffix(".json")))
    print(f"staged native {w}x{h} CPU still, {samples**2} samples/pixel, exact strip pixels and sources")


if __name__ == "__main__":
    main()
