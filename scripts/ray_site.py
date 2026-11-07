"""Build the ray-tracing gallery from curated, actually rendered media."""
import html
import json
import re
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/softwarewrighter/X_eTaL-gpu"

GLASS = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 64" role="img" aria-label="Planned glass sphere and crystal"><rect width="160" height="64" fill="#202c40"/><path d="M10 55h140" stroke="#667990"/><circle cx="55" cy="32" r="20" fill="none" stroke="#9ccce3" stroke-width="2"/><path d="m105 9 20 23-20 24-20-24zM105 9v47M85 32h40" fill="none" stroke="#9ccce3" stroke-width="2"/></svg>'
BOX = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 64" role="img" aria-label="Planned Cornell Box test scene"><rect width="160" height="64" fill="#202c40"/><path d="M34 5h92v53H34z" fill="#a8a8a2"/><path d="M34 5 55 17v29L34 58z" fill="#9c3333"/><path d="m126 5-21 12v29l21 12z" fill="#3c8255"/><path d="m34 58 21-12h50l21 12" fill="#d0cec4"/><path d="M64 3h32v5H64z" fill="#fff3c2"/><path d="M63 45V28h15v17zm25 0V22h13v23z" fill="#dfded5"/></svg>'


def cards():
    still_manifest = ROOT / "site-assets/ray/logo-still/manifest.json"
    high = still_manifest.is_file()
    poster = "logo-still/still.png" if high else "logo/poster.png"
    label = f"Native {json.loads(still_manifest.read_text())['width']}-pixel CPU still + preview video" if high else "CPU animation &middot; first low-resolution preview"
    if (ROOT / "site-assets/ray/logo-gpu/manifest.json").is_file():
        poster = "logo-gpu/still.png"
        label = "Native 1280-pixel RTX still + CPU preview video"
    return f'''<section id="ray-tracing" aria-labelledby="ray-title">
<h2 id="ray-title">Ray tracing</h2>
<p class="cap">Recorded X_eTaL renders. Open a thumbnail for demos, videos, settings and downloadable frames.</p>
<div class="grid ray-cards">
<a class="card" href="ray-tracing/#logo"><img src="ray-tracing/media/{poster}" width="160" height="64" alt="Silver X and copper lettered arrow vane"><h3>X and rotating _eTaL</h3><p>{label}</p></a>
<a class="card" href="ray-tracing/#glass">{GLASS}<h3>Glass spheres and crystals</h3><p>Planned &middot; transparent scene in development</p></a>
<a class="card" href="ray-tracing/#benchmark">{BOX}<h3>Test / benchmark scene</h3><p>Planned &middot; Cornell Box reference comparison</p></a>
</div></section>'''


def build():
    assets = ROOT / "site-assets/ray/logo"
    manifest = json.loads((assets / "manifest.json").read_text())
    out = ROOT / "pages/ray-tracing"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(assets, out / "media/logo", dirs_exist_ok=True)
    (out / "glass.svg").write_text(GLASS + "\n")
    (out / "benchmark.svg").write_text(BOX + "\n")
    text = (ROOT / "templates/ray-tracing.html").read_text()
    still_dir = ROOT / "site-assets/ray/logo-still"
    still = ""
    if (still_dir / "manifest.json").is_file():
        m = json.loads((still_dir / "manifest.json").read_text())
        shutil.copytree(still_dir, out / "media/logo-still", dirs_exist_ok=True)
        still = f'''<figure id="native-still"><img class="native-still" src="media/logo-still/still.png" width="{m['width']}" height="{m['height']}" alt="Native-resolution ray-traced X_eTaL logo and arrow">
<figcaption><strong>{m['width']} &times; {m['height']} native CPU still</strong> &middot; {m['samples_per_pixel']} sample(s) per pixel. This is newly traced at this native resolution. GPU timing comparisons are pending.</figcaption></figure>
<div class="links"><a href="media/logo-still/still.png">Open full-resolution PNG</a><a href="media/logo-still/still.tif" download>Uncompressed TIFF</a><a href="media/logo-still/masters.zip" download>Master + exact strip sources</a><a href="media/logo-still/manifest.json">Still settings</a></div>
<p class="muted">CPU library snapshot <code>{m['libraries'][:7]}</code>; language <code>{m['xetal'][:7]}</code>. Reproduce with <code>just ray-hires-cpu --libraries-ref {m['libraries']}</code>. The video below is the earlier low-resolution recording.</p>'''
    gpu_dir = ROOT / "site-assets/ray/logo-gpu"
    gpu = ""
    if (gpu_dir / "manifest.json").is_file():
        m = json.loads((gpu_dir / "manifest.json").read_text())
        shutil.copytree(gpu_dir, out / "media/logo-gpu", dirs_exist_ok=True)
        timing = next(line for line in m["timings"] if line.startswith("repeat 2 time:"))
        device_ms = float(re.search(r"kernels ([0-9.]+) ms", timing)[1])
        total_ms = float(re.search(r"total ([0-9.]+) ms", timing)[1])
        gpu = f'''<figure id="gpu-still"><img class="native-still" src="media/logo-gpu/still.png" width="{m['width']}" height="{m['height']}" alt="Native-resolution RTX ray-traced X_eTaL logo">
<figcaption><strong>{m['width']} &times; {m['height']} RTX GPU still</strong> &middot; one sample per pixel. Second resident run: {device_ms / 1000:.2f} s device work, {total_ms / 1000:.2f} s total including output readback. Initial GPU setup: 5.00 s.</figcaption></figure>
<div class="links"><a href="media/logo-gpu/still.png">Open GPU PNG</a><a href="media/logo-gpu/still.tif" download>GPU uncompressed TIFF</a><a href="media/logo-gpu/masters.zip" download>GPU masters + sources</a><a href="media/logo-gpu/manifest.json">Settings and timings</a></div>
<p class="muted">Shared X_eTaL math on RTX 5060 Ti; {m['execution_metrics']['launches']} launches for the complete scene. The experimental scalar-channel adapter traces each pixel three times. The same program passed CPU validation at 8 &times; 4; the CPU still below uses a different library snapshot and is not a matched timing baseline. These timings are individual observations. Reproduce with <code>just ray-render-regions --width 1280 --height 512</code>.</p>'''
    pair_dir = ROOT / "site-assets/ray/image-pair"
    pair = ""
    if (pair_dir / "manifest.json").is_file():
        m = json.loads((pair_dir / "manifest.json").read_text())
        shutil.copytree(pair_dir, out / "media/image-pair", dirs_exist_ok=True)
        comparison = m["cpu_comparison"]
        pixels = "RGB8 pixels are identical." if comparison["rgb8_different_channels"] == 0 else f"{comparison['rgb8_different_channels']} RGB8 channels differ; maximum byte difference {comparison['rgb8_max_absolute_difference']}."
        pair = f'''<details id="image-pair"><summary>CPU/GPU image validation downloads ({m['width']} &times; {m['height']})</summary>
<p>The exact same X_eTaL program produced both TIFFs. These small images test correctness, rather than presentation quality or performance. Maximum raw linear RGB error: {comparison['linear_max_absolute_error']:.3g}; maximum display RGB error: {comparison['display_max_absolute_error']:.3g}. {pixels}</p>
<div class="links"><a href="media/image-pair/cpu.png">CPU PNG</a><a href="media/image-pair/gpu.png">GPU PNG</a><a href="media/image-pair/cpu.tif" download>CPU TIFF</a><a href="media/image-pair/gpu.tif" download>GPU TIFF</a><a href="media/image-pair/masters.zip" download>Paired masters, raw Float arrays and source</a><a href="media/image-pair/manifest.json">Validation settings</a></div>
<p>Linear radiance is preserved before the shared X_eTaL display conversion. The host only quantizes RGB8 and encodes the images. Reproduce with <code>just ray-image-pair</code>.</p></details>'''
    values = {
        "GPU": gpu, "PAIR": pair, "HIRES": still, "WIDTH": str(manifest["width"]), "HEIGHT": str(manifest["height"]),
        "FRAMES": str(manifest["frame_count"]), "FPS": str(manifest["fps"]),
        "LAST": str(manifest["frame_count"] - 1), "REPO": REPO,
        "XETAL": html.escape(manifest["xetal"][:7]), "LIBRARIES": html.escape(manifest["libraries"][:7]),
        "SOURCE": html.escape((assets / "scene.xtl").read_text()),
        "DATA": json.dumps(dict(frames=manifest["frames"], fps=manifest["fps"])),
    }
    for name, value in values.items():
        text = text.replace("@@" + name + "@@", value)
    (out / "index.html").write_text(text)
    print("pages: ray-tracing/index.html (native GPU/CPU logo; glass and benchmark planned)")
