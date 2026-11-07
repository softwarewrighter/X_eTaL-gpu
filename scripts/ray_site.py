"""Build the ray-tracing gallery from curated, actually rendered media."""
import html
import json
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
    values = {
        "HIRES": still, "WIDTH": str(manifest["width"]), "HEIGHT": str(manifest["height"]),
        "FRAMES": str(manifest["frame_count"]), "FPS": str(manifest["fps"]),
        "LAST": str(manifest["frame_count"] - 1), "REPO": REPO,
        "XETAL": html.escape(manifest["xetal"][:7]), "LIBRARIES": html.escape(manifest["libraries"][:7]),
        "SOURCE": html.escape((assets / "scene.xtl").read_text()),
        "DATA": json.dumps(dict(frames=manifest["frames"], fps=manifest["fps"])),
    }
    for name, value in values.items():
        text = text.replace("@@" + name + "@@", value)
    (out / "index.html").write_text(text)
    print("pages: ray-tracing/index.html (recorded CPU logo; glass and benchmark planned)")
