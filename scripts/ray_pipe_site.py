"""Animated cards and full-size viewer for the executed pipe fly-through."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "site-assets/ray/pipes-rtx"


def video(prefix, m, controls=False):
    ext = m["selected_animation"].split('.')[-1]
    if ext in ["gif", "webp"]:
        return f'<img class="pipe-loop" src="{prefix}/{m["selected_animation"]}" width="{m["width"]}" height="{m["height"]}" alt="Ray-traced flight through hollow pipes with a sphere and pyramid">'
    codec = 'av01.0.04M.08' if m["selected_codec"] == 'av1' else 'vp9'
    kind = f'video/webm; codecs={codec}' if ext == 'webm' else 'video/mp4'
    fallback = f'<source src="{prefix}/flythrough.mp4" type="video/mp4">' if ext != 'mp4' else ''
    return f'<video class="pipe-loop" width="{m["width"]}" height="{m["height"]}" autoplay muted loop playsinline preload="metadata" poster="{prefix}/poster.png" {"controls" if controls else ""} aria-label="Ray-traced pipe fly-through"><source src="{prefix}/{m["selected_animation"]}" type="{kind}">{fallback}</video>'


def dialog(prefix, m, name):
    return f'''<dialog id="{name}" class="pipe-dialog"><form method="dialog"><button aria-label="Close full-size pipe fly-through">Close</button></form>{video(prefix,m,True)}<p>Native {m['width']} &times; {m['height']}, {m['fps']} fps. A continuous {m['frames']/m['fps']:g}-second loop.</p><a href="{prefix}/flythrough.mp4" download>Download MP4</a></dialog>
<style>.pipe-dialog{{max-width:96vw;border:1px solid #64748b;border-radius:12px;background:#101c2b;color:#eef4ff}}.pipe-dialog::backdrop{{background:#000b}}.pipe-dialog video,.pipe-dialog img{{display:block;width:min({m["width"]}px,88vw);height:auto;max-height:78vh;object-fit:contain}}.pipe-dialog form{{text-align:right}}.pipe-loop{{max-width:100%;height:auto;border-radius:8px}}.card .pipe-loop{{width:100%}}</style>
<script>(()=>{{const d=document.getElementById('{name}');for(const b of document.querySelectorAll('[data-pipe-open="{name}"]'))b.addEventListener('click',e=>{{e.preventDefault();d.showModal();const v=d.querySelector('video');if(v)v.play().catch(()=>{{}})}});d.addEventListener('close',()=>{{const v=d.querySelector('video');if(v)v.pause()}});if(matchMedia('(prefers-reduced-motion: reduce)').matches)for(const v of document.querySelectorAll('video.pipe-loop')){{v.pause();v.controls=true}}}})();</script>'''


def card():
    if not (ASSETS/"manifest.json").is_file():
        return "", ""
    m=json.loads((ASSETS/"manifest.json").read_text())
    prefix="ray-tracing/media/pipes-rtx"
    return (f'<a class="card" href="ray-tracing/#pipes" data-pipe-open="pipe-home-dialog">{video(prefix,m)}<h3>Pipe fly-through</h3><p>GPU ray tracing &middot; continuous {m["frames"]/m["fps"]:g}-second loop</p></a>', dialog(prefix,m,"pipe-home-dialog"))


def section(out):
    if not (ASSETS/"manifest.json").is_file():
        return ""
    m=json.loads((ASSETS/"manifest.json").read_text())
    shutil.copytree(ASSETS,out/"media/pipes-rtx",dirs_exist_ok=True)
    prefix="media/pipes-rtx"
    masters=' '.join(f'<a href="{prefix}/{z["file"]}" download>TIFF masters {z["first_frame"] + 1}-{z["first_frame"]+z["frames"]}</a>' for z in m["masters"])
    return f'''<section id="pipes"><h2>Pipe fly-through</h2><p>Fly through open pipes on a ground plane beneath a blue sky. After the first pipe, a red sphere appears on the right; after the second, a gold pyramid appears on the left. The repeating scene flows continuously back into the first view.</p><a href="#pipes" data-pipe-open="pipe-gallery-dialog">{video(prefix,m)}</a><p>Native {m['width']} &times; {m['height']} &middot; {m['fps']} fps &middot; {m['frames']/m['fps']:g} seconds &middot; GPU-ray-traced frames, one sample per pixel (no antialiasing). Select the animation for the full-size viewer.</p><button data-pipe-open="pipe-gallery-dialog">Open full-size animation</button><div class="links"><a href="{prefix}/flythrough.mp4" download>Download MP4</a><a href="{prefix}/{m['selected_animation']}" download>Download smallest tested animation</a><a href="{prefix}/manifest.json">Settings and format comparison</a></div><details><summary>Uncompressed frame masters and rendering source</summary><div class="links">{masters}</div><p>Every frame is independently ray traced in X_eTaL. Six frames share each resident GPU batch; the scene stays on the GPU between batches. These are opaque faceted shapes with direct light and hard shadows, one sample per pixel. The browser plays recorded frames. The final endpoint was rendered again and matches the first frame; it is excluded from playback to avoid a repeated-frame pause.</p><a href="{prefix}/function.xtl">Ray-tracing function</a></details>{dialog(prefix,m,'pipe-gallery-dialog')}</section>'''
