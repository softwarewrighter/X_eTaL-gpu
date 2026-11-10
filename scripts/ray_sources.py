"""Source navigation for recorded ray demos, with explicit library revision labels."""
import html
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/softwarewrighter/X_eTaL-gpu"
LIBRARIES = "https://github.com/softwarewrighter/X_eTaL-libraries"
DOCS = "https://softwarewrighter.github.io/X_eTaL-libraries/doc"


def blob(path):
    revision = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", path], cwd=ROOT, text=True).strip()
    if not revision:
        raise ValueError(f"recorded source is not committed: {path}")
    return f"{REPO}/blob/{revision}/{path}"


def shared(recording, names):
    m = json.loads((ROOT / "site-assets/ray" / recording / "manifest.json").read_text())
    pin = m["libraries"]
    xrefs = " ".join(f'<a href="{DOCS}/libs-{name}-src-{name}.xtl.html">{name} xref</a>' for name in names)
    xrefs += " " + " ".join(f'<a href="{DOCS}/libs-{name}-src-{name}.xtlm.html">{name} macro xref</a>' for name in names if name in ["Scene", "Extrusion"])
    sources = " ".join(f'<a href="{LIBRARIES}/blob/{pin}/libs/{name}/src/{name}.xtl">{name} source</a>' for name in names)
    sources += " " + " ".join(f'<a href="{LIBRARIES}/blob/{pin}/libs/{name}/src/{name}.xtlm">{name} macro source</a>' for name in names if name in ["Scene", "Extrusion"])
    return f'<div class="ray-library-sources"><p>Shared-library cross-references (current docs): {xrefs}</p><p>Sources used for this recording, library revision <a href="{LIBRARIES}/tree/{pin}/libs"><code>{pin[:7]}</code></a>: {sources}</p></div>'


def panel(recording, prefix, authoring, recorded, driver, libraries, note=""):
    links = [f'<a href="{REPO}/blob/main/{authoring}">Current scene authoring (X_eTaL)</a>']
    for file, label in recorded:
        path = f"site-assets/ray/{recording}/{file}"
        if not (ROOT / path).is_file():
            raise ValueError(f"missing recorded source: {path}")
        links.append(f'<a href="{prefix}/{file}">{html.escape(label)}</a>')
        links.append(f'<a href="{blob(path)}">Browse {html.escape(file)} on GitHub</a>')
    links.append(f'<a href="{blob(driver)}">Current rendering/encoding driver (Python)</a>')
    return f'<div class="ray-sources" id="sources-{recording}" data-recording="{recording}"><h3>Source code</h3><nav class="links ray-source-links" aria-label="Scene and renderer source">' + "".join(links) + '</nav>' + (f'<p class="muted">{html.escape(note)}</p>' if note else '') + shared(recording, libraries) + '</div>'


def pipe_dialog(prefix):
    pin = json.loads((ROOT / "site-assets/ray/pipes-rtx/manifest.json").read_text())["libraries"]
    return f'<nav class="links ray-source-links" aria-label="Pipe scene source"><a href="{prefix}/geometry.xtl">Geometry source (X_eTaL)</a><a href="{prefix}/function.xtl">Ray-tracing source (X_eTaL)</a><a href="{DOCS}/libs-RayTrace-src-RayTrace.xtl.html#l.s_hade">RayTrace shading xref (current docs)</a><a href="{LIBRARIES}/blob/{pin}/libs/RayTrace/src/RayTrace.xtl">Recorded RayTrace source ({pin[:7]})</a></nav>'
