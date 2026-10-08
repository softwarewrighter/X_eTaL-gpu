#!/usr/bin/env python3
"""Build the live site's landing page, pages/index.html: what this
repository is, how a program reaches a GPU, the measured results, and
links to the documentation (pages/doc, built by scripts/doc-site.sh),
the libraries, the extension and the plan. Its key lines are drawn
decorated by the pinned xetal (xetal render --html), and its footer
names the commits it was built from.

  scripts/build-pages.py
"""
import html
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/softwarewrighter/X_eTaL-gpu"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def render(line):
    out = subprocess.run([str(ROOT / "bin/xetal"), "render", "--html", "-e", line], capture_output=True, text=True, check=True).stdout
    return out.strip()


LINES = [
    ("The acceptance pipeline, as Accel writes it", "l:p_ipeline := { x w -> l:s_um l:r_elu x + x * w }"),
    ("A layer of a network", "l:d_ense := { x wb -> (x '+ '* i_nner -1 d_rop wb) + (o_ffsets t_ally x) 'r_ight t_able r_avel -1 t_ake wb }"),
    ("The same product on the GPU, from X_eTaL", "c := a gp:p_roduct a"),
]

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>X_eTaL on GPUs</title>
<meta name="description" content="X_eTaL array programs on GPUs, old ones included: one typed program, the same answer from the evaluator, an interpreter and an OpenCL device.">
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1d1f; --muted:#5f6368; --card:#ffffff; --line:#e3e0d8; --accent:#2457c5; --chip:#eef2fb; --good:#1f7a3a; --bad:#a3361f; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#141518; --fg:#e8e6e3; --muted:#a0a4ab; --card:#1d1f23; --line:#30333a; --accent:#8fb0ff; --chip:#262b36; --good:#5fcf7f; --bad:#f08a70; }} }}
:root[data-theme="dark"] {{ --bg:#141518; --fg:#e8e6e3; --muted:#a0a4ab; --card:#1d1f23; --line:#30333a; --accent:#8fb0ff; --chip:#262b36; --good:#5fcf7f; --bad:#f08a70; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main, footer {{ max-width: 980px; margin: 0 auto; padding: 0 16px; }}
header {{ padding: 40px 0 8px; }}
h1 {{ font-size: 2rem; margin: 0 0 8px; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.2rem; margin: 32px 0 10px; }}
.lede {{ color: var(--muted); max-width: 46rem; margin: 0; }}
a {{ color: var(--accent); }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; display:flex; flex-direction:column; gap:8px; }}
.card h3 {{ margin:0; font-size:1.05rem; }}
.card p {{ margin:0; color:var(--muted); }}
.card .go {{ margin-top:auto; font-weight:600; }}
pre.flow {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 18px; overflow-x:auto; font-size:.85rem; line-height:1.35; margin:0; }}
.xline {{ margin: 6px 0 14px; padding: 8px 10px; background: var(--chip); border-radius: 8px; overflow-x: auto; font-size: .92rem; line-height: 1.6; }}
.xline code, code, pre {{ font-family: "JuliaMono", "DejaVu Sans Mono", Menlo, ui-monospace, monospace; }}
.cap {{ color: var(--muted); font-size: .9rem; margin: 0; }}
.c-number {{ color: #9c6500; }} .c-symbol {{ color: #0b7285; }} .c-comment {{ color: var(--muted); }}
.c-builtin {{ color: #6f42c1; }} .c-libfunc {{ color: #2457c5; }} .c-string {{ color: #1f7a3a; }}
table {{ border-collapse: collapse; width: 100%; font-size: .92rem; }}
.scroll {{ overflow-x: auto; background:var(--card); border:1px solid var(--line); border-radius:12px; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
tr:last-child td {{ border-bottom: 0; }}
td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
.good {{ color: var(--good); font-weight: 600; }} .bad {{ color: var(--bad); font-weight: 600; }}
@media (max-width: 480px) {{ pre.flow {{ font-size: .68rem; padding: 12px 10px; }} }}
footer {{ border-top:1px solid var(--line); margin-top: 40px; padding-top:16px; padding-bottom:32px; color:var(--muted); font-size:.85rem; }}
</style>
</head>
<body>
<main>
<header>
<h1>X_eTaL on GPUs</h1>
<p class="lede"><a href="https://github.com/softwarewrighter/X_eTaL">X_eTaL</a> states a computation as whole-array operations, which already expose their parallelism. This repository runs the same typed X_eTaL programs on GPUs through OpenCL C 1.2, aiming at old cards the current CUDA toolkits have retired, and proves every answer against the X_eTaL evaluator. The goal is the concept, not speed.</p>
</header>

<h2>One program, three executions</h2>
<pre class="flow">                X_eTaL program (.xtl)
                        |
          +-------------+-------------+
          v             v             v
      evaluator    XIR interpreter   OpenCL on a GPU
          |             |             |
          +-------------+-------------+
                        v
                  the same arrays</pre>

<h2>Read the code</h2>
<div class="grid">
<div class="card"><h3>Documentation</h3><p>Every library, example program and the GPU facade: doc comments, examples that run, the source drawn decorated, every name linked to its definition and uses.</p><a class="go" href="doc/index.html">Browse the docs</a></div>
<div class="card"><h3>Accel</h3><p>The acceleratable subset of X_eTaL as a library: elementwise arithmetic, masks, reductions, products, a dense layer, a softmax. Each export is one array expression.</p><a class="go" href="doc/libs-Accel-src-Accel.xtl.html">Accel's reference</a></div>
<div class="card"><h3>The gpu extension</h3><p>X_eTaL programs that call the GPU themselves, through X_eTaL-extensions' native ABI and its <code>xetal-x</code> bridge.</p><a class="go" href="doc/extensions-gpu-lib-Gpu.xtl.html">Gpu's reference</a></div>
</div>

<h2>In X_eTaL</h2>
{lines}

<h2>Measured</h2>
<p class="cap">One X_eTaL program, <a href="doc/extensions-gpu-demos-offload.xtl.html">offload.xtl</a>, on an Apple M1 Max (OpenCL 1.2). The GPU side includes the bridge, which carries arrays as text.</p>
<div class="scroll"><table>
<tr><th>Work</th><th>Evaluator</th><th>GPU through the extension</th><th></th></tr>
<tr><td>512 by 512 product</td><td class="n">14545 ms</td><td class="n">462 ms</td><td class="good">31 times faster</td></tr>
<tr><td>a million items, elementwise, summed</td><td class="n">188 ms</td><td class="n">2447 ms</td><td class="bad">13 times slower</td></tr>
</table></div>
<p class="cap">Products gain even through the bridge; elementwise work does not. Why, and which macros and extensions are justified: <a href="{repo}/blob/main/docs/macros-and-extensions.md">the analysis</a>.</p>

<h2>More</h2>
<div class="grid">
<div class="card"><h3>The plan</h3><p>Architecture decisions, the sagas done and to come: products, models, the newer GPU, then the old cards.</p><a class="go" href="{repo}/blob/main/docs/plan.md">docs/plan.md</a></div>
<div class="card"><h3>Asks for X_eTaL</h3><p>What this needs from the language: an accelerator IR and a lowering to it, static shapes, a native hook with binary arrays.</p><a class="go" href="{repo}/blob/main/docs/xetal-asks.md">docs/xetal-asks.md</a></div>
<div class="card"><h3>Source</h3><p>The IR, the OpenCL emitter, the runtime, the tool, the extension, the tests.</p><a class="go" href="{repo}">GitHub</a></div>
</div>
</main>
<footer>
Built from <a href="{repo}/commit/{commit}">X_eTaL-gpu {commit}</a>, with X_eTaL {xetal} and X_eTaL-extensions {ext} (whose xetal-x, with X_eTaL v0.1.0, wrote the docs).
<br>Copyright (c) 2026 Michael A Wright. MIT License.
<br><a href="https://github.com/softwarewrighter/X_eTaL">X_eTaL</a> &middot; <a href="https://softwarewrighter.github.io/X_eTaL/">its live demo</a> &middot; <a href="https://softwarewrighter.github.io/X_eTaL-demos/">X_eTaL-demos</a> &middot; <a href="https://softwarewrighter.github.io/X_eTaL-ML/">X_eTaL-ML</a>
</footer>
</body>
</html>
"""


def main():
    lines = "\n".join(
        f'<p class="cap">{html.escape(cap)}</p>\n<div class="xline">{render(code)}</div>' for cap, code in LINES
    )
    out = ROOT / "pages" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(PAGE.format(
        lines=lines, repo=REPO, commit=git("rev-parse", "--short", "HEAD"),
        xetal=(ROOT / "XETAL_COMMIT").read_text()[:7], ext=(ROOT / "XETAL_EXTENSIONS_COMMIT").read_text()[:7]))
    print(f"pages: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
