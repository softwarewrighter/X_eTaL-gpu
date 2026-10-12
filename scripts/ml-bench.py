#!/usr/bin/env python3
"""X_eTaL-ML's benchmarks on this repository's devices (its ask GA1,
part 1): for each of bench/dense.xtl, inner-wide.xtl and softmax.xtl in
the pinned X_eTaL-ML (work/xetal-ml), the program is run by the pinned
xetal (the reference), and the same computation as an XIR program
(written here from the benchmark's definition, its data generated as
r_eshape cycles it) runs on the reference interpreter (exactly), on
every OpenCL device (Floats within 1e-5, the device's own time apart
from the host's: --time) and, with --facade, through the gpu extension
from an X_eTaL program under xetal-x (the bridge's time apart from the
device's).

  scripts/ml-bench.py --check          # the equivalences only (the gate)
  scripts/ml-bench.py                  # timed, docs/ml-results.md rewritten
  scripts/ml-bench.py --no-facade      # timed, without the xetal-x route
  scripts/ml-bench.py --require-device --output docs/rtx5060-ml-results.md
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "work" / "xetal-ml"
WORK = ROOT / "work" / "ml-bench"
XETAL = ROOT / "bin" / "xetal"
GPU = ROOT / "target" / "release" / "xetal-gpu"
XX = ROOT / "scripts" / "xx"
COMPARE = ROOT / "scripts" / "compare-out.py"
RESULTS = ROOT / "docs" / "ml-results.md"


def cycle(values, n):
    return [values[i % len(values)] for i in range(n)]


def product_twin(name, rows, k, cols, times):
    """x (rows by k) times w (k by cols), the sum of the ravel, `times`
    times, the sums added as '+ r_/ adds a vector (from the right)."""
    out = [f"%x = input f64 [{rows} {k}]", f"%w = input f64 [{k} {cols}]"]
    for t in range(1, times + 1):
        out += [f"%p{t} = matmul %x %w", f"%r{t} = ravel %p{t}", f"%s{t} = reduce add %r{t}"]
    out += fold_sum(times)
    return "\n".join(out) + "\n"


def fold_sum(times):
    """%s1 + (%s2 + (... + %sN)): the right fold of a vector of sums."""
    out = []
    acc = f"%s{times}"
    for t in range(times - 1, 0, -1):
        out.append(f"%a{t} = map add %s{t} {acc}")
        acc = f"%a{t}"
    return out + [f"output {acc}"]


def softmax_twin(rows, cols, times):
    """nn:s_oftmax by row (the row's largest taken off, exp, divided by
    the row's sum), the sum of the ravel, `times` times."""
    out = [f"%x = input f64 [{rows} {cols}]", f"%c = const i64 [{cols}] " + " ".join(str(i) for i in range(cols))]
    for t in range(1, times + 1):
        out += [
            f"%mx{t} = reduce max axis=2 %x",
            f"%mxs{t} = table left %mx{t} %c",
            f"%d{t} = map sub %x %mxs{t}",
            f"%e{t} = map exp %d{t}",
            f"%es{t} = reduce add axis=2 %e{t}",
            f"%ess{t} = table left %es{t} %c",
            f"%p{t} = map div %e{t} %ess{t}",
            f"%r{t} = ravel %p{t}",
            f"%s{t} = reduce add %r{t}",
        ]
    out += fold_sum(times)
    return "\n".join(out) + "\n"


BENCHES = {
    "dense": {
        "what": "a dense layer's product, 1024 x 16 by 16 x 16, four times",
        "twin": product_twin("dense", 1024, 16, 16, 4),
        "inputs": {"x": cycle([0.1, 0.2, -0.3, 0.5, 0.7], 1024 * 16), "w": cycle([0.3, -0.1, 0.2, 0.0, 0.4], 16 * 16)},
        "shapes": {"x": "1024 16", "w": "16 16"},
    },
    "inner-wide": {
        "what": "the CNN's dense layer, 1 x 1352 by 1352 x 10, forty times",
        "twin": product_twin("inner-wide", 1, 1352, 10, 40),
        "inputs": {"x": cycle([0.5, 0.25, 0.125], 1352), "w": cycle([0.01, -0.02, 0.03], 1352 * 10)},
        "shapes": {"x": "1 1352", "w": "1352 10"},
    },
    "softmax": {
        "what": "softmax by row on 1000 x 64, ten times",
        "twin": softmax_twin(1000, 64, 10),
        "inputs": {"x": cycle([0.5, -1.25, 2.0, 0.0, 3.5], 1000 * 64)},
        "shapes": {"x": "1000 64"},
    },
}


def run(cmd, cwd=ROOT, env=None):
    t = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, env=env)
    return r, (time.perf_counter() - t) * 1000.0


def parse_time(stderr):
    m = re.search(r"kernels ([\d.]+) ms \(device\), launches ([\d.]+) ms \(host\).*total ([\d.]+) ms", stderr)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (None, None, None)


def devices():
    r, _ = run([GPU, "devices"])
    return [line.split()[0] for line in r.stdout.splitlines() if line.startswith("opencl:")]


def facade_driver(name, b):
    """An X_eTaL program that runs the twin through the gpu extension and
    times the whole (ck:t_ime), the device's part known from --time."""
    rel = os.path.relpath(WORK, ROOT)
    lines = ['"gp:" u_se< "Gpu"', '"ck:" u_se< "Clock"']
    lines.append(f'prog := []N_GET "{rel}/{name}.xir"')
    for i, (tag, shape) in enumerate(b["shapes"].items(), 1):
        lines.append(f'{tag} := {shape} r_eshape n_umbers []N_GET "{rel}/{name}-{tag}.txt"')
    binds = "; ".join(f"n := {i} gp:b_ind {tag}" for i, tag in enumerate(b["shapes"], 1))
    # Each part timed on its own: load (parse and check), bind (the
    # inputs across the bridge as text), run (the device: compile,
    # upload, kernels, download), output (the result back as text).
    lines.append("tload := '{ p -> gp:l_oad p } ck:t_ime prog")
    lines.append(f"tbind := '{{ k -> {binds}; n }} ck:t_ime 0")
    lines.append("trun := '{ d -> gp:r_un d } ck:t_ime 0")
    lines.append("tout := '{ i -> gp:o_utput i } ck:t_ime 1")
    lines.append("gp:o_utput 1")
    lines.append("tload"); lines.append("tbind"); lines.append("trun"); lines.append("tout")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-facade", action="store_true")
    parser.add_argument("--require-device", action="store_true", help="fail instead of skipping the GPU when no device is visible")
    parser.add_argument("--output", type=Path, default=RESULTS, help="destination for the timed Markdown report")
    args = parser.parse_args()
    check_only = args.check
    facade = not args.no_facade and not check_only
    subprocess.run([ROOT / "scripts" / "xetal-ml.sh"], check=True, capture_output=True)
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "xetal-gpu-cli"], cwd=ROOT / "components", check=True)
    if facade:
        subprocess.run([ROOT / "scripts" / "gpu-ext.sh"], check=True, capture_output=True)
    WORK.mkdir(parents=True, exist_ok=True)
    devs = devices()
    if not devs:
        if args.require_device:
            sys.exit("ml-bench: no OpenCL device found; --require-device refuses to skip GPU measurements")
        print("ml-bench: no OpenCL device here; the device columns are skipped")
    ml_commit = (ROOT / "XETAL_ML_COMMIT").read_text()[:7]
    xetal_commit = (ROOT / "XETAL_COMMIT").read_text()[:7]
    libs = ":".join(str(d) for d in sorted((ML / "libs").glob("*/src")))
    rows = []
    fail = 0
    for name, b in BENCHES.items():
        (WORK / f"{name}.xir").write_text(b["twin"])
        for tag, items in b["inputs"].items():
            (WORK / f"{name}-{tag}.txt").write_text("\n".join(repr(x) for x in items) + "\n")
        env = dict(os.environ, XETAL_PATH=libs)
        r, t_eval = run([XETAL, "run", f"bench/{name}.xtl"], cwd=ML, env=env)
        if r.returncode != 0:
            print(f"FAIL: xetal on {name}: {r.stderr.strip()}"); fail = 1; continue
        want = WORK / f"{name}.out"
        want.write_text(r.stdout)
        binds = []
        for tag in b["inputs"]:
            binds += ["--bind", f"{tag}=@{WORK / f'{name}-{tag}.txt'}"]
        row = {"name": name, "what": b["what"], "result": r.stdout.strip(), "xetal_ms": t_eval}
        for dev in ["cpu"] + devs:
            r, t = run([GPU, "run", WORK / f"{name}.xir", "--device", dev, "--time"] + binds)
            if r.returncode != 0:
                print(f"FAIL: xetal-gpu on {dev}, {name}: {r.stderr.strip()}"); fail = 1; continue
            got = WORK / f"{name}-{dev.replace(':', '')}.out"
            got.write_text(r.stdout)
            tol = "0" if dev == "cpu" else "1e-5"
            c = subprocess.run([COMPARE, want, got, tol], capture_output=True, text=True)
            if c.returncode != 0:
                print(f"FAIL: {name} on {dev}: {(c.stdout + c.stderr).strip()}"); fail = 1; continue
            diff = abs(float(r.stdout.split()[0]) - float(row["result"]))
            print(f"ok: {name} on {dev}, {c.stdout.strip()} ({t:.0f} ms)")
            if dev == "cpu":
                row["cpu_ms"] = t
            else:
                k, h, total = parse_time(r.stderr)
                row[dev] = {"diff": diff, "kernels": k, "launch": h, "total": total, "process": t}
        if facade and devs:
            drv = WORK / f"{name}-facade.xtl"
            drv.write_text(facade_driver(name, b))
            r, t = run([XX, "run", os.path.relpath(drv, ROOT)])
            if r.returncode != 0:
                print(f"FAIL: facade on {name}: {r.stderr.strip()}"); fail = 1
            else:
                parts = r.stdout.split("\n")
                got_val, times = parts[0], [float(x) for x in parts[1:5]]
                diff = abs(float(got_val) - float(row["result"]))
                rel = diff / max(abs(float(row["result"])), 1.0)
                if rel <= 1e-5:
                    print(f"ok: {name} through the facade on opencl:0 (load {times[0]:.0f}, bind {times[1]:.0f}, run {times[2]:.0f}, output {times[3]:.0f} ms)")
                    row["facade"] = times
                else:
                    print(f"FAIL: {name} through the facade: {got_val} against {row['result']}"); fail = 1
        rows.append(row)
    if check_only or fail:
        print(f"ml-bench: {len(rows)} benchmarks" + (", all agree with the evaluator" if not fail else ", FAILURES"))
        sys.exit(fail)
    write_results(rows, devs, ml_commit, xetal_commit, facade, args.output)
    print(f"ml-bench: {len(rows)} benchmarks, all agree; {args.output} written")


def write_results(rows, devs, ml_commit, xetal_commit, facade, output=RESULTS):
    import platform
    host = platform.node()
    out = [
        "# X_eTaL-ML's benchmarks on GPUs",
        "",
        "X_eTaL-ML's ask GA1 (its `docs/gpu-asks.md`): its batch programs,",
        f"from X_eTaL-ML {ml_commit} (`XETAL_ML_COMMIT`), run by the pinned",
        f"X_eTaL {xetal_commit} (the reference) and as XIR programs on this",
        "repository's interpreter and devices. Written by `scripts/ml-bench.py`",
        f"(`just ml-bench`) on `{host}`; whole-process times include reading",
        "the data and, on a device, compiling the kernels. The device's time",
        "is the kernels' own, from OpenCL profiling events, summed; the",
        "launches' time is what the host waited from the first enqueue to",
        "`finish`. Through the facade, the program runs under `xetal-x` with",
        "the gpu extension, each call timed with `ck:t_ime`: the run is the",
        "device's (compiling, uploading, the kernels, reading back), and the",
        "bridge's cost is the bind and output calls, which carry the arrays",
        "as text. The run totals are dominated by compiling the kernels on",
        "every run (there is no kernel cache yet); the kernels' own time is",
        "the column that compares devices.",
        "Every result agrees with the evaluator: exactly on the interpreter,",
        "within a relative 1e-5 on a device (f32).",
        "",
        "| Benchmark | What | Evaluator's result | Evaluator (ms) | Interpreter (ms) |",
        "| --------- | ---- | ------------------ | -------------- | ---------------- |",
    ]
    for r in rows:
        out.append(f"| {r['name']} | {r['what']} | {r['result']} | {r['xetal_ms']:.0f} | {r['cpu_ms']:.0f} |")
    for dev in devs:
        names = {d: n for d, n in device_names().items()}
        out += ["", f"## {dev}: {names.get(dev, dev)}", "",
                "| Benchmark | Largest difference | Kernels, device (ms) | Launches, host (ms) | Run total (ms) | Whole process (ms) |" + (" Facade: run (ms) | Facade: bridge, bind and output (ms) |" if facade else ""),
                "| --------- | ------------------ | -------------------- | ------------------- | -------------- | ------------------ |" + (" ---------------- | ------------------------------------ |" if facade else "")]
        for r in rows:
            d = r.get(dev)
            if not d:
                continue
            line = f"| {r['name']} | {d['diff']:.3g} | {d['kernels']:.3f} | {d['launch']:.2f} | {d['total']:.0f} | {d['process']:.0f} |"
            if facade:
                if dev == "opencl:0" and "facade" in r:
                    f = r["facade"]
                    line += f" {f[2]:.0f} | {f[1] + f[3]:.0f} |"
                else:
                    line += " -- | -- |"
            out.append(line)
    out += ["", "The interpreter is this repository's reference interpreter (`--device cpu`), f64.", ""]
    output.write_text("\n".join(out))


def device_names():
    r, _ = run([GPU, "devices"])
    names = {}
    for line in r.stdout.splitlines():
        if line.startswith("opencl:"):
            dev, rest = line.split(None, 1)
            names[dev] = rest.split(" (")[0]
    return names


if __name__ == "__main__":
    main()
