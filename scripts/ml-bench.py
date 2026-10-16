#!/usr/bin/env python3
"""X_eTaL-ML's workloads on this repository's devices, from their own
source (its ask GA1; plan A4b): each program of the pinned X_eTaL-ML
(work/xetal-ml) is run by the pinned xetal (the reference) and lowered
whole by `xetal-gpu run` (components/lower: a statement the device
cannot take is evaluated by xetal, the rest becomes kernels) on the
reference interpreter (exactly), on every OpenCL device (Floats within
1e-5; the device's own time apart from the host's: --time) and, where
ml/ has the program written through the macro gp:a_ccel<, under
xetal-x with the gpu extension (the first call's time, with the
lowering, and a second call's).

The programs: bench/dense, inner-wide, softmax and conv (sums over
repeated products and softmaxes), the world model (432 pixels in, 64
ReLU, 144 sigmoid out; all 1,936 states as one batch and the 387 held
out: counts of next frames predicted exactly), the digit reader over
1,000 MNIST test digits (the count read right), the learned denoiser
(20 steps from given noise, 256 pictures) and the tiny GPT over the
first 1,000 names (mean loss and letters predicted right). A model's
reference is its reg-rs baseline (--check) or a run of the program
(timed); a benchmark's is always a run. A program missing from the
pin is skipped, saying so. The lowering is cached under
work/lower-cache (components/lower); the timed mode clears the cache
first and reports each program's cold lowering.

  scripts/ml-bench.py --check          # the equivalences only (the gate)
  scripts/ml-bench.py                  # timed, docs/ml-results.md rewritten
  scripts/ml-bench.py --no-macro       # timed, without the xetal-x route
  scripts/ml-bench.py --require-device --output docs/rtx5060-ml-results.md
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "work" / "xetal-ml"
WORK = ROOT / "work" / "ml-bench"
CACHE = ROOT / "work" / "lower-cache"
XETAL = ROOT / "bin" / "xetal"
XETAL_X = ROOT / "bin" / "xetal-x"
GPU = ROOT / "target" / "release" / "xetal-gpu"
COMPARE = ROOT / "scripts" / "compare-out.py"
RESULTS = ROOT / "docs" / "ml-results.md"
EXTS = ["--ext", "extensions/gpu", "--ext", "work/xetal-extensions/extensions/clock"]

# name, X_eTaL-ML's program, its baseline (models), what it computes,
# the output lines the table shows as the result, the macro program in ml/
PROGRAMS = [
    ("dense", "bench/dense.xtl", None, "a dense layer's product, 1024 x 16 by 16 x 16, four times, summed", (1,), "ml/dense.xtl"),
    ("inner-wide", "bench/inner-wide.xtl", None, "the CNN's dense layer, 1 x 1352 by 1352 x 10, forty times, summed", (1,), "ml/inner-wide.xtl"),
    ("softmax", "bench/softmax.xtl", None, "softmax by row on 1000 x 64, ten times, summed", (1,), "ml/softmax.xtl"),
    ("conv", "bench/conv.xtl", None, "nine shifted copies of a 28 x 28 picture times 8 x 9 filters, ten times, summed", (1,), "ml/conv.xtl"),
    ("world-model", "demos/world-model/world-model.xtl", "demos/world-model/reg/cli-world-model.out",
     "432 pixels in, 64 ReLU, 144 sigmoid out; every state's next frame as one batch of 1,936, the 387 held out, a dream of twelve steps: the next frames predicted exactly", (2, 3), "ml/world-model.xtl"),
    ("cnn-batch", "demos/cnn-digits/batch.xtl", "demos/cnn-digits/reg/cli-batch.out",
     "the digit reader over the first 1,000 MNIST test digits: windows, 8 filters, ReLU, pooling, the dense layer, argmax; the count read right", (1,), None),
    ("diffusion", "demos/diffusion/diffusion.xtl", "demos/diffusion/reg/cli-diffusion.out",
     "the learned denoiser: 20 steps from given noise, 128 pictures of 10 x 10 at once; how many have a shape's 8 to 40 pixels on", (42,), None),
    ("micro-gpt-batch", "demos/micro-gpt/batch.xtl", "demos/micro-gpt/reg/cli-batch.out",
     "the tiny GPT's forward pass over the first 1,000 names, a name per call (the loop on the evaluator): the mean loss per token and the next letters right", (1, 2), None),
]


# Lines of a program's output a device in f32 cannot be expected to give,
# compared on the interpreter only: the line number and why.
DEVICE_SKIP = {
    "diffusion": {1: "the program's own check of one picture against the trainer's sampler asks for 1e-6; f32 gives 8.5e-6 (asked of X_eTaL-ML: a tolerance f32 meets, as the world model's does)"},
}


def without_lines(path, skip):
    """A copy of an output without the given lines (1-based), for a comparison."""
    kept = [l for i, l in enumerate(path.read_text().splitlines(), 1) if i not in skip]
    out = path.with_name(path.stem + "-compared" + path.suffix)
    out.write_text("\n".join(kept) + "\n")
    return out


def run(cmd, cwd=ROOT, env=None):
    t = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, env=env)
    return r, (time.perf_counter() - t) * 1000.0


def run_warm(cmd, dev, cwd, env):
    """A device run twice, the second reported: the first after a change
    compiles the kernels cold and says nothing about the device."""
    r, t = run(cmd, cwd, env)
    if dev == "cpu" or r.returncode != 0:
        return r, t
    return run(cmd, cwd, env)


def parse_time(stderr):
    m = re.search(r"kernels ([\d.]+) ms \(device\), launches ([\d.]+) ms \(host\).*total ([\d.]+) ms", stderr)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (None, None, None)


def devices():
    r, _ = run([GPU, "devices"])
    return [line.split()[0] for line in r.stdout.splitlines() if line.startswith("opencl:")]


def device_names():
    r, _ = run([GPU, "devices"])
    names = {}
    for line in r.stdout.splitlines():
        if line.startswith("opencl:"):
            dev, _, rest = line.partition(" ")
            names[dev] = rest.split(" (")[0].strip()
    return names


def ml_env():
    libs = ":".join(str(d) for d in sorted((ML / "libs").glob("*/src")))
    return dict(os.environ, XETAL_PATH=libs + ":" + str(ROOT / "libs" / "Accel" / "src"))


def compare(want, got, tol):
    c = subprocess.run([COMPARE, want, got, tol], capture_output=True, text=True)
    return c.returncode == 0, (c.stdout + c.stderr).strip()


def largest_difference(want, got):
    """The largest absolute difference between two outputs' numbers, for the table."""
    worst = 0.0
    for a, b in zip(want.read_text().split(), got.read_text().split()):
        try:
            worst = max(worst, abs(float(a) - float(b)))
        except ValueError:
            pass
    return worst


def bench_program(name, rel, baseline, what, shown, macro, devs, check_only, use_macro, env):
    """One program: the reference, the cold lowering (timed mode), the
    interpreter, every device, the macro route. A row for the table, or
    None when the program is missing; a failure count."""
    prog = ML / rel
    if not prog.exists():
        print(f"skipped: {name}: no {rel} in the pinned X_eTaL-ML")
        return None, 0
    cwd = prog.parent
    want = WORK / f"{name}.out"
    row = {"name": name, "what": what}
    if check_only and baseline:
        want.write_text((ML / baseline).read_text())
        row["xetal_ms"] = None
    else:
        r, t = run([XETAL, "run", "--seed", "1", "--ascii", prog.name], cwd, env)
        if r.returncode != 0:
            print(f"FAIL: xetal on {name}: {r.stderr.strip()[:300]}")
            return None, 1
        want.write_text(r.stdout)
        row["xetal_ms"] = t
    row["result"] = result_summary(want.read_text(), shown)
    fail = 0
    if not check_only:
        r, t = run([GPU, "lower", prog.name], cwd, env)
        if r.returncode != 0:
            print(f"FAIL: lowering {name}: {r.stderr.strip()[:300]}")
            return None, 1
        row["lower_s"] = t / 1000.0
        row["host"] = sum(1 for line in r.stderr.splitlines() if line.startswith("evaluated by xetal:"))
    for dev in ["cpu"] + devs:
        r, t = run_warm([GPU, "run", prog.name, "--device", dev, "--time"], dev, cwd, env)
        if r.returncode != 0:
            print(f"FAIL: xetal-gpu on {dev}, {name}: {r.stderr.strip()[:300]}")
            fail = 1
            continue
        got = WORK / f"{name}-{dev.replace(':', '')}.out"
        got.write_text(r.stdout)
        skip = DEVICE_SKIP.get(name, {}) if dev != "cpu" else {}
        ok, msg = compare(without_lines(want, skip), without_lines(got, skip), "0" if dev == "cpu" else "1e-5")
        if skip:
            msg += f"; line{'s' if len(skip) > 1 else ''} {', '.join(str(i) for i in skip)} not compared on a device"
        if not ok:
            print(f"FAIL: {name} on {dev}: {msg}")
            fail = 1
            continue
        print(f"ok: {name} on {dev}, {msg} ({t:.0f} ms)")
        if dev == "cpu":
            row["cpu_ms"] = t
        else:
            k, h, total = parse_time(r.stderr)
            row[dev] = {"diff": largest_difference(without_lines(want, skip), without_lines(got, skip)), "kernels": k, "launch": h, "total": total, "process": t}
    if use_macro and macro and devs:
        r, t = run([XETAL_X] + EXTS + ["run", "--seed", "1", "--ascii", macro], ROOT, env)
        if r.returncode != 0:
            print(f"FAIL: {macro} through the macro: {r.stderr.strip()[:300]}")
            fail = 1
        else:
            lines = r.stdout.splitlines()
            values, times = lines[:-2], [float(x) for x in lines[-2:]]
            got = WORK / f"{name}-macro.out"
            got.write_text("\n".join(values) + "\n")
            ok, msg = macro_agrees(want, shown, values)
            if ok:
                print(f"ok: {macro} through the macro on opencl:0, {msg} (first call {times[0]:.0f} ms, second {times[1]:.0f} ms)")
                row["macro"] = times
            else:
                print(f"FAIL: {macro} through the macro: {msg}")
                fail = 1
    return row, fail


def macro_agrees(want, shown, values):
    """The macro program prints the program's result lines (a benchmark's
    sum; the world model's two counts): the same within 1e-5."""
    lines = want.read_text().splitlines()
    expected = [lines[i - 1] for i in shown]
    if len(values) != len(expected):
        return False, f"{values} against {expected}"
    for a, b in zip(expected, values):
        if a == b:
            continue
        try:
            x, y = float(a), float(b)
        except ValueError:
            return False, f"{b} against {a}"
        if abs(x - y) > 1e-5 * max(abs(x), 1.0):
            return False, f"{b} against {a}"
    return True, f"{len(values)} line{'s' if len(values) != 1 else ''} agree (tolerance 1e-05)"


def result_summary(text, shown):
    """What the table shows of a program's output: the given lines."""
    lines = text.splitlines()
    return " and ".join(lines[i - 1] for i in shown if i <= len(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-macro", action="store_true")
    parser.add_argument("--require-device", action="store_true", help="fail instead of skipping the GPU when no device is visible")
    parser.add_argument("--output", type=Path, default=RESULTS, help="destination for the timed Markdown report")
    args = parser.parse_args()
    check_only = args.check
    use_macro = not args.no_macro
    subprocess.run([ROOT / "scripts" / "xetal-ml.sh"], check=True, capture_output=True)
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "xetal-gpu-cli"], cwd=ROOT / "components", check=True)
    if use_macro:
        subprocess.run([ROOT / "scripts" / "gpu-ext.sh"], check=True, capture_output=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if not check_only and CACHE.exists():
        shutil.rmtree(CACHE)
    devs = devices()
    if not devs:
        if args.require_device:
            sys.exit("ml-bench: no OpenCL device found; --require-device refuses to skip GPU measurements")
        print("ml-bench: no OpenCL device here; the device columns are skipped")
    env = ml_env()
    rows, fail = [], 0
    for name, rel, baseline, what, shown, macro in PROGRAMS:
        row, f = bench_program(name, rel, baseline, what, shown, macro, devs, check_only, use_macro, env)
        fail |= f
        if row:
            rows.append(row)
    if check_only or fail:
        print(f"ml-bench: {len(rows)} programs from source" + (", all agree with the evaluator" if not fail else ", FAILURES"))
        sys.exit(fail)
    ml_commit = (ROOT / "XETAL_ML_COMMIT").read_text()[:7]
    xetal_commit = (ROOT / "XETAL_COMMIT").read_text()[:7]
    write_results(rows, devs, ml_commit, xetal_commit, use_macro, args.output)
    print(f"ml-bench: {len(rows)} programs from source, all agree; {args.output} written")


def write_results(rows, devs, ml_commit, xetal_commit, use_macro, output=RESULTS):
    host = platform.node().split(".")[0]
    names = device_names()
    out = [
        "# X_eTaL-ML's workloads on GPUs, from source",
        "",
        "X_eTaL-ML's ask GA1 (its `docs/gpu-asks.md`): its benchmarks and its",
        f"batch programs, from X_eTaL-ML {ml_commit} (`XETAL_ML_COMMIT`), run by the",
        f"pinned X_eTaL {xetal_commit} (the reference) and lowered from their own",
        "source by `xetal-gpu run` (`components/lower`, plan A4b: nothing is",
        "written by hand or by a generator) on this repository's interpreter",
        "and devices. Written by `scripts/ml-bench.py` (`just ml-bench`) on",
        f"`{host}`. A statement the device cannot take (a sort, a text, a loop",
        "over calls) is evaluated by the pinned `xetal` on the host and its",
        "value bound as an input; the `host` column counts them. The",
        "lowering runs the evaluator once per such statement, so it is slow",
        "the first time (the `lowering, cold` column) and cached after",
        "(`work/lower-cache`, keyed by everything it read and the tool); the",
        "whole-process times are with a warm cache and include reading the",
        "data and, on a device, compiling the kernels. The device's time is",
        "the kernels' own, from OpenCL profiling events, summed; the launches'",
        "time is what the host waited from the first enqueue to `finish`.",
        "Through the macro, the program in `ml/` runs under `xetal-x` with",
        "the gpu extension: the accelerated function's text goes to the",
        "runner, which lowers it for the arguments' shapes on the first call",
        "(the first call's time includes that, compiling and the upload) and",
        "keeps the session for the second; the arguments cross the bridge as",
        "text each call (ask G8), which is most of the world model's second",
        "call. Every result agrees with the evaluator: exactly on the",
        "interpreter, within a relative 1e-5 on a device (f32). A device run",
        "is made twice and the second reported, since the first after a",
        "change compiles the kernels cold. These are single samples after",
        "warmup, not medians. The evaluator and interpreter are reference",
        "implementations, not optimized CPU library baselines",
        "(`docs/m1max-session-results.md` compares against those); compare",
        "each GPU with its own host's evaluator.",
        "",
        "| Program | What | Evaluator's result | Evaluator (ms) | Interpreter (ms) | Lowering, cold (s) | Host |",
        "| ------- | ---- | ------------------ | -------------- | ---------------- | ------------------ | ---- |",
    ]
    for r in rows:
        out.append(f"| {r['name']} | {r['what']} | {r['result']} | {r['xetal_ms']:.0f} | {r['cpu_ms']:.0f} | {r.get('lower_s', 0):.1f} | {r.get('host', 0)} |")
    for dev in devs:
        cols = " Macro: first call (ms) | Macro: second call (ms) |" if use_macro and dev == "opencl:0" else ""
        dashes = " ---------------------- | ----------------------- |" if cols else ""
        out += ["", f"## {dev}: {names.get(dev, dev)}", "",
                "| Program | Largest difference | Kernels, device (ms) | Launches, host (ms) | Run total (ms) | Whole process (ms) |" + cols,
                "| ------- | ------------------ | -------------------- | ------------------- | -------------- | ------------------ |" + dashes]
        for r in rows:
            d = r.get(dev)
            if not d:
                continue
            line = f"| {r['name']} | {d['diff']:.3g} | {d['kernels']:.3f} | {d['launch']:.2f} | {d['total']:.0f} | {d['process']:.0f} |"
            if cols:
                m = r.get("macro")
                line += f" {m[0]:.0f} | {m[1]:.0f} |" if m else " -- | -- |"
            out.append(line)
    for name, lines in DEVICE_SKIP.items():
        if any(r["name"] == name for r in rows):
            for i, why in lines.items():
                out += ["", f"Not compared on a device: {name}'s output line {i}, {why}."]
    out += ["", "The interpreter is this repository's reference interpreter (`--device cpu`), f64.",
            "The `host` column: statements evaluated by `xetal` instead of the device",
            "(the world model's sort-free preparation is all device work; the",
            "denoiser's two text prints; the GPT batch's loop over names).", ""]
    output.write_text("\n".join(out))


if __name__ == "__main__":
    main()
