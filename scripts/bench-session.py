#!/usr/bin/env python3
"""Repeated trained-model inference: resident weights, refreshed batches,
reference interpreter, and (for timing) NumPy's BLAS: OpenBLAS with tuned
thread counts where NumPy carries it (Linux wheels), or the BLAS the wheel
links otherwise (Accelerate on macOS, which threadpoolctl cannot limit, so
it runs as it is and is named in the report).
Inputs are the world-model fixture's (scripts/world-model-fixture.py), prepared from the pinned model if missing.
"""
import argparse
import contextlib
import importlib.util
import json
import math
import platform
import re
import statistics
import struct
import subprocess
import tempfile
import time
from datetime import date
from pathlib import Path

spec = importlib.util.spec_from_file_location("world_model_fixture", Path(__file__).with_name("world-model-fixture.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def invoke(cmd):
    result, elapsed = bench.run(cmd)
    if result.returncode:
        raise RuntimeError(f"{cmd[0]}: {result.stderr.strip()}")
    return result, elapsed


def save_array(path, shape, items):
    path.write_bytes(b"XTLARR01" + bytes([2, len(shape)]) + b"\0" * 6 +
                     struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<" + "d" * len(items), *items))


def timings(stderr):
    pattern = r"time: setup ([\d.]+) ms, upload ([\d.]+) ms, kernels ([\d.]+) ms \(device\), launches ([\d.]+) ms \(host\), download ([\d.]+) ms, total ([\d.]+) ms"
    return [[float(x) for x in row] for row in re.findall(pattern, stderr)]


def median(values):
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256, 1024, 1936, 8192])
    parser.add_argument("--output", type=Path, default=bench.ROOT / "docs/rtx5060-session-results.md")
    args = parser.parse_args()
    if args.repeats < 5 or any(n < 1 for n in args.batches):
        parser.error("use at least five repeats and positive batch sizes")
    devices = bench.devices()
    if (args.require_device or not args.check) and not devices:
        raise RuntimeError("no OpenCL device; refusing to write GPU measurements")
    np = None
    if not args.check:
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits
        pools = [p for p in threadpool_info() if p.get("user_api") == "blas"]
        if any(p.get("internal_api") == "openblas" for p in pools):
            blas = f"OpenBLAS {pools[0].get('version')}, {pools[0].get('architecture')}"
            thread_settings = [1, 6, 12]
        else:
            # NumPy's macOS wheels link Accelerate, which threadpoolctl does not see or limit.
            try:
                name = np.__config__.CONFIG["Build Dependencies"]["blas"]["name"]
            except (AttributeError, KeyError):
                name = "the wheel's BLAS"
            blas = f"{name} (NumPy's wheel; its own threading, not limited here)"
            thread_settings = ["as is"]
    bench.prepare()
    data = {}
    for tag in bench.MODELS["world-model"]["shapes"]:
        data[tag] = [float(x) for x in (bench.WORK / f"world-model-{tag}.txt").read_text().split()]
    rows = []
    with tempfile.TemporaryDirectory(prefix="session-", dir=bench.WORK) as directory:
        directory = Path(directory)
        for batch in ([16] if args.check else args.batches):
            held = min(batch, 387)
            shapes = {"X": [batch, 432], "T": [batch, 144], "Xh": [held, 432], "Th": [held, 144],
                      "w1": [432, 64], "b1": [64], "w2": [64, 144], "b2": [144]}
            values = {tag: bench.cycle(data[tag], math.prod(shape)) for tag, shape in shapes.items()}
            program = directory / f"model-{batch}.xir"
            program.write_text(bench.world_model_twin(batch, held))
            binds = []
            for tag, shape in shapes.items():
                path = directory / f"{tag}.xta"
                save_array(path, shape, values[tag])
                binds += ["--bind", f"{tag}=@{path}"]
            repeat = 3 if args.check else args.repeats
            cpu, _ = invoke([bench.GPU, "run", program, "--device", "cpu", "--repeat", str(repeat), "--time"] + binds)
            expected = cpu.stdout.splitlines()
            if batch == 1936 and expected != ["1897", "348"]:
                raise RuntimeError(f"full model no longer matches evaluator baseline: {expected}")
            skip = 1 if args.check else 3
            cpu_ms = median([float(x) for x in re.findall(r"cpu time: total ([\d.]+) ms", cpu.stderr)][skip:])
            candidates = []
            for device in devices:
                for tile in ([0] if args.check else [0, 8, 16]):
                    base = [bench.GPU, "run", program, "--device", device, "--tile", str(tile), "--repeat", str(repeat), "--time"] + binds
                    resident, _ = invoke(base)
                    stream, _ = invoke(base + ["--refresh", "X", "--refresh", "Xh"])
                    if resident.stdout.splitlines() != expected or stream.stdout.splitlines() != expected:
                        raise RuntimeError(f"batch {batch} outputs disagree on {device}, tile {tile}")
                    resident_stats, stream_stats = timings(resident.stderr), timings(stream.stderr)
                    if len(resident_stats) != repeat or len(stream_stats) != repeat:
                        raise RuntimeError("missing repeat timing records")
                    if any(t[0] or t[1] for t in resident_stats[1:]):
                        raise RuntimeError("resident repeat performed setup or upload")
                    if any(t[0] for t in stream_stats[1:]):
                        raise RuntimeError("stream repeat rebuilt the program")
                    skip = 1 if args.check else 3
                    streamed_ms = median([t[5] for t in stream_stats[skip:]])
                    candidates.append({"device": device, "tile": tile, "resident": median([t[5] for t in resident_stats[skip:]]),
                                       "stream": streamed_ms, "setup": stream_stats[0][0], "first": stream_stats[0][5],
                                       "upload": median([t[1] for t in stream_stats[skip:]]), "kernel": median([t[2] for t in stream_stats[skip:]])})
                    print(f"ok: batch {batch}+{held}, {device}, tile {tile}: resident {candidates[-1]['resident']:.3f}, refreshed {streamed_ms:.3f} ms", flush=True)
            if args.check:
                continue
            arrays = {tag: np.array(v, dtype=np.float32).reshape(shapes[tag]) for tag, v in values.items()}

            def predict():
                counts = []
                for x, t in [("X", "T"), ("Xh", "Th")]:
                    z = np.maximum(arrays[x] @ arrays["w1"] + arrays["b1"], np.float32(0))
                    logits = z @ arrays["w2"] + arrays["b2"]
                    probability = np.float32(1) / (np.float32(1) + np.exp(-logits))
                    counts.append(str(int(np.all((probability > .5) == (arrays[t] > .5), axis=1).sum())))
                return counts

            numpy_rows = []
            for threads in thread_settings:
                limits = threadpool_limits(limits=threads, user_api="blas") if isinstance(threads, int) else contextlib.nullcontext()
                with limits:
                    samples = []
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        got = predict()
                        samples.append((time.perf_counter() - start) * 1000)
                        if got != expected:
                            raise RuntimeError(f"NumPy ({blas}) disagrees on batch {batch}: {got} vs {expected}")
                    numpy_rows.append((median(samples[3:]), threads))
            best_cpu, threads = min(numpy_rows, key=lambda r: r[0])
            best = min(candidates, key=lambda r: r["stream"])
            saving = best_cpu - best["stream"]
            # Both start with in-memory inputs. Charge the GPU's full first call
            # (not just compilation); then compare n calls on CPU and GPU.
            break_even = max(1, math.ceil((best["first"] - best["stream"]) / saving)) if saving > 0 else None
            rows.append(dict(batch=batch, held=held, counts=expected, interpreter=cpu_ms, cpu=best_cpu, threads=threads,
                             cpu_candidates=numpy_rows, candidates=candidates, best=best, break_even=break_even))
            print(f"CPU {blas.split(' (')[0]} {best_cpu:.3f} ms ({threads} threads); stream speedup {best_cpu / best['stream']:.2f}x, break-even {break_even}", flush=True)
        if not args.check:
            names = bench.device_names()
            device_name = names.get("opencl:0", "opencl:0")
            ml_commit = (bench.ROOT / "XETAL_ML_COMMIT").read_text()[:7]
            xetal_commit = (bench.ROOT / "XETAL_COMMIT").read_text()[:7]
            threads_text = ("the fastest of " + ", ".join(str(t) for t in thread_settings[:-1]) + f" and {thread_settings[-1]} BLAS threads"
                            if len(thread_settings) > 1 else "NumPy's BLAS as it is")
            wins = [r for r in rows if r["break_even"] is not None]
            verdict = (["This model does not beat optimized CPU inference with refreshed inputs at",
                        "any measured batch size. Resident-only timings isolate execution reuse,",
                        "but repeating identical inputs is not a practical throughput speedup."]
                       if not wins else
                       ["With refreshed inputs the GPU is ahead of the CPU at batch " + ", ".join(str(r["batch"]) for r in wins) +
                        " (break-even calls in the table) and behind elsewhere. Resident-only",
                        "timings isolate execution reuse, but repeating identical inputs is not a",
                        "practical throughput speedup."])
            lines = [f"# Repeated inference on the {device_name}", "", f"Measured {date.today()} on `{platform.node()}` ({device_name}).",
                     f"NumPy {np.__version__}; {blas}. ML {ml_commit}; X_eTaL {xetal_commit}.", "",
                     "The trained world model runs both dense layers, biases, ReLU, sigmoid",
                     "and exact-frame counts. The two batches contain N states and min(N, 387)",
                     "held-out states. Larger batches cycle the committed data; this is a throughput",
                     "test of a trained model, not additional unseen evaluation data.", "",
                     f"Each setting runs {args.repeats} times; report medians after discarding the first three.",
                     f"CPU: {threads_text}. GPU: the fastest refreshed",
                     "run of tiles 0, 8 and 16. All settings give the same exact counts as the",
                     "reference interpreter; the full batch gives the evaluator baseline 1897/348.", ""] + verdict + ["",
                     "| Batch N + held | Interpreter (ms) | CPU BLAS (ms), threads | Resident (ms) | New batch (ms) | Tile | Setup (ms) | First call (ms) | Stream speedup | Break-even calls |",
                     "| -------------- | ---------------- | ---------------------- | ------------- | -------------- | ---- | ---------- | --------------- | -------------- | ---------------- |"]
            for r in rows:
                b = r["best"]
                lines.append(f"| {r['batch']} + {r['held']} | {r['interpreter']:.3f} | {r['cpu']:.3f}, {r['threads']} | {b['resident']:.3f} | {b['stream']:.3f} | {b['tile']} | {b['setup']:.1f} | {b['first']:.1f} | {r['cpu']/b['stream']:.2f}x | {r['break_even'] or '--'} |")
            lines += ["", "Resident: no inputs change. New batch: reupload X and Xh each call, while",
                      "weights, biases and targets stay resident. The same batch bytes are reuploaded",
                      "to isolate transfer costs. The interval includes host conversion, upload,",
                      "kernel launches, synchronization and count download; file reads, binary",
                      "conversion and process startup are excluded on both CPU and GPU.",
                      "Break-even charges the GPU's complete first call, then its measured refreshed",
                      "latency, against repeated CPU calls. A dash means no measured steady-state win.",
                      "The first call uses the vendor's existing driver cache; this is not a cold-cache",
                      "setup measurement. Shape changes require a new program/session.", "",
                      "## All GPU settings", "", "| N | Tile | Resident (ms) | New batch (ms) | Upload (ms) | Kernels (ms) |", "| - | ---- | ------------- | -------------- | ----------- | ------------ |"]
            for r in rows:
                for c in r["candidates"]:
                    lines.append(f"| {r['batch']} | {c['tile']} | {c['resident']:.3f} | {c['stream']:.3f} | {c['upload']:.3f} | {c['kernel']:.3f} |")
            lines += ["", "## All CPU settings", "", "| N | BLAS threads | Median (ms) |", "| - | ------------ | ----------- |"]
            for r in rows:
                for value, threads in r["cpu_candidates"]:
                    lines.append(f"| {r['batch']} | {threads} | {value:.3f} |")
            args.output.write_text("\n".join(lines) + "\n")
            args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
