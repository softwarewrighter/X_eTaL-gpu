#!/usr/bin/env python3
"""Compare binary file and text bindings on prepared world-model inputs.
Run ml-bench --check first to prepare those inputs from the pinned model.
"""
import argparse
import importlib.util
import subprocess
import tempfile
import platform
from datetime import date
from pathlib import Path

spec = importlib.util.spec_from_file_location("ml_bench", Path(__file__).with_name("ml-bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def must(cmd):
    result, elapsed = bench.run(cmd)
    if result.returncode:
        raise RuntimeError(f"{cmd[0]}: {result.stderr.strip()}")
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--output", type=Path, default=bench.ROOT / "docs/rtx5060-binary-results.md")
    args = parser.parse_args()
    devices = bench.devices()
    if args.require_device and not devices:
        raise RuntimeError("no OpenCL device; --require-device refuses to skip GPU checks")
    model = bench.MODELS["world-model"]
    if not (bench.WORK / "world-model.xir").exists():
        raise RuntimeError("prepare world-model inputs first: just ml-bench --check")
    rows = []
    # All files created here are private scratch files, cleaned after the run.
    with tempfile.TemporaryDirectory(prefix="binary-", dir=bench.WORK) as scratch:
        scratch = Path(scratch)
        million = scratch / "million.txt"
        million.write_text("0.25\n" * 1_000_000)
        (bench.WORK / "million.xir").write_text("%x = input f64 [1000000]\n%s = reduce add %x\noutput %s\n")
        # facade_driver expects inputs below WORK; use a unique absolute path
        # in each driver's bind expressions instead of moving input files.
        cases = [("world-model", model, ["1897", "348"]),
                 ("million", {"shapes": {"x": "1000000"}}, ["250000.0"])]
        for name, case, expected in cases:
            files = {}
            binds = []
            for tag, shape in case["shapes"].items():
                text = bench.WORK / f"{name}-{tag}.txt" if name != "million" else million
                binary = scratch / f"{name}-{tag}.xta"
                must([bench.GPU, "pack", "f64", shape.replace(" ", ","), text, binary])
                files[tag] = (text, binary)
                binds += ["--bind", f"{tag}=@{binary}"]
            for device in ["cpu"] + devices:
                result, elapsed = must([bench.GPU, "run", bench.WORK / f"{name}.xir", "--device", device] + binds)
                if result.stdout.splitlines() != expected:
                    raise RuntimeError(f"{name} binary CLI on {device}: {result.stdout}")
                print(f"ok: {name}, binary CLI on {device}, exact outputs ({elapsed:.0f} ms)")
            # Measure both routes through the same facade with the same f64 data.
            times = {}
            for route in ["text", "binary"]:
                driver = bench.facade_driver(name, case, outputs=len(expected))
                for tag, shape in case["shapes"].items():
                    old = f'{tag} := {shape} r_eshape n_umbers []N_GET "work/ml-bench/{name}-{tag}.txt"'
                    driver = driver.replace(old, f'{tag} := {shape} r_eshape n_umbers []N_GET "{files[tag][0]}"')
                if route == "binary":
                    driver = "\n".join(line for line in driver.splitlines() if not any(line.startswith(f"{tag} :=") for tag in files)) + "\n"
                    for i, tag in enumerate(files, 1):
                        driver = driver.replace(f"{i} gp:b_ind {tag}", f'{i} gp:b_indFile "{files[tag][1]}"')
                if not devices:
                    driver = driver.replace("gp:r_un d } ck:t_ime 0", "gp:r_un d } ck:t_ime -1")
                path = scratch / f"{name}-{route}.xtl"
                path.write_text(driver)
                result, elapsed = must([bench.XX, "run", path])
                lines = result.stdout.splitlines()
                if lines[:len(expected)] != expected:
                    raise RuntimeError(f"{name} {route} facade: {result.stdout}")
                times[route] = [float(t) for t in lines[len(expected):len(expected) + 4]] + [elapsed]
                print(f"ok: {name}, {route} facade, exact outputs; bind {times[route][1]:.0f} ms")
            rows.append((name, times))
    if not args.check:
        if not devices:
            raise RuntimeError("GPU measurements need an OpenCL device; use --check for CPU checks")
        names = bench.device_names()
        lines = [f"# Binary file binding on {names.get('opencl:0')}", "", f"Measured on {platform.node()}, {date.today()}, {names.get('opencl:0')}; ML {(bench.ROOT / 'XETAL_ML_COMMIT').read_text().strip()[:7]}.", "",
                 "Both routes bind the same f64 inputs and run the default f32 schedule.",
                 "All CLI and facade outputs agree exactly: world-model counts 1897 and 348;",
                 "the million-element sum 250000.0. Binary file creation is outside these timings.",
                 "Each route is one sample; process totals include startup and data loading.", "",
                 "| Case | Route | Bind (ms) | Device run (ms) | Output (ms) | Whole process (ms) |",
                 "| ---- | ----- | --------- | --------------- | ----------- | ------------------ |"]
        for name, times in rows:
            for route, t in times.items():
                lines.append(f"| {name} | {route} | {t[1]:.0f} | {t[2]:.0f} | {t[3]:.0f} | {t[4]:.0f} |")
        lines += ["", "File binding includes reading and decoding the binary file. Text binding",
                  "includes serialization across the bridge, but the text route's initial",
                  "file parsing is outside its bind interval (inside its process total).",
                  "This avoids text array transfer; it still allocates and uploads device",
                  "buffers on every run. Outputs still use the text bridge. Resident weights",
                  "and native binary array calls remain separate work.", ""]
        args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
