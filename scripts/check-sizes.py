#!/usr/bin/env python3
"""The reductions and the pipeline at sizes beyond one work-group:
for each size N, deterministic data is generated into work/sizes/
(Ints in -1000..1000 and Floats in [0, 1)), an X_eTaL program on the
Accel library reads it and prints the sum, the largest item, an inner
product and the pipeline (the pinned evaluator's answer), and the
same computation as an XIR program with inputs bound to the same
files runs on the reference interpreter (exactly) and on every
OpenCL device found (Floats within 1e-5). Each run is timed.

  scripts/check-sizes.py [N...]        # default 1024 65536 1048576
  scripts/check-sizes.py --table N...  # also print a markdown table of the timings
"""
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work" / "sizes"
XT = ROOT / "scripts" / "xt"
GPU = ROOT / "target" / "release" / "xetal-gpu"
COMPARE = ROOT / "scripts" / "compare-out.py"

XTL = '''"ac:" u_se< "Accel"
x := f_loor n_umbers []N_GET "{d}/{n}-x.txt"
w := f_loor n_umbers []N_GET "{d}/{n}-w.txt"
ac:s_um x
ac:l_argest x
x ac:d_ot w
x ac:p_ipeline w
f := n_umbers []N_GET "{d}/{n}-f.txt"
ac:s_um f
f ac:d_ot f
'''

XIR = '''%x = input i64 [{n}]
%w = input i64 [{n}]
%s = reduce add %x
%m = reduce max %x
%xw = map mul %x %w
%d = reduce add %xw
%p = map add %x %xw
%zero = const i64 [] 0
%r = map max %p %zero
%ps = reduce add %r
%f = input f64 [{n}]
%fs = reduce add %f
%ff = map mul %f %f
%fd = reduce add %ff
output %s
output %m
output %d
output %ps
output %fs
output %fd
'''


def generate(n):
    WORK.mkdir(parents=True, exist_ok=True)
    rng = random.Random(n)
    for tag, make in (("x", lambda: rng.randint(-1000, 1000)), ("w", lambda: rng.randint(-1000, 1000)),
                      ("f", lambda: round(rng.random(), 6))):
        p = WORK / f"{n}-{tag}.txt"
        if not p.exists():
            p.write_text("\n".join(str(make()) for _ in range(n)) + "\n")
    rel = os.path.relpath(WORK, ROOT)
    (WORK / f"{n}.xtl").write_text(XTL.format(d=rel, n=n))
    (WORK / f"{n}.xir").write_text(XIR.format(n=n))


def timed(cmd, cwd=ROOT):
    t = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r, time.perf_counter() - t


def devices():
    r = subprocess.run([GPU, "devices"], capture_output=True, text=True)
    return [line.split()[0] for line in r.stdout.splitlines() if line.startswith("opencl:")]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    table = "--table" in sys.argv
    sizes = [int(a) for a in args] or [1024, 65536, 1048576]
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "xetal-gpu-cli"], cwd=ROOT / "components", check=True)
    devs = devices()
    if not devs:
        print("check-sizes: no OpenCL device here; the GPU side is skipped")
    rows = []
    fail = 0
    for n in sizes:
        generate(n)
        want = WORK / f"{n}.out"
        r, t_eval = timed([XT, "run", os.path.relpath(WORK / f"{n}.xtl", ROOT)])
        if r.returncode != 0:
            print(f"FAIL: xetal on {n}: {r.stderr.strip()}"); fail = 1; continue
        want.write_text(r.stdout)
        binds = []
        for tag in ("x", "w", "f"):
            binds += ["--bind", f"{tag}=@{WORK / f'{n}-{tag}.txt'}"]
        row = {"n": n, "xetal": t_eval}
        for dev, tol in [("cpu", "0")] + [(d, "1e-5") for d in devs]:
            r, t = timed([GPU, "run", WORK / f"{n}.xir", "--device", dev] + binds)
            if r.returncode != 0:
                print(f"FAIL: xetal-gpu on {dev}, {n}: {r.stderr.strip()}"); fail = 1; continue
            got = WORK / f"{n}-{dev.replace(':', '')}.out"
            got.write_text(r.stdout)
            c = subprocess.run([COMPARE, want, got, tol], capture_output=True, text=True)
            if c.returncode == 0:
                print(f"ok: {n} items on {dev}, {c.stdout.strip()} ({t:.2f} s; xetal {t_eval:.2f} s)")
            else:
                print(f"FAIL: {n} items on {dev}: {(c.stdout + c.stderr).strip()}"); fail = 1
            row[dev] = t
        rows.append(row)
    if table and rows:
        cols = ["xetal", "cpu"] + devs
        print("\n| items | " + " | ".join(f"{c} (s)" for c in cols) + " |")
        print("| ----- | " + " | ".join("-" * (len(c) + 4) for c in cols) + " |")
        for row in rows:
            print(f"| {row['n']} | " + " | ".join(f"{row.get(c, float('nan')):.2f}" for c in cols) + " |")
    print(f"check-sizes: {len(sizes)} size{'s' if len(sizes) != 1 else ''}" + (", all agree with the evaluator" if not fail else ", FAILURES"))
    sys.exit(fail)


if __name__ == "__main__":
    main()
