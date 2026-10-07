#!/usr/bin/env python3
"""The reductions, the pipeline and matrix products at large sizes:
for each size N, deterministic data is generated into work/sizes/
(Ints in -1000..1000 and Floats in [0, 1)), an X_eTaL program on the
Accel library reads it and prints the sum, the largest item, an inner
product and the pipeline (the pinned evaluator's answer), and the
same computation as an XIR program with inputs bound to the same
files runs on the reference interpreter (exactly) and on every
OpenCL device found (Floats within 1e-5). Each run is timed.

A size mN is an N by N matrix product instead (Ints in -9..9, so the
product is exact, and Floats in [0, 1)): its sum and largest item
from the evaluator, the interpreter and every device, on the device
both untiled and in 16 by 16 tiles.

  scripts/check-sizes.py [N|mN...]        # default 1024 65536 1048576 m64 m256
  scripts/check-sizes.py --table N|mN...  # also print a markdown table of the timings
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


MXTL = '''"ac:" u_se< "Accel"
a := {n} {n} r_eshape f_loor n_umbers []N_GET "{d}/m{n}-a.txt"
b := {n} {n} r_eshape f_loor n_umbers []N_GET "{d}/m{n}-b.txt"
c := a ac:m_atmul b
'+ r_/_12 c
'm_ax r_/_12 c
f := {n} {n} r_eshape n_umbers []N_GET "{d}/m{n}-f.txt"
g := {n} {n} r_eshape n_umbers []N_GET "{d}/m{n}-g.txt"
h := f ac:m_atmul g
'+ r_/_12 h
'm_ax r_/_12 h
'''

MXIR = '''%a = input i64 [{n} {n}]
%b = input i64 [{n} {n}]
%c = matmul %a %b
%cs = reduce add %c
%cm = reduce max %c
%f = input f64 [{n} {n}]
%g = input f64 [{n} {n}]
%h = matmul %f %g
%hs = reduce add %h
%hm = reduce max %h
output %cs
output %cm
output %hs
output %hm
'''


def generate_matrix(n):
    WORK.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1000 + n)
    for tag, make in (("a", lambda: rng.randint(-9, 9)), ("b", lambda: rng.randint(-9, 9)),
                      ("f", lambda: round(rng.random(), 6)), ("g", lambda: round(rng.random(), 6))):
        p = WORK / f"m{n}-{tag}.txt"
        if not p.exists():
            p.write_text("\n".join(str(make()) for _ in range(n * n)) + "\n")
    rel = os.path.relpath(WORK, ROOT)
    (WORK / f"m{n}.xtl").write_text(MXTL.format(d=rel, n=n))
    (WORK / f"m{n}.xir").write_text(MXIR.format(n=n))


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
    sizes = args or ["1024", "65536", "1048576", "m64", "m256"]
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "xetal-gpu-cli"], cwd=ROOT / "components", check=True)
    devs = devices()
    if not devs:
        print("check-sizes: no OpenCL device here; the GPU side is skipped")
    rows = []
    fail = 0
    for size in sizes:
        matrix = size.startswith("m")
        n = int(size[1:] if matrix else size)
        name = f"m{n}" if matrix else f"{n}"
        what = f"{n} by {n} product" if matrix else f"{n} items"
        if matrix:
            generate_matrix(n)
            tags = ("a", "b", "f", "g")
        else:
            generate(n)
            tags = ("x", "w", "f")
        want = WORK / f"{name}.out"
        r, t_eval = timed([XT, "run", os.path.relpath(WORK / f"{name}.xtl", ROOT)])
        if r.returncode != 0:
            print(f"FAIL: xetal on {what}: {r.stderr.strip()}"); fail = 1; continue
        want.write_text(r.stdout)
        binds = []
        for tag in tags:
            binds += ["--bind", f"{tag}=@{WORK / f'{name}-{tag}.txt'}"]
        row = {"n": what, "xetal": t_eval}
        runs = [("cpu", [], "0")] + [(d, [], "1e-5") for d in devs] + [(d + " tile 16", ["--tile", "16"], "1e-5") for d in devs if matrix]
        for label, extra, tol in runs:
            dev = label.split()[0]
            r, t = timed([GPU, "run", WORK / f"{name}.xir", "--device", dev] + extra + binds)
            if r.returncode != 0:
                print(f"FAIL: xetal-gpu on {label}, {what}: {r.stderr.strip()}"); fail = 1; continue
            got = WORK / f"{name}-{label.replace(':', '').replace(' ', '-')}.out"
            got.write_text(r.stdout)
            c = subprocess.run([COMPARE, want, got, tol], capture_output=True, text=True)
            if c.returncode == 0:
                print(f"ok: {what} on {label}, {c.stdout.strip()} ({t:.2f} s; xetal {t_eval:.2f} s)")
            else:
                print(f"FAIL: {what} on {label}: {(c.stdout + c.stderr).strip()}"); fail = 1
            row[label] = t
        rows.append(row)
    if table and rows:
        cols = ["xetal", "cpu"] + devs + [d + " tile 16" for d in devs]
        print("\n| work | " + " | ".join(f"{c} (s)" for c in cols) + " |")
        print("| ---- | " + " | ".join("-" * (len(c) + 4) for c in cols) + " |")
        for row in rows:
            print(f"| {row['n']} | " + " | ".join(f"{row[c]:.2f}" if c in row else "--" for c in cols) + " |")
    print(f"check-sizes: {len(sizes)} size{'s' if len(sizes) != 1 else ''}" + (", all agree with the evaluator" if not fail else ", FAILURES"))
    sys.exit(fail)


if __name__ == "__main__":
    main()
