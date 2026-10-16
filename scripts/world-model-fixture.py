#!/usr/bin/env python3
"""The world model as one fixed XIR program with its batch as inputs:
the fixture the session and binary-binding benchmarks measure on
(scripts/bench-session.py, scripts/bench-binary.py, the Arch lane's),
where what is timed is binding, uploading and reusing a session, not
the model. The program is written here from the model's definition
(432 pixels in, 64 ReLU, 144 sigmoid out) and its inputs are made by
the pinned xetal from the model's own core; the model itself runs from
its source in scripts/ml-bench.py (components/lower), and this twin is
not an example program (plan A4b): it is a benchmark fixture, kept so
the transfer measurements stay on one program whose inputs are known.

  prepare()   # work/ml-bench/world-model.xir and its inputs, if missing
"""
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "work" / "xetal-ml"
WORK = ROOT / "work" / "ml-bench"
XETAL = ROOT / "bin" / "xetal"
GPU = ROOT / "target" / "release" / "xetal-gpu"
XX = ROOT / "scripts" / "xx"


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


def device_names():
    r, _ = run([GPU, "devices"])
    names = {}
    for line in r.stdout.splitlines():
        if line.startswith("opencl:"):
            dev, _, rest = line.partition(" ")
            names[dev] = rest.split(" (")[0].strip()
    return names


def cycle(values, n):
    return [values[i % len(values)] for i in range(n)]


def offsets(n):
    return " ".join(str(i) for i in range(n))


def world_model_twin(batch=1936, held=387):
    """The world model's batch: a dense layer with its bias row (NN's
    d_ense: matmul, then the bias spread down the batch by a table),
    ReLU (max with 0), a second dense layer, the sigmoid
    1 / (1 + exp(-z)), then u:e_xact: floor of the sum over rows of the
    row minimum of (P > 0.5) = (T > 0.5), for all 1,936 states and for
    the 387 held out."""
    out = [
        f"%X = input f64 [{batch} 432]", f"%T = input f64 [{batch} 144]",
        f"%Xh = input f64 [{held} 432]", f"%Th = input f64 [{held} 144]",
        "%w1 = input f64 [432 64]", "%b1 = input f64 [64]",
        "%w2 = input f64 [64 144]", "%b2 = input f64 [144]",
        f"%o = const i64 [{batch}] " + offsets(batch), f"%oh = const i64 [{held}] " + offsets(held),
        "%half = const f64 [] 0.5", "%zero = const f64 [] 0.0", "%one = const f64 [] 1.0",
    ]
    for tag, x, t, o in (("", "%X", "%T", "%o"), ("h", "%Xh", "%Th", "%oh")):
        out += [
            f"%bias1{tag} = table right {o} %b1", f"%z1{tag} = matmul {x} %w1", f"%a1{tag} = map add %z1{tag} %bias1{tag}",
            f"%hid{tag} = map max %a1{tag} %zero",
            f"%bias2{tag} = table right {o} %b2", f"%z2{tag} = matmul %hid{tag} %w2", f"%a2{tag} = map add %z2{tag} %bias2{tag}",
            f"%nz{tag} = map neg %a2{tag}", f"%e{tag} = map exp %nz{tag}", f"%d{tag} = map add %one %e{tag}", f"%p{tag} = map div %one %d{tag}",
            f"%pm{tag} = map gt %p{tag} %half", f"%tm{tag} = map gt {t} %half", f"%eq{tag} = map eq %pm{tag} %tm{tag}",
            f"%eqf{tag} = cast f64 %eq{tag}", f"%rowmin{tag} = reduce min axis=2 %eqf{tag}", f"%sum{tag} = reduce add %rowmin{tag}",
            f"%n{tag} = cast i64 %sum{tag}",
        ]
    out += ["output %n", "output %nh"]
    return "\n".join(out) + "\n"


WORLD_PREPARE = '''
i := o_ffsets 1936
all := o_\\ 4 1936 r_eshape (i d_iv 176) c_at ((i d_iv 16) m_od 11) c_at ((1 + (i d_iv 4) m_od 4) s_elect -2 -1 1 2) c_at (1 + i m_od 4) s_elect -2 -1 1 2
held := 387 4 r_eshape f_loor n_umbers []N_GET "data/held-out.txt"
(X, T) := u:w_indow all
(Xh, Th) := u:w_indow held
r_avel X
r_avel T
r_avel Xh
r_avel Th
'''

MODELS = {
    "world-model": {
        "what": "432 pixels in, 64 ReLU, 144 sigmoid out; all 1,936 states as one batch, and the 387 held out: next frames predicted exactly",
        "twin": world_model_twin,
        "prepare": WORLD_PREPARE,            # appended to the program's core, run in the demo's directory
        "generated": ["X", "T", "Xh", "Th"],  # the lines it prints, one input each
        "data": {"w1": "w1.txt", "b1": "b1.txt", "w2": "w2.txt", "b2": "b2.txt"},
        "shapes": {"X": "1936 432", "T": "1936 144", "Xh": "387 432", "Th": "387 144", "w1": "432 64", "b1": "64", "w2": "64 144", "b2": "144"},
        "expected_lines": (2, 3),            # of its reg-rs baseline, and of a run
        "libs": ["NN"],
    },
}


def facade_driver(name, b, outputs=1):
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
    read = "gp:o_utputInts" if outputs > 1 else "gp:o_utput"
    lines.append(f"tout := '{{ i -> {read} i }} ck:t_ime 1")
    for i in range(1, outputs + 1):
        lines.append(f"{read} {i}")
    lines.append("tload"); lines.append("tbind"); lines.append("trun"); lines.append("tout")
    return "\n".join(lines) + "\n"


def prepare():
    """The fixture's files under work/ml-bench/, written if missing: the
    XIR program, the inputs the model's own core generates (run by the
    pinned xetal in the demo's directory) and its weights from data/.
    Returns the expected counts, from the demo's baseline."""
    subprocess.run([ROOT / "scripts" / "xetal-ml.sh"], check=True, capture_output=True)
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "xetal-gpu-cli"], cwd=ROOT / "components", check=True)
    WORK.mkdir(parents=True, exist_ok=True)
    name, m = "world-model", MODELS["world-model"]
    demo = ML / "demos" / name
    expected = (demo / "reg" / f"cli-{name}.out").read_text().splitlines()[1:3]
    if not all((WORK / f"{name}-{tag}.txt").exists() for tag in m["shapes"]):
        libs = ":".join(str(ML / "libs" / l / "src") for l in m["libs"])
        env = dict(os.environ, XETAL_PATH=libs)
        core = (demo / f"{name}.xtl").read_text().split("# -- end of the core")[0]
        prep = WORK / f"{name}-prepare.xtl"
        prep.write_text(core + m["prepare"])
        r, _ = run([XETAL, "run", prep], cwd=demo, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"preparing {name}'s inputs: {r.stderr.strip()[:300]}")
        lines = r.stdout.splitlines()
        for tag, line in zip(m["generated"], lines[-len(m["generated"]):]):
            (WORK / f"{name}-{tag}.txt").write_text(line + "\n")
        for tag, f in m["data"].items():
            (WORK / f"{name}-{tag}.txt").write_text((demo / "data" / f).read_text())
    (WORK / f"{name}.xir").write_text(m["twin"]())
    return expected


if __name__ == "__main__":
    print("expected counts:", " and ".join(prepare()))
