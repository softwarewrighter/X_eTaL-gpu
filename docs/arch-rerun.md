# What to re-run on the Arch machine

For the Arch lane (the RTX 5060 Ti, 16 GB), after pulling `main` at
or past the commit that added this file (2026-10-16): the Mac's
changes since the Linux source-demo run of 2026-10-14 moved the
benchmarks to X_eTaL-ML's own source, changed the session and
binary-binding benchmarks' fixture, and added the Mojo backend. Each
item says what to run, what it writes, and what to look at. Results
belong to the Arch lane: write them to the RTX files named here, not
to the Mac's (`docs/ml-results.md`, `docs/m1max-*.md`).

## 1. The pins and the gate

```sh
git pull
just xetal && just xetal-x && just gpu-ext && just xetal-ml
just gate
```

The gate now runs `scripts/ml-bench.py --check` over eight programs
from X_eTaL-ML's own source (four benchmarks, the world model, the
digit reader's batch, the denoiser, the tiny GPT's batch), which
lowers each program once (the world model and the denoiser take 30
to 40 s each the first time, then `work/lower-cache` holds them),
and `scripts/mojo.sh --check`, which only reports whether the Mojo
toolchain is installed.

## 2. The models from source on the RTX 5060 Ti

```sh
just ml-bench --require-device --output docs/rtx5060-ml-results.md
```

Rewrites `docs/rtx5060-ml-results.md` in the new form: every program
from source (the `host` column counts the statements the evaluator
computed), the cold lowering time, the device's kernels and launches,
and the macro route (`ml/*.xtl` under `xetal-x`, first and second
call) on `opencl:0`. The denoiser's first output line is compared on
the interpreter only (its own 1e-6 self-check fails in f32; asked of
X_eTaL-ML). Expected to agree: the world model's 1897 and 348, the
digit reader's 971, the GPT batch's 2.2496 and 2098 7000. Compare the
device columns with the 2026-10-12 report's, and say in the commit
which numbers moved.

## 3. The session and binary-binding benchmarks

```sh
just binary-bench --require-device
just session-bench --require-device
```

Both now take their fixed world-model program from
`scripts/world-model-fixture.py` (the same twin as before, kept as a
benchmark fixture; it is prepared from the pinned model if missing),
no longer from `ml-bench --check`. They should give the same numbers
as before within noise; `docs/rtx5060-binary-results.md` and
`docs/rtx5060-session-results.md` are theirs.

## 4. The Mojo backend on the RTX 5060 Ti

```sh
just mojo              # MAX 26.7.0.dev2026101305 into work/mojo/.venv, bin/mojo
just mojo --check
```

Needs `uv` and an NVIDIA driver 580 or newer; with an older driver,
set `MODULAR_NVPTX_COMPILER_PATH` to a CUDA toolkit's `ptxas` as
rust-demo's `doc/porting-arch-rtx5060.md` describes (the same machine
ran rust-demo's five programs on 2026-10-16, so the setup is known
to work there). Then:

```sh
cd components && cargo test -p xetal-gpu-mojo -- --nocapture; cd ..
for d in saxpy threshold reduce-sum pipeline vector-add; do
  just run-mojo libs/Accel/demos/$d.xtl
done
```

The test prints `ok: <demo> on the GPU through Mojo` for each of the
five demos the backend takes (the others are refused by name; the
list is `components/mojo/tests/expected/refused.txt`); `run-mojo`
prints each demo's output and a line with the device (`api=cuda;
name=NVIDIA GeForce RTX 5060 Ti`), the build time (about 4.5 s on
that card in rust-demo, zero once cached) and the run time. Record
the five outputs and times in a new `docs/rtx5060-mojo-results.md`
(the device line, the Mojo version from `just mojo --check`, the
driver, one table of demo, build, run), add it to the gate's markdown
list in `scripts/gate.sh`, and say in the README's results-by-GPU
section that the Mojo numbers on the RTX go through the CUDA driver.

## 5. The NVIDIA OpenCL program binary (the blog agent's ask A7)

After pulling the 2026-10-21 commit that added `xetal-gpu binary`:

```sh
just build
just opencl-evidence docs/evidence/rtx5060-opencl-$(date +%Y%m%d)
```

The script lowers the threshold demo, writes its OpenCL C and plan,
runs it on `opencl:0` against the evaluator's baseline, and asks the
driver for `CL_PROGRAM_BINARIES`. NVIDIA's OpenCL returns PTX text,
which the script recognizes and saves as `threshold.ptx` beside the
raw bytes and its decoding. Commit the directory and a short
`docs/rtx5060-opencl-handoff.md` in the form of
`docs/m1max-opencl-handoff.md` (what the post can say, the evidence
table, what is and is not established: here the driver's PTX for the
same program the Mojo and nvcc captures used), then tell the blog
agent (session `blog-b4`) the paths. If `opencl:0` is not the RTX
5060 Ti on that machine, pass `--device opencl:N` after the
directory.

Completed 2026-10-21 on the RTX 5060 Ti: see
[the handoff](rtx5060-opencl-handoff.md) and its linked evidence.

## 6. The Mojo backend's structural operations (2026-10-26)

After pulling the 2026-10-26 commit (Mojo backend, step 3):

```sh
cd components && cargo test -p xetal-gpu-mojo -- --nocapture; cd ..
for d in lookup windows; do just run-mojo libs/Accel/demos/$d.xtl; done
just run-mojo libs/Accel/demos/matmul.xtl --tile 16
```

The demos test should print `ok:` for all thirteen demos, and the
semantics test `ok: concat`, `topk`, `reshape`, `rotate`, `lookup`,
`products`, `products_t4` and the stray-position refusal. Add lookup
and windows to `docs/rtx5060-mojo-results.md` with their times, note
the tiled product, and say in the README's Mojo paragraph that all
thirteen agree on the RTX through the CUDA driver.

## 7. Push

Commit with a `CHANGES.md` line per commit, as before, and push to
`main` (rebase first; the Mac lane commits there too).
