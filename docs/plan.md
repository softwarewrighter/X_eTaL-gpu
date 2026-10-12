# X_eTaL-gpu -- Implementation Plan

X_eTaL programs on GPUs: the GPU compiler, runtime and scheduling for
X_eTaL (the eXperimental Extensible Typed Array Language, developed in
`../X_eTaL`), aimed at the user's own cards, old and new (the
hardware table below: Quadro K4200, Tesla M10 and P4, GTX 1660 Super, RTX 3060 and RTX 3060 Laptop, RTX 3090,
RTX 4070 Laptop, RTX 5060 Ti, A2, and the Apple M1 Max and M3 Pro), by way of
OpenCL C 1.2. The goal is to demonstrate the concept, that one
typed array program runs unchanged as sequential software, as
massively parallel software, and (in the sibling project) as spatial
hardware, not to be fast.

## Hardware

The cards this work runs on are the user's (2026-10-12); this table
is the one place that names them, and the README, the component
pages and the schedule templates follow it. `docs/research7.txt`
named other cards (K80, M40, P40, P100, RTX 2060) as examples; they
are not the user's and are not targets.

| Card | Architecture | Memory | Role here | Schedule |
| ---- | ------------ | ------ | --------- | -------- |
| Apple M1 Max (this Mac) | Apple GPU, OpenCL 1.2 | shared | development; every check so far; no fp64 | `apple-m1-max.toml` (run) |
| Apple M3 Pro | Apple GPU, Apple's OpenCL | shared | a second Apple-silicon target; its OpenCL version, fp64 and work-group limit to be recorded when first run | `apple-m1-max.toml` until run (a file of its own if it differs) |
| Quadro K4200 | Kepler | 4 GB | the oldest generation here and the smallest memory: the case where weights stay on the host (a schedule field, saga gpu-models); the user has several | `nvidia-legacy.toml` (template) |
| Tesla M10 | Maxwell | four GPUs of 8 GB | an old card the newest CUDA toolkits retire; each GPU an OpenCL device | `nvidia-legacy.toml` (template) |
| Tesla P4 | Pascal | 8 GB | an old card the newest CUDA toolkits retire | `nvidia-legacy.toml` (template) |
| GTX 1660 Super | Turing | 6 GB | with the RTX 3060 Laptop, the smallest memory of the current cards | `nvidia-modern.toml` (template) |
| RTX 3060 | Ampere | 12 GB | current PCIe cards; the user has several | `nvidia-modern.toml` (template) |
| RTX 3060 Laptop | Ampere | 6 GB (NVIDIA's specification for the laptop part; to be confirmed on the machine) | a current card, in a System76 laptop | `nvidia-modern.toml` (template) |
| RTX 3090 | Ampere | 24 GB | a current card; the most memory | `nvidia-modern.toml` (template) |
| RTX 4070 Laptop | Ada Lovelace | 8 GB | a current card, in an Asus ROG laptop | `nvidia-modern.toml` (template) |
| RTX 5060 Ti | Blackwell | 16 GB | Arch Linux, large12; NVIDIA 615.78.08, OpenCL 3.0, 36 compute units, work-groups to 1024, fp64 | `nvidia-rtx5060-ti.toml` and `nvidia-rtx5060-ti-f64.toml` (run) |
| A2 | Ampere | 16 GB | a low-power data-center card | `nvidia-modern.toml` (template) |

The user has other cards too; one is added here when it is used.
The Arch machine `large12` holds the RTX 5060 Ti; measurements are
in `docs/rtx5060-results.md` and `docs/rtx5060-ml-results.md`.

Source: `docs/research7.txt`, the user's planning conversation
(archival, NOT normative), which places the work in three
repositories with a strict division of responsibility:

| Repo | Owns |
| ---- | ---- |
| `softwarewrighter/X_eTaL` | the language, the typed Core IR, the accelerator IR and the lowering to it ("here is the computation") |
| **`softwarewrighter/X_eTaL-gpu`** (here) | **scheduling, OpenCL kernels, the host runtime: how an existing GPU realizes that computation** |
| `hardwarewrighter/X_eTaL-fpga` | scheduling, Verilog, synthesis: how new hardware realizes it |

The X_eTaL part (the IR and the lowering) is another agent's work in
`../X_eTaL` and, as of 2026-10-06, not yet planned there; the FPGA
part is a different agent's in a different repository. This
repository changes only itself. What it needs from X_eTaL is in
`docs/xetal-asks.md`, whose first entries (G1 to G4) are the blockers.

Development is driven by agentrail sagas (one active saga in
`.agentrail/`, finished sagas archived to `.agentrail-archive/`), as in
the sibling repos. Every step ends with the gate (`just gate`), docs
updated (`README.md`, `CHANGES.md`, this plan, `docs/xetal-asks.md`),
a sane `.gitignore`, a detailed commit to `main` (with the
`.agentrail/` changes), `agentrail complete`, and a push.

## Guiding principle

> X_eTaL meaning is not the execution device.

An X_eTaL array expression (`c := a + b`, `'+ r_/ c`) describes a
computation; the evaluator in X_eTaL is its golden reference. This
repository takes the same computation to a GPU and proves, for every
example, that the GPU gives the same answer:

```
                X_eTaL program (.xtl)
                        |
          +-------------+-------------+
          v             v             v
      evaluator    XIR interpreter   OpenCL on a GPU
          |             |             |
          +-------------+-------------+
                        v
                  the same arrays
```

The interesting part is not the kernel generator but the layer
between the IR and the kernels, the **schedule**: the same `matmul`
can be sequential loops, work-groups and work-items with a tile size,
or (over in X_eTaL-fpga) a handful of multiply-accumulate lanes. That
separation of what is computed from how this machine computes it is
in the spirit of an array language, and it is what this repository
is for.

## Architecture decisions

| # | Decision | Why |
| - | -------- | --- |
| A1 | X_eTaL is **pinned, not tracked** (`../X_eTaL/docs/vendoring.md`, as X_eTaL-demos and X_eTaL-ML do it): `XETAL_COMMIT` holds the known-good commit; `just xetal` (scripts/xetal.sh) clones X_eTaL into the gitignored `work/xetal/`, checks that commit out, builds the CLI into `target/xetal/` and links `bin/xetal`; `just xetal-pin [REF]` moves the pin to a committed ref of `../X_eTaL`, at a saga start or when an ask has landed, never mid-step, in its own commit after the baselines pass. | X_eTaL moves fast; this repo needs a recent but stable evaluator as its reference, changed deliberately, without carrying a copy of its source. |
| A2 | **The evaluator is the specification.** Every example exists first as an ordinary `.xtl` program (or a function of an `.xtl` library) that the pinned `xetal` runs; its printed output is a reg-rs baseline; the GPU path must reproduce it (Ints exactly, Floats within a tolerance, since the device computes in f32 unless told otherwise). | research7: "X_eTaL becomes the specification"; "the existing evaluator is the golden reference implementation". |
| A3 | **The acceleratable subset is a library**, `libs/Accel/` (alias `ac:`), in X_eTaL-libraries' layout (`src/Accel.xtl`, `tests/` with reg-rs baselines and pinned export types, `demos/*.xtl`, `docs/README.md`), written in the subset research7 says to accept first: static scalar types and ranks, pure functions, arithmetic, comparisons, scalar extension, elementwise operations, reductions, then inner products. Its demos are the example programs (vector add, saxpy, threshold, reduce, the acceptance pipeline multiply, add, select, reduce). | What the GPU runs must be ordinary, typed, tested X_eTaL; a library is the form X_eTaL has for that, and the same files can move to X_eTaL-libraries or X_eTaL-ML whole. |
| A4 | **A provisional IR, to be replaced.** The accelerator IR and the lowering from Core belong to X_eTaL (asks G1, G2) and do not exist yet. This repo carries `components/xir` (crate `xetal-gpu-xir`): the IR of research7 (Input, Const, elementwise ops, Map, Reduce, later Dot and Matmul), typed values with static shapes, a textual form that round-trips, and a reference interpreter. Each example's `.xtl` has a hand-lowered `.xir` twin, and `scripts/check-equiv.sh` proves the twin's interpretation equals the evaluator's output. No bridge from `xetal core`'s dump is built (ask G4: it is an undocumented internal form). When X_eTaL's IR lands, `components/xir` becomes an adapter and the twins are deleted. | The GPU side (schedule, kernels, runtime) can be built and tested now, against the real evaluator, without inventing language semantics here or duplicating X_eTaL's compiler. |
| A5 | **OpenCL C 1.2 first; CUDA is a later backend.** The emitter (`components/opencl`) writes plain OpenCL C 1.2 kernels as text (testable without a GPU: the kernel text is pinned); the runtime (`components/runtime`) is a thin host on the `opencl3` crate: buffers in, kernel, buffers out. No vendor extensions beyond `cl_khr_fp64` when f64 is asked for. | research7 sections 7 and 8: OpenCL reaches every generation in the hardware table, Kepler to Blackwell, with one toolchain; CUDA would make this repo know NVIDIA's toolkit matrix. Apple's OpenCL (1.2, deprecated but present on this Mac) is enough for development; the old NVIDIA cards are the target. |
| A6 | **Schedule is separate from semantics.** An XIR program is scheduled before it is emitted: which values become one fused elementwise kernel, how a reduction is split (work-group partials, then a second pass), the element type on the device (f32 by default for Float, i64 for Int), work-group size. The schedule is data (a Rust struct, later a TOML file per device), never written into the `.xtl` or the IR. | research7 "How scheduling should work": the model does not change, only the schedule; it is also the FPGA repo's natural counterpart. |
| A7 | **One tool, `xetal-gpu`** (`components/cli`): `devices` (the OpenCL platforms and devices found), `check FILE.xir` (parse, type, shapes), `run FILE.xir [--device cpu or opencl:N]` (interpret, or run on a device; prints as `xetal` prints), `kernel FILE.xir` (the OpenCL C it would run), `explain FILE.xir` (the schedule: which operations became which kernels, how many work-items). Research7 section 10's `xetal run --device opencl:0 demo.xtl` waits on asks G2 and G6. | One command per question; the explain command is the seed of the "visual hardware explanation" milestone (H3). |
| A8 | **Rust components in one workspace**, `components/` (`xir`, `opencl`, `runtime`, `cli`; crates `xetal-gpu-*`), each small, with its README, unit tests, `cargo fmt` and `clippy -D warnings` in the gate; `.cargo/config.toml` points every workspace at one `target/`. GPU tests run when a device is present and are skipped, loudly, when none is (CI without a GPU). | The same component discipline as X_eTaL; a test suite that is honest about hardware. |
| A9 | **Process as the siblings'**: `just` is the entry point (recipes call `scripts/*.sh`); `CHANGES.md` gets a line per commit; American spellings only (`scripts/check-spelling.py` in the gate); docs are ASCII-only markdown (`sw-markdown-checker`); a missing X_eTaL feature or bug goes in `docs/xetal-asks.md` and is not fixed or hidden here; work goes to `main` and is pushed; `.xtlm` macro libraries are not needed by this plan (device selection is a command-line concern, research7 section 11), so nothing here waits on them. | Same process, same reviewers, same tools. |

## The subset, in X_eTaL and in XIR

| X_eTaL | XIR | GPU (OpenCL) | Saga |
| ------ | --- | ------------ | ---- |
| `a + b`, `-`, `*`, `/`, `m_ax`, `m_in`, `n_eg`, `a_bs` on arrays, with scalar extension | `map add %a %b` (a rank-0 operand broadcasts) | one work-item per element; a chain of maps is one fused kernel | 1 |
| `a > b`, `=`, `<`, `&`, `\|`, `n_ot`, masks in arithmetic | `map gt`, `map and`, ... (Bool as i32 0 or 1) | the same kernel | 1 |
| `(m * x) + (n_ot m) * y` (a select by mask) | `select %m %x %y` | `m ? x : y` | 1 |
| `'+ r_/ v`, `'m_ax r_/ v` (a vector to a scalar) | `reduce add %v` | work-group partial sums in local memory, then a second pass | 1 |
| `e_xp`, `l_og`, `f_loat` | `map exp`, `map log`, `cast f64 %a` | `exp`, `log`, a cast | 1 |
| `'+ r_/_2 m`, `'+ r_/ m` (a matrix along an axis) | `reduce add axis=2 %m` | one work-item per row (or column) | 2 |
| `x '+ '* i_nner w` (matrix-vector, matrix-matrix) | `matmul %x %w` | one work-item per output element, then tiled | 2 |
| `nn:s_oftmax`, `nn:d_ense`, `nn:r_elu` (X_eTaL-ML's NN) | compositions of the above | the kernels above, scheduled | 3 |
| `o_\ m`, `p t_ranspose m`, `o_\_23 t` | `transpose %m`, `transpose 3 1 2 %m`, `transpose 1 3 2 %t` | a gather, one work-item per item; the last two axes through local memory in tiles when the schedule tiles; a vector is no launch | 4 |

## Roadmap

Implementable work first (the GPU side, against hand-lowered
examples); what needs an X_eTaL ask (the lowering, the one-command
demo) is in the last saga and moves up when the ask lands.

The order after saga 4 (the user, 2026-10-12, from the question
"what useful work can we show?"): the operations a transformer needs
(transpose, windows, lookup, top-k; saga 4), then weights kept on the
device and binary arrays (saga 5, without which every model's time
is setup), a tiny GPT's batched inference with the Jev-like
classifier as its head (saga 6), a training step with an explicit
backward pass, from scratch and then fine-tuning a small pretrained
model (saga 7), a conditional memory, Engram (saga 8), a mixture of
experts (saga 9); each card as it is at hand (saga 10) and the
lowering when X_eTaL has it (saga 11). The rule for what to demo:
the batch is the array and the work is products, since products win
on every card measured (31 and 25 times, through the bridge) and
elementwise work through the bridge loses; a tiny model on one
prompt never looks good.

## Asks from other repositories

What the sibling repositories ask of this one, and where each is
answered. X_eTaL-ML filed GA1 to GA6 on 2026-10-12 (its
`docs/gpu-asks.md`): its models are to be tested, measured and shown
on real GPUs through this repository; it writes no GPU code itself,
and gives each model as a batch X_eTaL program, its weights in
`data/` and its expected outputs.

| # | From | Ask | Status here | Answered by |
| - | ---- | --- | ----------- | ----------- |
| GA1 | X_eTaL-ML | Pin an X_eTaL-ML commit and run its batch models (the program, the weights, the expected outputs); a results table per model and device, the device's time and the bridge's apart | in progress: benchmarks and world model agree on the M1 Max and RTX 5060 Ti; ML pin fac586c includes the committed model and the gate checks it; the denoiser waits on its batch run there | saga gpu-models: ml-bench, world-model, denoiser; tables in `docs/ml-results.md` and `docs/rtx5060-ml-results.md` |
| GA2 | X_eTaL-ML | `transpose` in the IR, for attention scores (`Q K^T`) and every backward pass | done (2026-10-12) | saga gpu-models: transpose (0h): `transpose` in the IR, a gather or tiles; Accel `s_cores` and `w_eightGrad` are the two uses, checked on the devices |
| GA3 | X_eTaL-ML | The convolution's windows in the IR: reshape and rotate along an axis, or one windows operation | taken | saga gpu-models: windows (reshape and rotate first, as general X_eTaL primitives; one windows operation only if measurement asks for it) |
| GA4 | X_eTaL-ML | A newer X_eTaL inside `xetal-x` (its libraries use `h:` and tuples) | ours already, asked of X_eTaL-extensions | `docs/tuple-plan.md`, X-1 |
| GA5 | X_eTaL-ML | Arrays across the bridge in binary (the world model's batch is about 836,000 inputs, a third of a second as text) | ours already, asked of X_eTaL-extensions and X_eTaL | `docs/xetal-asks.md`, G5 and G8; meanwhile the bridge's time is reported apart from the device's |
| GA6 | X_eTaL-ML | The accelerator IR and the lowering, so no model needs a hand-written twin | ours already, asked of X_eTaL | `docs/xetal-asks.md`, G1 to G3 |

The world model is the "complete tiny trained network" that step
tiny-nn asked for, so tiny-nn is answered by step world-model.

### Saga 1 -- gpu-foundation  [DONE]

Goal: the process, the pinned X_eTaL, the acceleratable subset as a
tested library, the provisional IR with an interpreter, OpenCL
kernels for elementwise operations and reductions, a host runtime,
and the first proof: one X_eTaL program's result reproduced on a GPU.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | scaffold | DONE (2026-10-06): as planned. Planned: agentrail saga, CLAUDE.md/AGENTS.md, README, COPYRIGHT, LICENSE, `.gitignore`, justfile, `scripts/gate.sh`, the pinned X_eTaL and `scripts/xetal.sh`, the library tooling, this plan, `docs/xetal-asks.md` (the blockers), `CHANGES.md` |
| 2 | accel-lib | DONE: nine exports (`t_hreshold` is Float-only: a Bool mask converts only to the type of the literal it meets; three-argument functions are not applicable, so every export takes one or two); basics and checks tests; the five demos with baselines; the reference page. Planned: `libs/Accel`: the acceleratable subset as exported functions (`ac:v_add`, `ac:s_axpy`, `ac:t_hreshold`, `ac:s_um`, `ac:d_ot`, `ac:r_elu`, `ac:p_ipeline` ...), tests, pinned types, the example programs as its demos (vector-add, saxpy, threshold, reduce-sum, pipeline), reg-rs baselines, the reference page |
| 3 | xir | DONE: `xetal-gpu-xir` (ir, text, check, interp, format; 17 tests), the five twins, `scripts/compare-out.py` and `check-equiv.sh` (exact on cpu; a demo without a twin fails), `xetal-gpu check`, `run --device cpu`, `print`; found that X_eTaL prints Floats in full decimal (no exponent) and folds a reduce from the right, both matched. Planned: `components/xir`: the IR (types, shapes, ops), its text form (parse and print, round-trip tested), shape and type checking, the reference interpreter (i64 and f64, X_eTaL's semantics); each demo's `.xir` twin; `scripts/check-equiv.sh` (the twin's interpretation against the baseline, Ints exact, Floats within tolerance); `xetal-gpu check` and `run --device cpu` |
| 4 | opencl-emit | DONE: `xetal-gpu-opencl` (Schedule: float/int widths, work-group; Plan: source, buffers, launches, outputs; fusion of consecutive elementwise values of one shape, only needed values written, single-value constants as literals; a tree reduction kernel per op and type, launched on its partials until one is left, two partial buffers alternating; floored idiv/mod helpers), `explain`, `xetal-gpu kernel` and `explain` with `--float --int --work-group`; 7 unit tests and every twin's `.cl` and `.plan` pinned. Planned: `components/opencl`: the schedule (fused elementwise kernels, a reduction in two passes, device element types) and the OpenCL C 1.2 emitter; the kernel text of every twin pinned as a test; `xetal-gpu kernel` and `explain` |
| 5 | opencl-run | DONE: `xetal-gpu-runtime` on opencl3 (dynamic loading, so the tool runs without OpenCL and reports no device); devices listed across platforms; execute: device checks (fp64, work-group), build with the device's log as the error, typed buffers uploaded in the schedule's widths, launches on an in-order queue, outputs widened back; 4 device tests (skipped without one); `xetal-gpu devices`, `run --device opencl:N`; check-equiv runs every twin on every device within 1e-5; all five agree on the Apple M1 Max (OpenCL 1.2, no fp64), including a 70000-item three-pass reduction in the tests. Planned: `components/runtime`: devices, buffers, compile, run, read back (`opencl3`); `xetal-gpu devices` and `run --device opencl:N`; the equivalence script runs the GPU side too when a device is present (this Mac's Apple OpenCL device); every demo reproduced on the GPU, Ints exact |
| 6 | reduce-gpu | DONE: `scripts/check-sizes.py` (`just sizes`; the gate runs 1024 and 65536): generated data, the evaluator's answer on the same files, the XIR program with bound inputs on the interpreter (exactly) and on the GPU (within 1e-5), timed; 1024, 65536 and 1048576 items all agree (a million-item reduce is three launches); timings in the runtime README. Planned: the reduction on the device (work-group partials in local memory, second pass), the pipeline example end to end on the GPU; sizes beyond one work-group (1024, 1 M elements) checked; timings noted (not a goal) |

### Saga 1 retrospective

Delivered in one day: the process, the pinned evaluator as the
specification, the Accel library with five example programs and
their baselines, a provisional IR (text form, checker, interpreter
with the evaluator's exact semantics), a schedule and an OpenCL C
emitter (fused elementwise kernels, tree reductions over any size),
a runtime on opencl3, and the proof: every example and every size
up to a million items agrees with the evaluator on this Mac's GPU,
Ints exactly. Learned: (1) the hand-lowered twin is a workable
stand-in for the lowering, but every twin is a transcription a
reader must trust, so ask G2 stays the first ask; (2) a Bool mask
in X_eTaL converts only to the type of the literal it meets, which
shaped `t_hreshold` (Float-only) and `k_eep` (Int masks); (3) the
evaluator's Float printing (full decimal, no exponent) and fold
order (from the right) are not in the reference and were found by
probing (ask G7); (4) Apple's OpenCL has no fp64, so the default
schedule computes Float as `float`, and the f32 results stay within
1e-5 of the f64 evaluator on everything tried; (5) whole-process
timings are dominated by compiling the kernels (0.15 s), so a
kernel cache is the first speed step when speed becomes a goal.

### Saga 2 -- gpu-algebra  [DONE]

Goal: axes and inner products, the operations a model is made of.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 0 | macros-extensions-analysis | DONE (2026-10-07, inserted at the user's request): `docs/macros-and-extensions.md`, every macro and extension candidate judged by the three justifications (concision, speed, a C-ABI library), with measurements: one extension justified (the `Gpu` package, saga gpu-extension below), no new macro; `ffi:b_ind<`, `c_fg<` and `k:c_ases<` reused as they are; ask G8 |
| 1 | axis-reduce | DONE: `reduce OP axis=K` in the IR (text, checker: rank n to n - 1, an empty axis refused; interpreter: each line folded from the right, `r_/_2` and `r_/` of a matrix, rank 3 checked against xetal), the emitter's `reduce_axis_*` kernel (one work-item per result, the evaluator's fold order, so exact), Accel `r_owSums`, `c_olSums`, `r_owMax`, the axis-reduce demo and twin; six twins agree on the interpreter and the GPU. Planned: `reduce` along an axis of a matrix (`r_/_2`, `r_/`), one work-item per row or column; examples and twins |
| 2 | matvec | DONE: `matmul` in the IR (any ranks, a's last axis with b's first; checked; interpreted in `i_nner`'s right-fold order), an untiled kernel (one work-item per result) and a tiled one (`--tile T`, local memory, the same fold order), Accel `m_atmul` (one export for every rank pairing), the matmul demo and twin, check-equiv runs every twin tiled too, `check-sizes` products to 512 (GPU 0.13 s against the evaluator's 28.75 s). Found: `'+ r_/_12` folds down the columns first (the interpreter now does too; ask G7). Planned: `matmul` for matrix-vector (`x '+ '* i_nner w`): one work-item per output; a tiled schedule for matrix-matrix; `ac:m_atvec`, `ac:m_atmul`; sizes to 1024 by 1024 |
| 3 | dense-layer | DONE: Accel `d_ense` (NN's bias-row convention, spread by `'r_ight t_able`) and `s_oftmax` by row (spread by `'l_eft t_able`), the tiny-net demo (two layers, weights from `data/`) and its twin; the XIR needed only general operations: `take`, `drop`, `ravel`, `table` (with `left`, `right`), on top of exp, axis reductions and matmul; the network agrees with the evaluator on the GPU (8 twins). `explain` now describes each shared kernel truthfully (it had called all of them tree reductions). Planned: `y = relu(W x + b)` as X_eTaL (`ac:d_ense`, X_eTaL-ML's NN shape of weights with the bias row) and on the GPU; a softmax by row (exp, row reduce, divide) |
| 4 | schedules | DONE: `Schedule::from_toml` and `validate` (unknown keys refused, each error naming its field), `--schedule FILE` (flags override it) for run, kernel and explain, `schedules/` with the Apple file (tested; equal to the defaults, checked) and templates for the newer NVIDIA GPU (f32 and f64) and the legacy cards (work-groups of 1024, 16 by 16 tiles); the runtime's refusals name the field and the value to use (and suggest the device's own maximum, not half of it). Planned: the schedule as a TOML file per device (work-group size, tile, element types, weight storage host or device); `xetal-gpu run --schedule FILE`; `explain` shows it |

### Saga 2 retrospective

Delivered: the macros-and-extensions analysis (one extension
justified, no new macro), reductions along an axis, inner products
untiled and tiled, a dense layer and a row softmax through four
general IR operations (take, drop, ravel, table), and schedules as
files. Every example, now eight, agrees with the evaluator on the
interpreter exactly and on the GPU within 1e-5, tiled and untiled; a
512 by 512 product runs in 0.13 s on the GPU against the evaluator's
29 s. Learned: (1) the evaluator's fold orders are part of its
meaning for Floats and must be probed (`'+ r_/_12` folds the columns
first, which the interpreter got wrong until a product's sum showed
it; ask G7); (2) a network needed no model-specific IR, only X_eTaL's
own structural primitives, which keeps the IR a candidate for
X_eTaL's (G1); (3) pinned explain output caught nothing about its own
wording, so a wrong description sat pinned for two steps: the
pinning tests the plan, a reader still has to read it.

### Saga 3 -- gpu-extension  [DONE]

Goal: X_eTaL programs call the GPU themselves, through
X_eTaL-extensions' ABI V1 and its `xetal-x` bridge (justified in
`docs/macros-and-extensions.md`, X1: OpenCL is a C-ABI library, and
products and models are faster on the device even through the
text bridge). Testable on this Mac.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | pin-extensions | DONE: `XETAL_EXTENSIONS_COMMIT` (7bdd827), `scripts/xetal-extensions.sh` (clone, its own X_eTaL from the sibling checkout, `xetal-x`, hello and clock built into `target/xetal-extensions/`, `bin/xetal-x` a wrapper: a symlink is not followed when xetal-x looks for libraries beside itself), `just xetal-x`, `just extensions-pin`, `scripts/check-xetal-x.sh` in the gate (its commit, hello through the bridge, every Accel test and demo identical under xetal-x's X_eTaL v0.1.0, which differs from this repo's 75e6a5c). Planned: | `XETAL_EXTENSIONS_COMMIT`, `scripts/xetal-extensions.sh` (clone into `work/xetal-extensions/`, build `xetal-x`), a gate check that `xetal-x` loads hello |
| 2 | gpu-package | DONE: `extensions/gpu/` (`extension.toml`, `rust/` the `xetal-ext-gpu` cdylib, its own workspace on the pinned SDK and this repo's runtime): devices, load, bind, schedule, run, output, explain, kernel, one session per thread; 5 tests through the raw ABI (interpreter and device sessions, Ints kept, errors); `scripts/gpu-ext.sh` (`just gpu-ext`; `--check` in the gate). Planned: | `extensions/gpu/`: `extension.toml`, `rust/` (a cdylib on `xetal-ext-sdk` and this repo's runtime: devices, load, bind, run, output, explain, one session per thread), Rust tests |
| 3 | gpu-facade | DONE: `extensions/gpu/lib/Gpu.xtl` (nine `ffi:b_ind<` lines; the X_eTaL half: `p_roductProgram`, `d_enseProgram`, `p_roduct`, `d_ense`, `n_ear`), `scripts/xx` (xetal-x with the package and the libraries), `scripts/test-ext.sh` (reg-rs: types, basics on the interpreter, bad-shape, products-device skipped without a device; in the gate), `just test-ext`, `just run-x`. Planned: | `lib/Gpu.xtl` by `ffi:b_ind<`, its X_eTaL half (program writers for products and layers, `gp:r_un1`, `gp:n_ear`), reg-rs tests with `xetal-x`, types pinned |
| 4 | offload-demo | DONE: `extensions/gpu/demos/offload.xtl` (`just offload`): 512 by 512 product 14545 ms in the evaluator, 462 ms on the GPU through the extension (31 times; the bridge 416 ms of it); a million elementwise items 188 ms against 2447 ms; `tests/offload-device.xtl` pins the stable facts at 256; demos run with `XETAL_DEMOS=1`; `scripts/xx` loads clock too. Planned: | one X_eTaL program computing a 512 by 512 product with the evaluator and on the GPU, compared and timed; the bridge's cost stated |

### Saga 3 retrospective

Delivered: X_eTaL-extensions pinned like X_eTaL, the GPU as an ABI V1
package (a cdylib with a per-thread session, tested through the raw
ABI), its facade by the existing binding macro plus ordinary X_eTaL
(program writers, one-call products and layers, an f32 check), reg-rs
tests run through `xetal-x`, and the offload demo, whose numbers
confirmed the analysis: products 31 times faster through the bridge,
elementwise work 13 times slower. Learned: (1) `xetal-x` embeds
X_eTaL-extensions' own X_eTaL (v0.1.0), so the gate now checks every
library program prints the same under it; (2) `xetal-x` does not
follow a symlink to find libraries beside itself (an ask of
X_eTaL-extensions; a wrapper here); (3) piping a version to `head`
under `pipefail` is a flaky gate; (4) the bridge's text, not the
device, is the cost: G8 (binary arrays through a native hook) is now
the ask with the clearest payoff.

### Saga 4 -- gpu-models  [ACTIVE]

Goal: research7's PoC G2, a tiny Jev-like typed-decision model (a
small transformer-like classifier, trained from scratch, inference
written as X_eTaL array code) whose complete inference runs on a GPU,
then on the old cards.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 0 | doc-site | DONE (2026-10-08, inserted at the user's request, after X_eTaL's live `/doc`): Accel and Gpu documented with `##` blocks, `###` sections and `## >>` examples (43, run by `scripts/doc-test.sh` in the gate); `scripts/doc-site.sh` builds `pages/doc` with `xetal-x doc --out` (the one host that can expand the facade's binding macro; relative paths so no page names the checkout); `scripts/build-pages.py` the landing page (decorated key lines, the measured results, links); `pages/` untracked; `just pages`, `just publish` (the `gh-pages` branch, one commit), `just check-live`. |
| 0b | repin-doc | DONE (2026-10-08, at the user's request): X_eTaL pinned at d284a8c, then 1998414 (each in its own commit, the gate unchanged); the live `/doc` now built by this repo's `xetal doc` (the sidebar grouped by directory and sorted; headers with or without a blank line), not `xetal-x`'s v0.1.0: documenting only expands Ffi.xtlm and never calls the bridge. `h:` and tuples work in the pinned `xetal` but not under `xetal-x`, so the Gpu facade keeps its bare helper (an ask of X_eTaL-extensions: a newer pin). |
| 0b | repin-doc | DONE (2026-10-08, at the user's request): X_eTaL pinned at d284a8c, then 1998414, each in its own commit with the gate unchanged; the live `/doc` now built by this repo's `xetal doc` (the sidebar grouped by directory and sorted; headers with or without a blank line) instead of `xetal-x`'s v0.1.0, since documenting only expands Ffi.xtlm and never calls the bridge. `h:` and tuples work in the pinned `xetal` but not under `xetal-x`, so the Gpu facade keeps its bare helper (an ask of X_eTaL-extensions: a newer pin). |
| 0c | tuple-plan | DONE (2026-10-08, at the user's request): `docs/tuple-plan.md`: no tuples used yet; five uses judged (a device per call, all outputs at once, a model's weights, training state, the lowering's inputs and outputs); which code may use them now (code run by the pinned `xetal`: the models) and which waits on X_eTaL-extensions' newer pin (X-1) or tuples in ABI V1 (X-2); ask G9. |
| 0e | take-gpu-asks | DONE (2026-10-12): X_eTaL-ML's asks GA1 to GA6 taken into the ledger above, the steps below queued |
| 0f | ml-bench | DONE (2026-10-12): `XETAL_ML_COMMIT` (0ff84e8) and `scripts/xetal-ml.sh`; `execute_timed` and `xetal-gpu run --time` (OpenCL profiling events: the kernels' own time; setup, upload, launches, download apart); `scripts/ml-bench.py` (`just ml-bench`; `--check` in the gate) writes the three benchmarks' twins from their definitions, runs the evaluator, the interpreter (exact), every device (1e-5) and the facade under `xetal-x` with bind, run and output timed apart; `docs/ml-results.md`. On the M1 Max the kernels take under a millisecond, compiling about 85 ms, the bridge 3 to 10 ms. |
| 0g | world-model | DONE (2026-10-12): the world model (432 in, 64 ReLU, 144 sigmoid; 1,936 states and 387 held out) in `scripts/ml-bench.py`: its windows generated by the program's own core under the pinned xetal and bound as inputs with the weights from its `data/`, the twin written from the model's definition; the counts 1897 and 348 the same on the evaluator, the interpreter, the Apple M1 Max in f32 and through the facade (binding 1.2 million inputs as text: a quarter to half a second across runs, GA5's cost). Initially measured from the Mac working tree; ML fac586c now pins the committed demo and the gate covers it. The RTX 5060 Ti and its extension give the same exact counts; their separate timings are in `docs/rtx5060-ml-results.md`. Answers tiny-nn. |
| 0h | transpose | DONE (2026-10-12): `transpose` (`o_\`) and `transpose p` (`p t_ranspose`) in the IR, checked and interpreted as the evaluator computes them (rank 3 probed); a gather kernel per transpose, the same buffer for one that moves nothing, and a tiled kernel for the last two axes under the schedule's `tile` (measured on the M1 Max at 4096 by 4096: 0.046 ms against the gather's 0.10 and a copy's 0.011; the first tiled kernel's 64-bit block divisions made it slower than the gather until they were `int`); Accel `s_cores` (attention scores, `q k^T`) and `w_eightGrad` (`x^T dy`); the transpose demo and twin (9 twins agree on the interpreter and the M1 Max, tiled and untiled). Found: `check-equiv.sh` died under macOS's bash 3.2 after PR #1 (an empty array under `set -u`) and the gate let it through (`[ -x f ] && cmd` under `set -e`): both fixed |
| 0i | windows | GA3: reshape and rotate along an axis in the IR; Conv's windows and a small convolution as twins; one windows operation only if measured worth it |
| 0k | lookup | `lookup` in the IR: X_eTaL's indexing (`i s_elect v`; rows of a matrix by an Int vector), the embedding gather of every transformer and the memory of Engram (saga 8); a gather kernel, one work-item per result item; a twin (an embedding table and a batch of token ids); equivalence |
| 0l | topk | the index of the largest item of each row (argmax: the decision, the next token) and the k largest with their indices (a router's choice, saga 9) in the IR, as the evaluator computes them (ties: the first); kernels; twins; equivalence |
| 0j | denoiser | GA1: X_eTaL-ML's denoiser, once its batch run lands there: about 20 calls, the fixed cost per call shown |
| 1 | tiny-nn | answered by world-model (0g); closed when reached. Planned: a complete tiny trained network (two dense layers, from X_eTaL-ML's trainers or a std-only trainer here) in X_eTaL, weights as data files, on the GPU; the same decision on CPU and GPU |
| 2 | jev-model | moved to saga 6 (gpt-model and jev-head: the classifier is the GPT block with a classification head); closed when reached. Planned: `examples/jev/`: `model.xtl` (embedding lookup, attention as array operations, normalization, feed-forward, classifier), `tokenizer.xtl`, `config.toml`, weights; the needed XIR operations added only when measurement says so (research7: attention decomposes, it is not an intrinsic) |
| 3 | jev-train | moved to saga 7 (jev-train, after the training step exists); closed when reached. Planned: a trainer for the typed-decision task (requests classified as DRAW, ARRAY_QUERY, EXPLAIN, EXECUTE, DOC_SEARCH), f32 |
| 4 | jev-gpu | moved to saga 6 (gpt-batch); closed when reached. Planned: the whole inference scheduled as OpenCL kernels; the same prompt, model and weights giving the same decision on CPU and GPU; a visualization of which X_eTaL operations became which kernels |

### Saga 5 -- gpu-session

Goal: a model's measured time is its kernels' own. Today every run
uploads the weights, compiles the kernels and reads the result back
through the text bridge: the RTX 5060 Ti's 979 ms product is mostly
that, and a training loop would compile once per step. Everything
here is this repository's own (no ask): the runtime, the tool and
the gpu extension.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | kernel-cache | compiled programs cached by the source's hash and the device, in memory for a session and on disk under `work/`; `--time` shows setup fall from about 80 ms (M1 Max) and 200 to 300 ms (RTX 5060 Ti) to the cache's read |
| 2 | resident-weights | the extension's session keeps bound inputs and their device buffers across runs: bind the weights once, rebind only the batch, run again (`gp:r_erun`); the offload demo rerun so: the product's time the kernels' own plus the batch's bind |
| 3 | binary-arrays | Input binding DONE in the independent Arch lane (2026-10-12): checked f32, f64 and i64 files, CLI `pack` and `--bind w=@file`, extension `gp:b_indFile`, world-model exact counts on the interpreter, RTX and bridge; text bind 460 ms versus file bind 13 ms, million-element bind 411 ms versus 10 ms (`docs/rtx5060-binary-results.md`). The million-item output round trip remains to measure; outputs still cross the text bridge. |
| 4 | session-measure | the second call's time on every card at hand (the M1 Max, the RTX 5060 Ti, the RTX 3060 12 GB), the model table of `docs/ml-results.md` and the Arch reports regenerated with the session |

### Saga 6 -- gpu-gpt

Goal: a tiny character-level GPT written in X_eTaL, its whole
forward pass as one XIR program on a batch of prompts, measured on
every card; the Jev-like typed-decision classifier is the same block
with a classification head, so saga 4's jev-model and jev-gpu are
done here.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | gpt-model | X_eTaL-ML's micro-gpt (`demos/micro-gpt/`: Karpathy's microgpt, one layer, 16 wide, 4 heads, 4,192 weights, trained on baby names, its scores equal to its Rust trainer's to 1e-9; on the Norm, Attention and Sample libraries), taken from the pinned X_eTaL-ML as the model (the user, 2026-10-12: start from it rather than write a GPT here); its forward pass over a whole name under the causal mask as the reference, the evaluator's output the baseline; what its libraries use that the IR lacks, listed |
| 2 | gpt-twin | the XIR program written by a generator from micro-gpt's config and weights (`scripts/gpt-twin.py`, as `ml-bench.py` writes the benchmarks'), not by hand; the interpreter exact, every device within tolerance; `explain` shows which X_eTaL lines became which kernels |
| 3 | gpt-batch | a batch of prompts as one run; tokens per second against the evaluator on the M1 Max, the RTX 5060 Ti and the RTX 3060; the results table on the landing page |
| 4 | jev-head | the typed-decision classifier (DRAW, ARRAY_QUERY, EXPLAIN, EXECUTE, DOC_SEARCH): the block above with a classification head and argmax (saga 4's topk); the same decision on every card |

### Saga 7 -- gpu-train

Goal: a training step as X_eTaL array code with the backward pass
written out, running on the device: the tiny GPT trained from
scratch, then a small pretrained model fine-tuned. Training is where
products dominate most, and a backward pass is a dozen whole-array
lines, each a product with a transpose (saga 4's transpose is for
this).

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | backward | a library of layer gradients (dense, layer norm, attention, softmax cross-entropy), each checked by the evaluator against finite differences; a twin each |
| 2 | train-step | forward, backward and Adam as one XIR program per step, the weights resident (saga 5); a few steps' losses the same on the evaluator and the device |
| 3 | train-gpt | the tiny GPT trained from scratch on a small committed corpus (a fixed seed), samples before and after, steps per second per card |
| 4 | fine-tune | GPT-2 small's weights converted to binary files (saga 5), a few hundred steps on a small corpus, perplexity before and after; the memory each card needs (f32 with activations fits the 6 GB and larger cards, not the K4200) |
| 5 | jev-train | the typed-decision data set generated and committed small, the classifier of saga 6 trained with the step above; the trained weights as data |

### Saga 8 -- gpu-memory

Goal: a conditional memory beside the embeddings, after DeepSeek's
Engram (2026): hashed n-grams of the recent tokens looked up in a
table and gated into the residual stream. In X_eTaL the module is a
few lines (`d_iv` and `m_od` for the hash, the lookup of saga 4, a
gate); on a GPU it is a gather, so the demonstration is the array
language's concision and the memory's size, not a speedup.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | ngram-hash | the hashes of the last n tokens as array code; the same table positions on the evaluator and the device |
| 2 | engram-layer | the table lookup and the gate added to the GPT of saga 6; the evaluator's baseline; the twin from the generator |
| 3 | engram-measure | the gather's cost beside a product's on each card; the table scaled up on the 12 GB and 16 GB cards against the 6 GB one |

### Saga 9 -- gpu-moe

Goal: a mixture-of-experts layer in the array language's own way
first: the router chooses, every expert computes, the router's mask
keeps what was chosen; sparse dispatch only if measurement asks for
it.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | router | softmax over the experts and top-k (saga 4) per token; the same choices on the evaluator and the device |
| 2 | dense-moe | every expert as one product over a rank-3 weight tensor (a product over a leading axis in the IR, or one `matmul` per expert in the program), masked by the router: wasteful, whole-array, entirely on the device; the GPT with a MoE feed-forward |
| 3 | sparse-moe | only if dense-moe is measured too slow: tokens sorted by expert (a grade or a scatter in the IR), each expert on its slice |
| 4 | moe-measure | per card, dense against sparse, against the evaluator |

### Saga 10 -- gpu-hardware

The RTX 5060 Ti validation was brought forward on 2026-10-12 in an
independent `rtx5060` branch and saga lane while the Mac continues
gpu-models. Its hardware and ML results are recorded separately in
`docs/rtx5060-results.md` and `docs/rtx5060-ml-results.md`; the ML
runner accepts `--output` and `--require-device` so another host's
measurements are preserved and a hidden GPU cannot produce a report
that silently skips the device. A second Arch lane, `rtx5060-world`,
pins ML fac586c, the committed world model, and checks its exact
counts (1897 and 348) on the interpreter, RTX and extension bridge.
The language and extension pins are unchanged. Other NVIDIA
cards still need their own runs; the shared modern schedule remains
a template. The remaining model steps belong to the Mac's lane.

Goal: the cards of the hardware table, first those in the Arch
machine the work moves to soon (the user, 2026-10-07), then the old
Quadro K4200, Tesla M10 and Tesla P4; the Apple M3 Pro whenever it is at hand (it needs
no new setup, only a run).
Nothing in the code changes on the move: the OpenCL library is
loaded at run time and the checks use every device found. The
machine needs `ocl-icd`, the vendor's OpenCL (`opencl-nvidia`,
`rocm-opencl-runtime`) and `clinfo`; with fp64 there, `--float f64`
schedules become testable. Nothing here can be tested on this Mac.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 0 | apple-m3-pro | the checks (`just gate`, `just sizes`, `just offload`) on the Apple M3 Pro: its row of the hardware table filled in from `xetal-gpu devices`, `apple-m1-max.toml` confirmed for it or a file of its own, its timings beside the M1 Max's |
| 1 | nvidia-opencl | the runtime on the Arch machine's cards: the hardware table's row filled in (devices, driver, OpenCL version, fp64, work-group limit), `nvidia-modern.toml` run and marked tested, every example's equivalence and `just sizes` on each. The RTX 5060 Ti: DONE 2026-10-12 in the Arch agent's own lane (`schedules/nvidia-rtx5060-ti*.toml`, `docs/rtx5060-results.md`, `docs/rtx5060-ml-results.md`, PR #1) |
| 1b | rtx3060 | the RTX 3060 12 GB (the user, 2026-10-12): the same checks, the six size cases and the ML benchmarks with `just ml-bench --require-device --output docs/rtx3060-ml-results.md`, a schedule file of its own if the modern template does not fit, its row of the hardware table and a column of the README's results table; and, as they land, the model demos of sagas 6 to 9, so every result names its card |
| 2 | legacy | the Quadro K4200 (Kepler, 4 GB), the Tesla M10 (four Maxwell GPUs) and the Tesla P4 (Pascal): the driver that still serves each recorded (Kepler may need NVIDIA's legacy branch), whether `nvidia-legacy.toml` fits each or one needs a file of its own, `nvidia-legacy.toml` run and marked tested, every example on each |
| 3 | generations | the one program on every card present, Kepler to Blackwell, the K4200's 4 GB as the memory-limited case; a table of what ran where and how fast |
| 4 | quantized | f16, then i8, then the ternary (1.58-bit) Jev: the GPU schedule decides the physical representation (packed 2-bit storage, i32 accumulation) of the semantic type (research7 "1.58-bit becomes a terrific later experiment") |

### Saga 11 -- xetal-lowering (deferred: asks G1, G2, G6)

Goal: the one-command demo, `xetal run --device opencl:0 demo.xtl`,
once X_eTaL has the accelerator IR and the lowering.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | adopt-ir | `components/xir` becomes an adapter from X_eTaL's IR; the hand-written twins deleted; the equivalence script runs `xetal` for both sides |
| 2 | one-command | `xetal run --device opencl:N FILE.xtl` through the backend registration X_eTaL offers (ask G6), or a `xetal-gpu run FILE.xtl` that calls X_eTaL's lowering |
| 3 | explain | `xetal explain --device opencl FILE.xtl`: the program, the IR, the schedule, the kernels, side by side (research7 section 10) |
