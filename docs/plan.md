# X_eTaL-gpu -- Implementation Plan

X_eTaL programs on GPUs: the GPU compiler, runtime and scheduling for
X_eTaL (the eXperimental Extensible Typed Array Language, developed in
`../X_eTaL`), aimed first at old GPUs the current CUDA toolkits no
longer support (K80, M40, P40, P100, GTX 1660, RTX 2060 and up), by
way of OpenCL C 1.2. The goal is to demonstrate the concept, that one
typed array program runs unchanged as sequential software, as
massively parallel software, and (in the sibling project) as spatial
hardware, not to be fast.

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
| A5 | **OpenCL C 1.2 first; CUDA is a later backend.** The emitter (`components/opencl`) writes plain OpenCL C 1.2 kernels as text (testable without a GPU: the kernel text is pinned); the runtime (`components/runtime`) is a thin host on the `opencl3` crate: buffers in, kernel, buffers out. No vendor extensions beyond `cl_khr_fp64` when f64 is asked for. | research7 sections 7 and 8: OpenCL reaches Kepler through Ampere with one toolchain; CUDA would make this repo know NVIDIA's toolkit matrix. Apple's OpenCL (1.2, deprecated but present on this Mac) is enough for development; the old NVIDIA cards are the target. |
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

## Roadmap

Implementable work first (the GPU side, against hand-lowered
examples); what needs an X_eTaL ask (the lowering, the one-command
demo) is in the last saga and moves up when the ask lands.

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

### Saga 3 -- gpu-extension  [ACTIVE]

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
| 4 | offload-demo | one X_eTaL program computing a 512 by 512 product with the evaluator and on the GPU, compared and timed; the bridge's cost stated |

### Saga 4 -- gpu-models

Goal: research7's PoC G2, a tiny Jev-like typed-decision model (a
small transformer-like classifier, trained from scratch, inference
written as X_eTaL array code) whose complete inference runs on a GPU,
then on the old cards.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | tiny-nn | a complete tiny trained network (two dense layers, from X_eTaL-ML's trainers or a std-only trainer here) in X_eTaL, weights as data files, on the GPU; the same decision on CPU and GPU |
| 2 | jev-model | `examples/jev/`: `model.xtl` (embedding lookup, attention as array operations, normalization, feed-forward, classifier), `tokenizer.xtl`, `config.toml`, weights; the needed XIR operations added only when measurement says so (research7: attention decomposes, it is not an intrinsic) |
| 3 | jev-train | a trainer for the typed-decision task (requests classified as DRAW, ARRAY_QUERY, EXPLAIN, EXECUTE, DOC_SEARCH), f32 |
| 4 | jev-gpu | the whole inference scheduled as OpenCL kernels; the same prompt, model and weights giving the same decision on CPU and GPU; a visualization of which X_eTaL operations became which kernels |

### Saga 5 -- gpu-hardware

Goal: other GPUs, first the Arch machine with a newer GPU that the
work moves to soon (the user, 2026-10-07), then the old cards.
Nothing in the code changes on the move: the OpenCL library is
loaded at run time and the checks use every device found. The
machine needs `ocl-icd`, the vendor's OpenCL (`opencl-nvidia`,
`rocm-opencl-runtime`) and `clinfo`; with fp64 there, `--float f64`
schedules become testable. Nothing here can be tested on this Mac.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | nvidia-opencl | the runtime on an M40 or P40 (the trophy K80 later); device quirks recorded; every example's equivalence on it |
| 2 | generations | the one program across every generation present (Kepler to Ampere and newer); a table of what ran where |
| 3 | quantized | f16, then i8, then the ternary (1.58-bit) Jev: the GPU schedule decides the physical representation (packed 2-bit storage, i32 accumulation) of the semantic type (research7 "1.58-bit becomes a terrific later experiment") |

### Saga 6 -- xetal-lowering (deferred: asks G1, G2, G6)

Goal: the one-command demo, `xetal run --device opencl:0 demo.xtl`,
once X_eTaL has the accelerator IR and the lowering.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | adopt-ir | `components/xir` becomes an adapter from X_eTaL's IR; the hand-written twins deleted; the equivalence script runs `xetal` for both sides |
| 2 | one-command | `xetal run --device opencl:N FILE.xtl` through the backend registration X_eTaL offers (ask G6), or a `xetal-gpu run FILE.xtl` that calls X_eTaL's lowering |
| 3 | explain | `xetal explain --device opencl FILE.xtl`: the program, the IR, the schedule, the kernels, side by side (research7 section 10) |
