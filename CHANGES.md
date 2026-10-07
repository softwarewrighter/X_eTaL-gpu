# Changes

Every commit, newest first, grouped by day. Times are Pacific
(UTC-07:00), as committed.

Categories: `feat` new capability, `fix` a bug or wrong behavior,
`refactor` structure without behavior change, `test` tests only,
`build` build and tooling, `lib` an X_eTaL library or a change to
one, `docs` documentation, `plan` saga planning and reordering,
`release` milestone release, `chore` agentrail bookkeeping (step
complete, saga archive).

## 2026-10-07

- 07:25 `docs` CHANGES.md times set to the commits' (as committed, not as estimated).
- 07:15 `plan` Saga gpu-foundation archived; saga gpu-algebra started with its four steps (axis-reduce, matvec, dense-layer, schedules), as `docs/plan.md` plans it.
- 07:05 `chore` Saga step reduce-gpu completed; saga gpu-foundation done.
- 07:05 `test` The reductions beyond one work-group (step 6): `scripts/check-sizes.py` (`just sizes`; the gate runs 1024 and 65536) generates Ints and Floats into `work/sizes/`, runs an Accel program on them with the pinned `xetal` and the same computation as an XIR program with bound inputs on the interpreter (exactly) and on every device (within 1e-5), timed; 1024, 65536 and 1048576 items all agree on the Apple M1 Max (a million-item reduce is three launches; 0.24 s against the evaluator's 1.06 s, compile included). Timings in the runtime README; saga 1 retrospective in the plan; saga gpu-foundation done.

## 2026-10-06

- 20:35 `chore` Saga step opencl-run completed.
- 20:35 `feat` The OpenCL runtime (step 5): `xetal-gpu-runtime` on the opencl3 crate (the OpenCL library loaded at run time, so the tool builds and runs anywhere and `devices` says when there is none); every device of every platform listed with its kind, version, units, work-group limit, memory and fp64; `execute`: the device checked against the plan (double precision, work-group size, with the flag to pass), the source built with the device's compiler and its log as the error, every buffer allocated in the schedule's element type with inputs and constants converted up, the launches run in order on an in-order queue, the outputs read back and widened to the program's types. `xetal-gpu devices` and `run --device opencl:N` (with the schedule flags). `scripts/check-equiv.sh` runs every twin on every device found (Floats within 1e-5, Ints exactly) and says when the GPU side is skipped. On this Mac (Apple M1 Max, OpenCL 1.2, no fp64) all five twins agree with the evaluator, `-0.0` included; the runtime's four tests pass, among them a 70000-item reduction in three passes.
- 19:29 `chore` Saga step opencl-emit completed.
- 19:29 `feat` The schedule and the OpenCL C 1.2 emitter (step 4): `xetal-gpu-opencl`; a Schedule (Float as `float` or `double`, Int as `long` or `int`, the work-group size) and a Plan (kernel source, buffers with who fills them, launches with sizes, outputs); consecutive elementwise values of one shape fused into one kernel with one work-item per element, only the values an output or a later kernel needs written to buffers, single-value constants as literals; a tree reduction in local memory per operation and type, launched again on its partials until one value is left (a million items: three launches); floored `idiv` and `mod`, `floor` casts, `!= 0` to Bool; `explain` in words. `xetal-gpu kernel` and `explain` with `--float`, `--int`, `--work-group`. 7 unit tests, and every twin's generated source and plan pinned under `components/opencl/tests/expected/` (threshold is one kernel of ten statements).
- 18:36 `chore` Saga step xir completed.
- 18:36 `feat` The provisional accelerator IR (step 3): `components/` workspace with `xetal-gpu-xir` (the IR: i32/i64/f32/f64/bool, static shapes, const, input, map with scalar extension, select, reduce, cast; a text form that round-trips; a checker that types every value and names the one at fault; the reference interpreter with X_eTaL's semantics: floored `idiv` and `mod`, a reduce folded from the right, f32 and i32 rounded after every operation; printing as `xetal` prints, Floats in full decimal with a point) and `xetal-gpu-cli` (`xetal-gpu check`, `run --device cpu --bind NAME=...`, `print`); each Accel demo's hand-lowered `.xir` twin; `scripts/compare-out.py` (Ints exact, Floats within a tolerance, -0.0 equal to 0.0) and `scripts/check-equiv.sh` (every twin against its baseline, exactly on the interpreter; in the gate). 17 unit tests; the five twins agree with the evaluator exactly.
- 18:24 `chore` Saga step accel-lib completed.
- 18:24 `lib` Accel (`ac:`), the acceleratable subset as a library (step 2): `v_add`, `s_cale`, `r_elu`, `t_hreshold` (Float: the mask made Float and multiplied in, since a Bool mask converts only to the type of the literal it meets), `k_eep` (an Int mask), `s_um`, `l_argest`, `d_ot`, `p_ipeline` (the sum of the positive items of x + x * w: research7's acceptance test); tests (basics; checks against direct computations), the exports' types pinned, the five example programs as its demos (vector-add, saxpy, threshold, reduce-sum, pipeline) with reg-rs baselines, the reference page; `libs/` in the README.
- 18:04 `chore` Saga step scaffold completed.
- 18:04 `build` Scaffold (saga gpu-foundation, step 1): the process of the sibling repos (agentrail saga, CLAUDE.md with AGENTS.md as a symlink, COPYRIGHT, LICENSE, `.gitignore`, justfile, `scripts/gate.sh`), X_eTaL pinned at 75e6a5c (`XETAL_COMMIT`, `scripts/xetal.sh`, `scripts/check-xetal.sh`, which also checks the Core dump), the library tooling of X_eTaL-ML (`scripts/xt`, `libs.py`, `test-libs.sh`, `new-lib.sh`, `selftest-libs.sh`, `templates/Library`), `docs/research7.txt` (archival), `docs/plan.md` (architecture decisions A1 to A9, the roadmap: sagas gpu-foundation, gpu-algebra, gpu-models, gpu-hardware), `docs/xetal-asks.md` (G1 to G7: the accelerator IR, the lowering, static shapes and the Core dump are the blockers; this repo carries a provisional IR until they land), this file, the README.
