# Changes

Every commit, newest first, grouped by day. Times are Pacific
(UTC-07:00), as committed.

Categories: `feat` new capability, `fix` a bug or wrong behavior,
`refactor` structure without behavior change, `test` tests only,
`build` build and tooling, `lib` an X_eTaL library or a change to
one, `docs` documentation, `plan` saga planning and reordering,
`release` milestone release, `chore` agentrail bookkeeping (step
complete, saga archive).

## 2026-10-06

- 19:30 `feat` The provisional accelerator IR (step 3): `components/` workspace with `xetal-gpu-xir` (the IR: i32/i64/f32/f64/bool, static shapes, const, input, map with scalar extension, select, reduce, cast; a text form that round-trips; a checker that types every value and names the one at fault; the reference interpreter with X_eTaL's semantics: floored `idiv` and `mod`, a reduce folded from the right, f32 and i32 rounded after every operation; printing as `xetal` prints, Floats in full decimal with a point) and `xetal-gpu-cli` (`xetal-gpu check`, `run --device cpu --bind NAME=...`, `print`); each Accel demo's hand-lowered `.xir` twin; `scripts/compare-out.py` (Ints exact, Floats within a tolerance, -0.0 equal to 0.0) and `scripts/check-equiv.sh` (every twin against its baseline, exactly on the interpreter; in the gate). 17 unit tests; the five twins agree with the evaluator exactly.
- 18:45 `chore` Saga step accel-lib completed.
- 18:40 `lib` Accel (`ac:`), the acceleratable subset as a library (step 2): `v_add`, `s_cale`, `r_elu`, `t_hreshold` (Float: the mask made Float and multiplied in, since a Bool mask converts only to the type of the literal it meets), `k_eep` (an Int mask), `s_um`, `l_argest`, `d_ot`, `p_ipeline` (the sum of the positive items of x + x * w: research7's acceptance test); tests (basics; checks against direct computations), the exports' types pinned, the five example programs as its demos (vector-add, saxpy, threshold, reduce-sum, pipeline) with reg-rs baselines, the reference page; `libs/` in the README.
- 18:12 `chore` Saga step scaffold completed.
- 18:10 `build` Scaffold (saga gpu-foundation, step 1): the process of the sibling repos (agentrail saga, CLAUDE.md with AGENTS.md as a symlink, COPYRIGHT, LICENSE, `.gitignore`, justfile, `scripts/gate.sh`), X_eTaL pinned at 75e6a5c (`XETAL_COMMIT`, `scripts/xetal.sh`, `scripts/check-xetal.sh`, which also checks the Core dump), the library tooling of X_eTaL-ML (`scripts/xt`, `libs.py`, `test-libs.sh`, `new-lib.sh`, `selftest-libs.sh`, `templates/Library`), `docs/research7.txt` (archival), `docs/plan.md` (architecture decisions A1 to A9, the roadmap: sagas gpu-foundation, gpu-algebra, gpu-models, gpu-hardware), `docs/xetal-asks.md` (G1 to G7: the accelerator IR, the lowering, static shapes and the Core dump are the blockers; this repo carries a provisional IR until they land), this file, the README.
