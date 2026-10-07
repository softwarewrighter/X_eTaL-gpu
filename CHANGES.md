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

- 18:10 `build` Scaffold (saga gpu-foundation, step 1): the process of the sibling repos (agentrail saga, CLAUDE.md with AGENTS.md as a symlink, COPYRIGHT, LICENSE, `.gitignore`, justfile, `scripts/gate.sh`), X_eTaL pinned at 75e6a5c (`XETAL_COMMIT`, `scripts/xetal.sh`, `scripts/check-xetal.sh`, which also checks the Core dump), the library tooling of X_eTaL-ML (`scripts/xt`, `libs.py`, `test-libs.sh`, `new-lib.sh`, `selftest-libs.sh`, `templates/Library`), `docs/research7.txt` (archival), `docs/plan.md` (architecture decisions A1 to A9, the roadmap: sagas gpu-foundation, gpu-algebra, gpu-models, gpu-hardware), `docs/xetal-asks.md` (G1 to G7: the accelerator IR, the lowering, static shapes and the Core dump are the blockers; this repo carries a provisional IR until they land), this file, the README.
