# Saga: gpu-foundation

Goal: the process, the pinned X_eTaL, the acceleratable subset as a
tested X_eTaL library, a provisional accelerator IR with an
interpreter, OpenCL kernels for elementwise operations and
reductions, a host runtime, and the first proof: one X_eTaL program's
result reproduced on a GPU. The roadmap and the architecture
decisions are in docs/plan.md; the blockers in docs/xetal-asks.md.

1. scaffold: agentrail saga, CLAUDE.md/AGENTS.md, README, COPYRIGHT,
   LICENSE, .gitignore, justfile, scripts/gate.sh, the pinned X_eTaL
   (XETAL_COMMIT, scripts/xetal.sh, check-xetal.sh), the library
   tooling (xt, libs.py, test-libs.sh, new-lib.sh, selftest-libs.sh,
   templates/Library), docs/plan.md, docs/xetal-asks.md, CHANGES.md.
2. accel-lib: libs/Accel, the acceleratable subset as exported
   functions, tests, pinned types, the example programs as its demos
   (vector-add, saxpy, threshold, reduce-sum, pipeline), reg-rs
   baselines, the reference page.
3. xir: components/xir, the provisional IR (types, shapes, ops), its
   text form (parse, print, round-trip), checking, the reference
   interpreter; each demo's .xir twin; scripts/check-equiv.sh;
   xetal-gpu check and run --device cpu.
4. opencl-emit: components/opencl, the schedule and the OpenCL C 1.2
   emitter; every twin's kernel text pinned; xetal-gpu kernel and
   explain.
5. opencl-run: components/runtime on opencl3; xetal-gpu devices and
   run --device opencl:N; the equivalence script runs the GPU side
   when a device is present; every demo reproduced on the GPU.
6. reduce-gpu: the reduction on the device (work-group partials,
   second pass); the pipeline example end to end on the GPU; sizes
   beyond one work-group; timings noted.
