# ml: X_eTaL-ML's workloads through the macro

The programs here run X_eTaL-ML's benchmarks and its world model
through `gp:a_ccel<` (the Gpu extension's macro, `docs/README.md` of
`extensions/gpu`): the accelerated function is written as ordinary
X_eTaL, the macro hands its text to the runner, which lowers it for
the arguments' shapes, keeps the weights resident and runs it on
opencl:0; the rest of the program runs in `xetal-x`'s evaluator. Each
prints the value X_eTaL-ML's own program prints (so the result is
checked against the evaluator), then the first call's time (lowering,
compiling and uploading included) and a second call's.

`scripts/ml-bench.py` runs them, from the repository's root, with
X_eTaL-ML's libraries on `XETAL_PATH` (`work/xetal-ml/libs/*/src`),
beside the same programs lowered whole from X_eTaL-ML's own source by
`xetal-gpu run`. They read X_eTaL-ML's data from `work/xetal-ml/`,
so `just xetal-ml` first.

| Program | X_eTaL-ML's | Accelerated |
| ------- | ----------- | ----------- |
| `dense.xtl` | `bench/dense.xtl` | the four products and their sums |
| `inner-wide.xtl` | `bench/inner-wide.xtl` | the forty wide products |
| `softmax.xtl` | `bench/softmax.xtl` | ten softmaxes by row |
| `conv.xtl` | `bench/conv.xtl` | the nine shifted copies and the filters' product, ten times |
| `world-model.xtl` | `demos/world-model/world-model.xtl` | the two-layer prediction of every window; the two counts |
