# Accel

The acceleratable subset of X_eTaL: elementwise arithmetic, masks and
a select, reductions, each one whole-array expression that a GPU (or
an FPGA) can take on directly. Its demos are the example programs
X_eTaL-gpu runs on a GPU and checks against the X_eTaL evaluator.

```
"ac:" u_se< "Accel"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `Accel.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples |
| [`demos/`](demos/) | the example programs: vector add, SAXPY, threshold, reductions, the acceptance pipeline (each with a `.xir` twin for the GPU tool, once step 3 lands) |
| [`data/`](data/) | the tiny network's weights, bias rows last |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`), the reference the GPU must reproduce |

```bash
just demo-lib Accel        # run its demos
just run-lib Accel         # run its test programs
just test-lib Accel        # check every baseline
```
