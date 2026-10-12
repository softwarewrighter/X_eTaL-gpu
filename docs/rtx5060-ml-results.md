# X_eTaL-ML's benchmarks on GPUs

X_eTaL-ML's ask GA1 (its `docs/gpu-asks.md`): its batch programs,
from X_eTaL-ML 0ff84e8 (`XETAL_ML_COMMIT`), run by the pinned
X_eTaL 1998414 (the reference) and as XIR programs on this
repository's interpreter and devices. Written by `scripts/ml-bench.py`
(`just ml-bench`) on `large12`; whole-process times include reading
the data and, on a device, compiling the kernels. The device's time
is the kernels' own, from OpenCL profiling events, summed; the
launches' time is what the host waited from the first enqueue to
`finish`. Through the facade, the program runs under `xetal-x` with
the gpu extension, each call timed with `ck:t_ime`: the run is the
device's (compiling, uploading, the kernels, reading back), and the
bridge's cost is the bind and output calls, which carry the arrays
as text. The run totals are dominated by compiling the kernels on
every run (there is no kernel cache yet); the kernels' own time is
the column that compares devices.
Every result agrees with the evaluator: exactly on the interpreter,
within a relative 1e-5 on a device (f32).

| Benchmark | What | Evaluator's result | Evaluator (ms) | Interpreter (ms) |
| --------- | ---- | ------------------ | -------------- | ---------------- |
| dense | a dense layer's product, 1024 x 16 by 16 x 16, four times | 40394.999999997555 | 214 | 11 |
| inner-wide | the CNN's dense layer, 1 x 1352 by 1352 x 10, forty times | 1014.2999999999904 | 115 | 3 |
| softmax | softmax by row on 1000 x 64, ten times | 10000.000000000902 | 357 | 53 |

## opencl:0: NVIDIA GeForce RTX 5060 Ti

| Benchmark | Largest difference | Kernels, device (ms) | Launches, host (ms) | Run total (ms) | Whole process (ms) | Facade: run (ms) | Facade: bridge, bind and output (ms) |
| --------- | ------------------ | -------------------- | ------------------- | -------------- | ------------------ | ---------------- | ------------------------------------ |
| dense | 2.44e-09 | 0.077 | 0.22 | 329 | 428 | 204 | 16 |
| inner-wide | 0.00251 | 1.174 | 1.33 | 383 | 478 | 208 | 15 |
| softmax | 0.000977 | 0.551 | 1.54 | 946 | 1052 | 205 | 26 |

The interpreter is this repository's reference interpreter (`--device cpu`), f64.
