# X_eTaL-ML's workloads on GPUs, from source

X_eTaL-ML's ask GA1 (its `docs/gpu-asks.md`): its benchmarks and its
batch programs, from X_eTaL-ML c032768 (`XETAL_ML_COMMIT`), run by the
pinned X_eTaL 27b3fd0 (the reference) and lowered from their own
source by `xetal-gpu run` (`components/lower`, plan A4b: nothing is
written by hand or by a generator) on this repository's interpreter
and devices. Written by `scripts/ml-bench.py` (`just ml-bench`) on
`max`. A statement the device cannot take (a sort, a text, a loop
over calls) is evaluated by the pinned `xetal` on the host and its
value bound as an input; the `host` column counts them. The
lowering runs the evaluator once per such statement, so it is slow
the first time (the `lowering, cold` column) and cached after
(`work/lower-cache`, keyed by everything it read and the tool); the
whole-process times are with a warm cache and include reading the
data and, on a device, compiling the kernels. The device's time is
the kernels' own, from OpenCL profiling events, summed; the launches'
time is what the host waited from the first enqueue to `finish`.
Through the macro, the program in `ml/` runs under `xetal-x` with
the gpu extension: the accelerated function's text goes to the
runner, which lowers it for the arguments' shapes on the first call
(the first call's time includes that, compiling and the upload) and
keeps the session for the second; the arguments cross the bridge as
text each call (ask G8), which is most of the world model's second
call. Every result agrees with the evaluator: exactly on the
interpreter, within a relative 1e-5 on a device (f32). A device run
is made twice and the second reported, since the first after a
change compiles the kernels cold. These are single samples after
warmup, not medians. The evaluator and interpreter are reference
implementations, not optimized CPU library baselines
(`docs/m1max-session-results.md` compares against those); compare
each GPU with its own host's evaluator.

| Program | What | Evaluator's result | Evaluator (ms) | Interpreter (ms) | Lowering, cold (s) | Host |
| ------- | ---- | ------------------ | -------------- | ---------------- | ------------------ | ---- |
| dense | a dense layer's product, 1024 x 16 by 16 x 16, four times, summed | 40394.999999997555 | 113 | 7 | 0.0 | 0 |
| inner-wide | the CNN's dense layer, 1 x 1352 by 1352 x 10, forty times, summed | 1014.2999999999904 | 58 | 7 | 0.0 | 0 |
| softmax | softmax by row on 1000 x 64, ten times, summed | 10000.000000000902 | 174 | 27 | 0.0 | 0 |
| conv | nine shifted copies of a 28 x 28 picture times 8 x 9 filters, ten times, summed | 12947.999999999645 | 73 | 8 | 0.0 | 0 |
| world-model | 432 pixels in, 64 ReLU, 144 sigmoid out; every state's next frame as one batch of 1,936, the 387 held out, a dream of twelve steps: the next frames predicted exactly | 1897 and 348 | 8282 | 167 | 41.4 | 3 |
| cnn-batch | the digit reader over the first 1,000 MNIST test digits: windows, 8 filters, ReLU, pooling, the dense layer, argmax; the count read right | 971 | 8651 | 527 | 0.3 | 0 |
| diffusion | the learned denoiser: 20 steps from given noise, 128 pictures of 10 x 10 at once; how many have a shape's 8 to 40 pixels on | 94.0 | 6417 | 16 | 32.0 | 3 |
| micro-gpt-batch | the tiny GPT's forward pass over the first 1,000 names, a name per call (the loop on the evaluator): the mean loss per token and the next letters right | 2.2496 and 2098 7000 | 3188 | 9 | 3.2 | 2 |

## opencl:0: Apple M1 Max

| Program | Largest difference | Kernels, device (ms) | Launches, host (ms) | Run total (ms) | Whole process (ms) | Macro: first call (ms) | Macro: second call (ms) |
| ------- | ------------------ | -------------------- | ------------------- | -------------- | ------------------ | ---------------------- | ----------------------- |
| dense | 2.44e-09 | 0.008 | 5.68 | 70 | 79 | 94 | 22 |
| inner-wide | 0.0019 | 0.567 | 21.23 | 83 | 95 | 97 | 31 |
| softmax | 0.00195 | 0.075 | 12.08 | 74 | 86 | 107 | 47 |
| conv | 3.55e-10 | 0.050 | 7.60 | 71 | 81 | 80 | 23 |
| world-model | 0 | 0.415 | 29.83 | 128 | 150 | 362 | 320 |
| cnn-batch | 0 | 0.591 | 33.72 | 110 | 159 | -- | -- |
| diffusion | 0 | 0.240 | 23.89 | 123 | 141 | -- | -- |
| micro-gpt-batch | 6.64e-08 | 0.003 | 1.20 | 63 | 74 | -- | -- |

Not compared on a device: diffusion's output line 1, the program's own check of one picture against the trainer's sampler asks for 1e-6; f32 gives 8.5e-6 (asked of X_eTaL-ML: a tolerance f32 meets, as the world model's does).

The interpreter is this repository's reference interpreter (`--device cpu`), f64.
The `host` column: statements evaluated by `xetal` instead of the device
(the world model's sort-free preparation is all device work; the
denoiser's two text prints; the GPT batch's loop over names).
