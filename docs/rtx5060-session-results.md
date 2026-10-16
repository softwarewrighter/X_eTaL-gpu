# Repeated inference on the NVIDIA GeForce RTX 5060 Ti

Measured 2026-10-16 on `large12` (NVIDIA GeForce RTX 5060 Ti).
NumPy 2.5.4; OpenBLAS 0.3.34.106.0, SkylakeX. ML c032768; X_eTaL 27b3fd0.

The trained world model runs both dense layers, biases, ReLU, sigmoid
and exact-frame counts. The two batches contain N states and min(N, 387)
held-out states. Larger batches cycle the committed data; this is a throughput
test of a trained model, not additional unseen evaluation data.

Each setting runs 21 times; report medians after discarding the first three.
CPU: the fastest of 1, 6 and 12 BLAS threads. GPU: the fastest refreshed
run of tiles 0, 8 and 16. All settings give the same exact counts as the
reference interpreter; the full batch gives the evaluator baseline 1897/348.

This model does not beat optimized CPU inference with refreshed inputs at
any measured batch size. Resident-only timings isolate execution reuse,
but repeating identical inputs is not a practical throughput speedup.

| Batch N + held | Interpreter (ms) | CPU BLAS (ms), threads | Resident (ms) | New batch (ms) | Tile | Setup (ms) | First call (ms) | Stream speedup | Break-even calls |
| -------------- | ---------------- | ---------------------- | ------------- | -------------- | ---- | ---------- | --------------- | -------------- | ---------------- |
| 1 + 1 | 0.336 | 0.078, 6 | 0.220 | 0.244 | 0 | 564.8 | 635.5 | 0.32x | -- |
| 16 + 16 | 1.458 | 0.112, 12 | 0.221 | 0.295 | 16 | 477.6 | 546.8 | 0.38x | -- |
| 64 + 64 | 6.076 | 0.212, 6 | 0.222 | 0.411 | 0 | 487.3 | 558.1 | 0.52x | -- |
| 256 + 256 | 24.545 | 0.553, 12 | 0.248 | 0.946 | 16 | 444.4 | 449.1 | 0.58x | -- |
| 1024 + 387 | 68.373 | 1.251, 12 | 0.252 | 2.848 | 8 | 457.1 | 534.9 | 0.44x | -- |
| 1936 + 387 | 113.724 | 1.953, 6 | 0.308 | 4.894 | 8 | 556.1 | 634.4 | 0.40x | -- |
| 8192 + 387 | 381.337 | 7.814, 6 | 0.720 | 17.407 | 8 | 435.9 | 478.5 | 0.45x | -- |

Resident: no inputs change. New batch: reupload X and Xh each call, while
weights, biases and targets stay resident. The same batch bytes are reuploaded
to isolate transfer costs. The interval includes host conversion, upload,
kernel launches, synchronization and count download; file reads, binary
conversion and process startup are excluded on both CPU and GPU.
Break-even charges the GPU's complete first call, then its measured refreshed
latency, against repeated CPU calls. A dash means no measured steady-state win.
The first call uses the vendor's existing driver cache; this is not a cold-cache
setup measurement. Shape changes require a new program/session.

## All GPU settings

| N | Tile | Resident (ms) | New batch (ms) | Upload (ms) | Kernels (ms) |
| - | ---- | ------------- | -------------- | ----------- | ------------ |
| 1 | 0 | 0.220 | 0.244 | 0.025 | 0.095 |
| 1 | 8 | 0.221 | 0.246 | 0.025 | 0.105 |
| 1 | 16 | 0.220 | 22.822 | 3.188 | 1.087 |
| 16 | 0 | 0.232 | 0.297 | 0.066 | 0.101 |
| 16 | 8 | 0.218 | 0.297 | 0.066 | 0.110 |
| 16 | 16 | 0.221 | 0.295 | 0.065 | 0.106 |
| 64 | 0 | 0.222 | 0.411 | 0.172 | 0.104 |
| 64 | 8 | 0.223 | 0.413 | 0.174 | 0.114 |
| 64 | 16 | 0.219 | 0.418 | 0.175 | 0.110 |
| 256 | 0 | 0.219 | 0.958 | 0.708 | 0.119 |
| 256 | 8 | 0.219 | 0.958 | 0.715 | 0.128 |
| 256 | 16 | 0.248 | 0.946 | 0.703 | 0.124 |
| 1024 | 0 | 0.243 | 2.926 | 2.581 | 0.180 |
| 1024 | 8 | 0.252 | 2.848 | 2.522 | 0.195 |
| 1024 | 16 | 0.261 | 2.983 | 2.651 | 0.191 |
| 1936 | 0 | 3.322 | 5.067 | 4.732 | 0.208 |
| 1936 | 8 | 0.308 | 4.894 | 4.562 | 0.238 |
| 1936 | 16 | 0.285 | 4.973 | 4.649 | 0.224 |
| 8192 | 0 | 0.576 | 17.886 | 17.273 | 0.522 |
| 8192 | 8 | 0.720 | 17.407 | 16.646 | 0.667 |
| 8192 | 16 | 0.647 | 18.770 | 18.073 | 0.595 |

## All CPU settings

| N | BLAS threads | Median (ms) |
| - | ------------ | ----------- |
| 1 | 1 | 0.119 |
| 1 | 6 | 0.078 |
| 1 | 12 | 0.079 |
| 16 | 1 | 0.167 |
| 16 | 6 | 0.168 |
| 16 | 12 | 0.112 |
| 64 | 1 | 0.290 |
| 64 | 6 | 0.212 |
| 64 | 12 | 0.228 |
| 256 | 1 | 0.626 |
| 256 | 6 | 0.705 |
| 256 | 12 | 0.553 |
| 1024 | 1 | 1.558 |
| 1024 | 6 | 1.404 |
| 1024 | 12 | 1.251 |
| 1936 | 1 | 2.613 |
| 1936 | 6 | 1.953 |
| 1936 | 12 | 2.069 |
| 8192 | 1 | 11.232 |
| 8192 | 6 | 7.814 |
| 8192 | 12 | 10.003 |
