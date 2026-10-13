# Repeated inference on the Apple M1 Max

Measured 2026-10-12 on `max` (Apple M1 Max).
NumPy 2.5.4; accelerate (NumPy's wheel; its own threading, not limited here). ML fac586c; X_eTaL 27b3fd0.

The trained world model runs both dense layers, biases, ReLU, sigmoid
and exact-frame counts. The two batches contain N states and min(N, 387)
held-out states. Larger batches cycle the committed data; this is a throughput
test of a trained model, not additional unseen evaluation data.

Each setting runs 21 times; report medians after discarding the first three.
CPU: NumPy's BLAS as it is. GPU: the fastest refreshed
run of tiles 0, 8 and 16. All settings give the same exact counts as the
reference interpreter; the full batch gives the evaluator baseline 1897/348.

This model does not beat optimized CPU inference with refreshed inputs at
any measured batch size. Resident-only timings isolate execution reuse,
but repeating identical inputs is not a practical throughput speedup.

| Batch N + held | Interpreter (ms) | CPU BLAS (ms), threads | Resident (ms) | New batch (ms) | Tile | Setup (ms) | First call (ms) | Stream speedup | Break-even calls |
| -------------- | ---------------- | ---------------------- | ------------- | -------------- | ---- | ---------- | --------------- | -------------- | ---------------- |
| 1 + 1 | 0.081 | 0.022, as is | 1.348 | 1.539 | 16 | 58.4 | 65.2 | 0.01x | -- |
| 16 + 16 | 1.141 | 0.043, as is | 1.337 | 1.834 | 16 | 59.6 | 66.7 | 0.02x | -- |
| 64 + 64 | 4.413 | 0.100, as is | 1.430 | 1.785 | 8 | 60.0 | 68.5 | 0.06x | -- |
| 256 + 256 | 17.627 | 0.307, as is | 1.668 | 1.777 | 16 | 74.3 | 88.3 | 0.17x | -- |
| 1024 + 387 | 48.738 | 0.782, as is | 1.744 | 2.099 | 8 | 57.6 | 69.5 | 0.37x | -- |
| 1936 + 387 | 80.662 | 1.288, as is | 1.595 | 2.514 | 8 | 65.9 | 79.2 | 0.51x | -- |
| 8192 + 387 | 297.030 | 4.676, as is | 3.204 | 5.498 | 8 | 58.9 | 80.2 | 0.85x | -- |

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
| 1 | 0 | 1.467 | 1.607 | 0.414 | 0.011 |
| 1 | 8 | 1.234 | 1.824 | 0.432 | 0.020 |
| 1 | 16 | 1.348 | 1.539 | 0.388 | 0.007 |
| 16 | 0 | 1.257 | 1.956 | 0.457 | 0.027 |
| 16 | 8 | 1.440 | 1.857 | 0.471 | 0.021 |
| 16 | 16 | 1.337 | 1.834 | 0.459 | 0.020 |
| 64 | 0 | 1.559 | 2.061 | 0.479 | 0.029 |
| 64 | 8 | 1.430 | 1.785 | 0.467 | 0.014 |
| 64 | 16 | 1.463 | 1.885 | 0.503 | 0.021 |
| 256 | 0 | 1.714 | 1.913 | 0.589 | 0.014 |
| 256 | 8 | 1.728 | 2.058 | 0.570 | 0.022 |
| 256 | 16 | 1.668 | 1.777 | 0.553 | 0.009 |
| 1024 | 0 | 1.807 | 2.325 | 0.784 | 0.030 |
| 1024 | 8 | 1.744 | 2.099 | 0.751 | 0.019 |
| 1024 | 16 | 2.721 | 2.469 | 0.799 | 0.037 |
| 1936 | 0 | 3.721 | 3.075 | 1.049 | 0.046 |
| 1936 | 8 | 1.595 | 2.514 | 0.903 | 0.028 |
| 1936 | 16 | 1.673 | 2.786 | 1.039 | 0.036 |
| 8192 | 0 | 4.579 | 6.921 | 2.253 | 0.153 |
| 8192 | 8 | 3.204 | 5.498 | 2.234 | 0.094 |
| 8192 | 16 | 3.758 | 6.275 | 2.298 | 0.126 |

## All CPU settings

| N | BLAS threads | Median (ms) |
| - | ------------ | ----------- |
| 1 | as is | 0.022 |
| 16 | as is | 0.043 |
| 64 | as is | 0.100 |
| 256 | as is | 0.307 |
| 1024 | as is | 0.782 |
| 1936 | as is | 1.288 |
| 8192 | as is | 4.676 |
