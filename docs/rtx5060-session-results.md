# Repeated inference on the RTX 5060 Ti

Measured 2026-10-12 on `large12` (NVIDIA GeForce RTX 5060 Ti).
NumPy 2.5.4; OpenBLAS 0.3.34.106.0, SkylakeX. ML fac586c; X_eTaL 1998414.

The trained world model runs both dense layers, biases, ReLU, sigmoid
and exact-frame counts. The two batches contain N states and min(N, 387)
held-out states. Larger batches cycle the committed data; this is a throughput
test of a trained model, not additional unseen evaluation data.

Each setting runs 21 times; report medians after discarding the first three.
CPU: the fastest of 1, 6 and 12 OpenBLAS threads. GPU: the fastest refreshed
run of tiles 0, 8 and 16. All settings give the same exact counts as the
reference interpreter; the full batch gives the evaluator baseline 1897/348.

This model does not beat optimized CPU inference with refreshed inputs at
any measured batch size. Resident-only timings isolate execution reuse,
but repeating identical inputs is not a practical throughput speedup.

| Batch N + held | Interpreter (ms) | OpenBLAS (ms), threads | Resident (ms) | New batch (ms) | Tile | Setup (ms) | First call (ms) | Stream speedup | Break-even calls |
| -------------- | ---------------- | ---------------------- | ------------- | -------------- | ---- | ---------- | --------------- | -------------- | ---------------- |
| 1 + 1 | 0.260 | 0.073, 12 | 0.187 | 0.206 | 0 | 160.9 | 163.5 | 0.35x | -- |
| 16 + 16 | 1.786 | 0.083, 12 | 0.188 | 0.230 | 8 | 158.1 | 160.6 | 0.36x | -- |
| 64 + 64 | 6.845 | 0.226, 6 | 0.188 | 0.296 | 0 | 161.3 | 164.0 | 0.76x | -- |
| 256 + 256 | 28.782 | 0.474, 12 | 0.188 | 0.659 | 16 | 158.4 | 162.5 | 0.72x | -- |
| 1024 + 387 | 78.742 | 1.060, 12 | 0.213 | 1.688 | 0 | 157.5 | 166.7 | 0.63x | -- |
| 1936 + 387 | 127.216 | 1.940, 12 | 0.253 | 2.974 | 0 | 149.3 | 162.4 | 0.65x | -- |
| 8192 + 387 | 451.161 | 8.641, 6 | 0.564 | 10.532 | 0 | 145.0 | 177.8 | 0.82x | -- |

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
| 1 | 0 | 0.187 | 0.206 | 0.020 | 0.091 |
| 1 | 8 | 0.188 | 0.209 | 0.020 | 0.104 |
| 1 | 16 | 0.186 | 0.208 | 0.020 | 0.101 |
| 16 | 0 | 0.188 | 0.239 | 0.043 | 0.098 |
| 16 | 8 | 0.188 | 0.230 | 0.042 | 0.104 |
| 16 | 16 | 0.189 | 0.237 | 0.043 | 0.104 |
| 64 | 0 | 0.188 | 0.296 | 0.094 | 0.102 |
| 64 | 8 | 0.189 | 0.298 | 0.095 | 0.110 |
| 64 | 16 | 0.189 | 0.297 | 0.094 | 0.107 |
| 256 | 0 | 0.186 | 0.666 | 0.456 | 0.116 |
| 256 | 8 | 0.193 | 0.681 | 0.471 | 0.127 |
| 256 | 16 | 0.188 | 0.659 | 0.446 | 0.123 |
| 1024 | 0 | 0.213 | 1.688 | 1.414 | 0.172 |
| 1024 | 8 | 0.233 | 1.710 | 1.435 | 0.188 |
| 1024 | 16 | 0.225 | 1.690 | 1.421 | 0.181 |
| 1936 | 0 | 0.253 | 2.974 | 2.694 | 0.203 |
| 1936 | 8 | 0.290 | 2.992 | 2.671 | 0.241 |
| 1936 | 16 | 0.276 | 3.061 | 2.752 | 0.229 |
| 8192 | 0 | 0.564 | 10.532 | 9.934 | 0.520 |
| 8192 | 8 | 0.720 | 10.849 | 10.104 | 0.664 |
| 8192 | 16 | 0.631 | 10.976 | 10.306 | 0.590 |

## All CPU settings

| N | OpenBLAS threads | Median (ms) |
| - | ---------------- | ----------- |
| 1 | 1 | 0.114 |
| 1 | 6 | 0.098 |
| 1 | 12 | 0.073 |
| 16 | 1 | 0.126 |
| 16 | 6 | 0.123 |
| 16 | 12 | 0.083 |
| 64 | 1 | 0.280 |
| 64 | 6 | 0.226 |
| 64 | 12 | 0.237 |
| 256 | 1 | 0.614 |
| 256 | 6 | 0.621 |
| 256 | 12 | 0.474 |
| 1024 | 1 | 1.361 |
| 1024 | 6 | 1.460 |
| 1024 | 12 | 1.060 |
| 1936 | 1 | 2.719 |
| 1936 | 6 | 2.042 |
| 1936 | 12 | 1.940 |
| 8192 | 1 | 13.094 |
| 8192 | 6 | 8.641 |
| 8192 | 12 | 10.845 |
