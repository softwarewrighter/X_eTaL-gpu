# Binary file binding on the RTX 5060 Ti

Measured on large12, 2026-10-12, NVIDIA GeForce RTX 5060 Ti; ML fac586c.

Both routes bind the same f64 inputs and run the default f32 schedule.
All CLI and facade outputs agree exactly: world-model counts 1897 and 348;
the million-element sum 250000.0. Binary file creation is outside these timings.
Each route is one sample; process totals include startup and data loading.

| Case | Route | Bind (ms) | Device run (ms) | Output (ms) | Whole process (ms) |
| ---- | ----- | --------- | --------------- | ----------- | ------------------ |
| world-model | text | 460 | 202 | 0 | 1057 |
| world-model | binary | 13 | 220 | 0 | 385 |
| million | text | 411 | 192 | 0 | 1002 |
| million | binary | 10 | 219 | 0 | 416 |

File binding includes reading and decoding the binary file. Text binding
includes serialization across the bridge, but the text route's initial
file parsing is outside its bind interval (inside its process total).
This avoids text array transfer; it still allocates and uploads device
buffers on every run. Outputs still use the text bridge. Resident weights
and native binary array calls remain separate work.
