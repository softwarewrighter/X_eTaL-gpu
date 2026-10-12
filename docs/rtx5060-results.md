# RTX 5060 Ti OpenCL results

Measured on 2026-10-12 on `large12`, Arch Linux x86_64, an Intel Xeon
W-2135 (six cores, twelve threads) and NVIDIA GeForce RTX 5060 Ti,
16 GB. The same hand-written XIR programs are checked against the
pinned X_eTaL evaluator; X_eTaL source is not compiled to GPU kernels
yet. The [Mac's ML report](ml-results.md) is preserved; the
[Arch ML report](rtx5060-ml-results.md) is separate.

## Device and setup

| Property | Observed |
| -------- | -------- |
| NVIDIA driver and OpenCL library | 615.78.08 |
| Platform | NVIDIA CUDA, OpenCL 3.0 CUDA 13.4.96 |
| Device compiler | OpenCL C 1.2; also reports OpenCL C 3.0 |
| Compute units | 36 |
| Device memory reported by OpenCL | 16707158016 bytes, 15.56 GiB |
| Largest allocation | 4176789504 bytes, 3.89 GiB |
| Work-group limit | 1024 |
| Local memory | 48 KiB |
| Preferred work-group multiple | 32 |
| Double precision | `cl_khr_fp64` available |
| Profiling timer resolution | 1000 ns |
| X_eTaL pin | 1998414 |
| X_eTaL-ML pin | fac586c (world model; initial measurements used 0ff84e8) |
| X_eTaL-extensions pin | 7bdd827 |

Arch packages `opencl-nvidia`, `ocl-icd` and `clinfo` are installed.
After the package upgrade, the old loaded driver (615.71.09) did
not match the new libraries. Rebooting loaded 615.78.08 and restored
GPU access. `nvidia-smi` and `clinfo -l` now detect the card.

This session's sandbox reports zero OpenCL platforms; outside it,
the runtime sees `opencl:0`, the RTX 5060 Ti. All GPU measurements
were run outside that sandbox. A check that skips absent hardware
is not evidence of a GPU pass.

The default Rust compiler on this host is 1.88, which cannot build
the pinned X_eTaL's `if let` match guards. Builds and checks used
the already-installed `nightly` toolchain, rustc 1.100.0-nightly
(2e2b193f8, 2026-09-02), selected with `RUSTUP_TOOLCHAIN=nightly`.
The default compiler and the vendored source were left unchanged.
[Rust 1.95](https://blog.rust-lang.org/2026/04/16/Rust-1.95.0/)
stabilized that syntax; a recent stable toolchain is also suitable
for this requirement.

## Size benchmarks

`just sizes 1024 65536 1048576 m64 m256 m512`, release builds. Each
row is one measurement, not a median. Times include process startup
and reading the input files; the GPU route also includes context
creation, kernel compilation, transfers and output. Driver caches
and first-use setup affect these totals; they do not compare raw
GPU throughput with the Mac. No GPU benchmarks ran concurrently.

The default schedule uses f32, i64, work-groups of 256 and untiled
products; the last column changes only the tile to 16. Every
interpreter output agrees exactly with the evaluator. Every GPU
output agrees within relative 1e-5 for Floats; Ints agree exactly.

| Work | Evaluator (s) | Interpreter (s) | RTX 5060 Ti (s) | RTX 5060 Ti, tile 16 (s) |
| ---- | ------------- | --------------- | --------------- | ------------------------ |
| 1024 items | 0.17 | 0.01 | 1.29 | -- |
| 65536 items | 0.21 | 0.03 | 0.55 | -- |
| 1048576 items | 2.07 | 0.14 | 0.64 | -- |
| 64 by 64 product | 0.23 | 0.00 | 0.62 | 0.50 |
| 256 by 256 product | 6.15 | 0.05 | 0.41 | 0.25 |
| 512 by 512 product | 55.12 | 0.85 | 0.41 | 0.30 |

The tiled 512 by 512 case is about 184 times faster than this
evaluator for the complete checked program. The interpreter is
also much faster than the evaluator; this is not a comparison
against an optimized CPU matrix library. Small GPU jobs can lose
because setup costs more than the computation.

## ML benchmarks

The [generated Arch report](rtx5060-ml-results.md) checks dense,
inner-wide and softmax on the evaluator, the interpreter, the GPU
and the extension bridge. These use the default schedule. The
device times, summed from profiling events, are 0.077, 1.174 and
0.551 ms respectively. The bridge's bind and output calls take
16, 15 and 26 ms. All agree with the evaluator within relative
1e-5. Kernel compilation and context setup dominate these small
jobs; the direct and facade runs happen sequentially, so driver
cache reuse can change the later run's setup cost. There is no
application kernel cache or retained GPU session yet.

The initial benchmark timings used ML 0ff84e8 and single GPU runs.
The [Arch ML report](rtx5060-ml-results.md) was regenerated after
pinning the committed model at fac586c. It reports the second GPU
run after warming up; the original size and extension measurements
on this page are unchanged.

## World model

ML fac586c includes the trained model, weights and reg-rs baseline:
432 pixels in, 64 ReLU units, 144 sigmoid outputs, with all 1,936
states as one batch and another batch of 387 held-out states.
The evaluator, f64 reference interpreter, f32 RTX kernels and
extension bridge all give exactly 1897 and 348 predicted frames.
These counts describe the model's accuracy, not numerical errors.

| Route or component | Time (ms) |
| ------------------ | --------- |
| Pinned evaluator, complete world-model demo | 15120 |
| XIR interpreter, inference and counts | 147 |
| RTX kernels, profiling events | 0.237 |
| RTX launches, host interval | 0.46 |
| RTX runtime total | 158 |
| RTX CLI whole process | 317 |
| Extension: bind arrays through text | 458 |
| Extension: device run | 197 |
| Extension: output counts | 0 (rounded to milliseconds) |

The CLI GPU sample is the second of two separate process runs.
The extension is a separate call, not part of that CLI timing;
binding and running together take about 655 ms. Text binding costs
more than twice the device run, supporting the planned binary-array
bridge and resident-weight session work.

The evaluator runs the whole demo, including its windows, displays
and dream rollout. The twin runs only batched inference and the two
counts using prepared input files. Its process time includes reading
those files but excludes generating the windows. These are useful
route costs, not an identical-work speedup comparison or an optimized
CPU library baseline. Each timing is one sample, not a median.

## End-to-end extension demo

`just offload` times the operations inside one X_eTaL program using
the extension host's evaluator. Both result comparisons print `ok`.
The product input and outputs differ from the size benchmark above,
so these totals are not directly comparable with that table.

| Work | Evaluator (ms) | GPU through the text bridge (ms) |
| ---- | -------------- | -------------------------------- |
| 512 by 512 product | 24994 | 979 |
| Million-item elementwise pipeline and sum | 254 | 1440 |

The product is about 25 times faster through the extension. A later
breakdown of the product call measured load and bind at 476 ms, the
device run at 148 ms and output at 204 ms; those are a separate run,
not components to add up to the earlier 979 ms. The pipeline is
about 5.7 times slower through the text bridge. Compute-heavy
products benefit; this elementwise workload does not.

## Reproduce

Run with access to the GPU, not in a sandbox that hides it:

```bash
export RUSTUP_TOOLCHAIN=nightly  # this host's installed newer compiler
clinfo -l
just build
target/release/xetal-gpu devices
just gate
just sizes 1024 65536 1048576 m64 m256 m512
just ml-bench --require-device --output docs/rtx5060-ml-results.md
just offload
just check-schedule schedules/nvidia-rtx5060-ti.toml
just check-schedule schedules/nvidia-rtx5060-ti-f64.toml 1e-12
target/release/xetal-gpu run libs/Accel/demos/pipeline.xir --device opencl:0 --schedule schedules/nvidia-rtx5060-ti.toml --time
target/release/xetal-gpu run libs/Accel/demos/pipeline.xir --device opencl:0 --schedule schedules/nvidia-rtx5060-ti-f64.toml --time
```

The specific schedules use work-groups of 256 and 16 by 16 tiles,
with f32 or f64 respectively. The shared `nvidia-modern` templates
remain untested on the other cards. The model runner currently uses
the default f32 schedule; additional f64 model timing and repeated
inference with resident weights remain separate work.

The shared modern template's work-group size of 1024 was refused
on this card: although the device permits 1024, its compiled
`reduce_add_float` kernel permits only 256. The runtime correctly
reports that per-kernel limit. The RTX schedules use 256 instead.

## Validation

The full `just gate` passed with access to the actual GPU: library
baselines and types, extension tests, component formatting, clippy
with warnings denied, unit tests, all eight Accel demo twins (both
default and tile 8 schedules), documentation examples, site build,
smaller generated sizes, ML equivalences, spelling and Markdown.
No GPU checks were skipped.

Both RTX-specific schedule files were also checked on all eight
Accel twins and the six generated sizes above, fourteen cases per
schedule. f32 agrees within relative 1e-5; f64 agrees within relative
1e-12; Ints agree exactly. The comparison utility also accepts an
absolute Float error up to 1e-9, including values near zero.
No runtime or emitter changes were needed.
