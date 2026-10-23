# Mojo on the RTX 5060 Ti

Measured October 16, 2026 on Linux host `large12`, starting from
repository revision `e900a58`, with the `run-mojo` argument-forwarding
fix included in this report's commit. All five supported Accel demos
lowered from X_eTaL source, emitted Mojo, built as shared libraries,
and matched the committed evaluator baselines within `1e-5`.

## Environment

| Item | Value |
| ---- | ----- |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB |
| NVIDIA driver | 615.78.08 |
| Device reported by every timed run | `api=cuda;name=NVIDIA GeForce RTX 5060 Ti` |
| MAX pin (`MOJO_VERSION`) | `26.7.0.dev2026101305` |
| `just mojo --check` | `Mojo 1.2.0.dev2026101305 (bb6fedb4)` |
| Pinned X_eTaL | `27b3fd0` |
| Rust toolchain | installed `nightly`, Rust 1.100.0-nightly |

Mojo uses the CUDA driver on this NVIDIA device. The separate OpenCL
backend uses the NVIDIA OpenCL driver. No `ptxas` override was needed
with this driver, and no other GPU workload was running at the start.
The default schedule uses f32 Floats and i64 Ints on the device;
this backend version carries host ABI items as f64, so these demos
are not a validation of Ints beyond 2^53.

## Reproduce

```sh
just mojo
just mojo --check
cd components
RUSTUP_TOOLCHAIN=nightly cargo test -p xetal-gpu-mojo -- --nocapture
cd ..
```

The crate's three unit tests and its demo integration test passed.
The integration test printed `ok: <demo> on the GPU through Mojo` for
all five demos and checked the other eight demos' pinned refusals.

For the timing table, a separate, initially empty cache kept the test
builds from hiding compilation costs:

```sh
for demo in saxpy threshold reduce-sum pipeline vector-add; do
  RUSTUP_TOOLCHAIN=nightly XETAL_GPU_MOJO_CACHE="$PWD/work/mojo-rerun-20261016/cache" \
    just run-mojo "libs/Accel/demos/$demo.xtl" --repeat 2
done
```

Use a new empty cache directory to repeat the cold builds. Reusing
this directory measures cached builds instead. The `run-mojo` recipe
now removes its file parameter before forwarding the remaining CLI
arguments; previously it passed the filename twice and failed.

## Build and run times

All times are milliseconds, one sample per call, not medians. Each
demo runs twice in one process. Build time measures the compiler;
run time measures the library call, including its device context,
uploads, kernels and downloads. Total also includes building, loading
and initializing the library, querying the device and host work in
this execution branch; it excludes source lowering and process startup.
The second call benefits from process and driver warmup as well as
the build cache; it is not cold-start latency or kernel-only time.

| Demo | First build (ms) | First run (ms) | First total (ms) | Second build (ms) | Second run (ms) | Second total (ms) |
| ---- | ---------------- | -------------- | ---------------- | ----------------- | --------------- | ----------------- |
| saxpy | 4250.354 | 407.527 | 5817.872 | 0.000 | 0.050 | 0.249 |
| threshold | 4373.201 | 388.821 | 5797.241 | 0.000 | 0.071 | 0.282 |
| reduce-sum | 4779.796 | 431.453 | 6168.516 | 0.000 | 0.094 | 0.324 |
| pipeline | 4609.646 | 392.767 | 6167.860 | 0.000 | 0.074 | 0.289 |
| vector-add | 4249.280 | 593.080 | 5769.169 | 0.000 | 0.058 | 0.269 |

## Outputs

These are the CLI outputs, checked against `libs/Accel/tests/demo-*.out`
with `scripts/compare-out.py` at tolerance `1e-5`. The crate's pinned
source and refusal tests also passed; no baselines were reblessed.

### saxpy

```text
1.5 1.5 1.75 2.125 2.5 2.875 3.25 3.5
```

### threshold

```text
-0.75 -0.5 -0.25 0.0 0.25 0.5 0.75 1.0
-0.0 -0.0 -0.0 0.0 0.0 0.0 0.75 1.0
0.0 0.0 0.0 0.0 0.25 0.5 0.75 1.0
0 0 0 0 1 2 3 4
-3 0 -1 0 1 0 3 0
```

### reduce-sum

```text
108
24
408
4.5
3.1875
```

### pipeline

```text
32
16.625
```

### vector-add

```text
11 22 33 44 55 66 77 88
0.375 0.75 1.125 1.5 1.875 2.25 2.625 3.0
```

## Other Arch checks

The full gate passed before installing Mojo, with its absence reported.
It passed again after installation, including the Mojo device tests,
with `RUSTUP_TOOLCHAIN=nightly PYTHONUNBUFFERED=1 just gate`.
The eight source ML workloads also passed in the timed rerun; see
[the refreshed RTX ML report](rtx5060-ml-results.md). The denoiser's
first self-check line remains interpreter-only because its requested
`1e-6` tolerance is tighter than the f32 path meets; its other 151 lines
passed on the GPU. The GPT batch retains host evaluation for its loop.

The [binary binding report](rtx5060-binary-results.md) and
[session report](rtx5060-session-results.md) use the fixed world-model
fixture, not generated example programs. Counts remained `1897/348`.
Binary binding reduced the world-model bind interval from 460 to
15 ms. The refreshed-input session sweep still did not beat OpenBLAS
at any tested batch size. These OpenCL measurements are separate from
the Mojo timings above.

## Expanded source coverage, October 23, 2026

The backend now runs eleven Accel demos through the CUDA driver on this
RTX 5060 Ti. In addition to the five measured above, axis-reduce,
host-data, matmul, tiny-net, topk and transpose match their committed
evaluator baselines. Float comparisons use 1e-5 tolerance; integer
comparisons are exact. Host-data includes evaluator-produced input and
output, merged with device output in source order.

Axis folds, tables, slices, ravel, untiled inner products and transposes
are generated from XIR. The host ABI now carries Ints as i64, Float as
f64 and Bool as i32; no integer passes through a Float. Additional device
regressions compare with the IR interpreter, including integers above
2^53, i32 schedules, rank-3 strided folds, Bool tables, floored casts,
empty slices and singleton reductions. The Mojo and X_eTaL pins and the
NVIDIA driver are unchanged from the environment above. No timings for
these six added demos were collected; the earlier table remains dated.

```sh
cd components
cargo test -p xetal-gpu-mojo -- --nocapture
```

Only lookup and windows remain refused among the thirteen Accel demos.
Mac verification of the added coverage is pending.
