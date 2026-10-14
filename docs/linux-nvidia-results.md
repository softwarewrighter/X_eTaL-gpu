# Linux NVIDIA source-demo results

On October 14, 2026, all 13 Accel demos passed on the NVIDIA GeForce
RTX 5060 Ti on Linux. Each `.xtl` was lowered from source and checked
against its committed evaluator baseline, on the XIR interpreter and
on the GPU with two tile settings. No source changes were needed.

## Environment

| Item | Observed value |
| ---- | -------------- |
| Repository revision tested | `6ccdbc171d9b9df58b58d9c26807be8fa6b4fbf1` |
| Pinned X_eTaL | `27b3fd0d4903e43be4b9b6e93da7476987383dcf` |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB |
| NVIDIA driver | 615.78.08 |
| OpenCL device | `opencl:0`, OpenCL 3.0 CUDA |
| Compute units | 36 |
| Maximum work-group size | 1024 |
| Double precision | Available; this run used f32 |
| Rust compiler used | `rustc 1.100.0-nightly (2e2b193f8 2026-09-02)` |
| Schedule | `schedules/nvidia-rtx5060-ti.toml` |

The schedule selects f32 Floats, i64 Ints, work-groups of 256 and
tiles of 16. The checks also override the tile size to 8.

## Reproduce

From the repository root, with the NVIDIA OpenCL device accessible:

```bash
target/release/xetal-gpu devices
RUSTUP_TOOLCHAIN=nightly just check-schedule schedules/nvidia-rtx5060-ti.toml
```

The default compiler here was Rust 1.88.0, which failed to build the
pinned X_eTaL dependency because it uses `if let` guards. The installed
nightly compiler built the current CLI successfully. The executable
present before rebuilding was older and did not accept `.xtl` source.

The sandboxed check could not see an OpenCL device and skipped the GPU.
The successful run used device access outside the sandbox; its final
summary explicitly reported one OpenCL device.

## Results

Each row passed on the interpreter and on `opencl:0`, both with the
schedule's tile size and with `--tile 8`.

| Accel demo | Output lines checked per run | Interpreter | GPU, tile 16 | GPU, tile 8 |
| ---------- | ---------------------------- | ----------- | ------------ | ----------- |
| axis-reduce | 9 | Pass | Pass | Pass |
| host-data | 7 | Pass | Pass | Pass |
| lookup | 18 | Pass | Pass | Pass |
| matmul | 8 | Pass | Pass | Pass |
| pipeline | 2 | Pass | Pass | Pass |
| reduce-sum | 5 | Pass | Pass | Pass |
| saxpy | 1 | Pass | Pass | Pass |
| threshold | 5 | Pass | Pass | Pass |
| tiny-net | 5 | Pass | Pass | Pass |
| topk | 16 | Pass | Pass | Pass |
| transpose | 46 | Pass | Pass | Pass |
| vector-add | 2 | Pass | Pass | Pass |
| windows | 21 | Pass | Pass | Pass |

The interpreter matched the evaluator baselines exactly. GPU outputs
passed `scripts/compare-out.py` with tolerance `1e-5`, as selected by
`scripts/check-equiv.sh`. The final summary was:

```text
check-equiv: 13 demos lowered from source, on cpu and 1 OpenCL device, all agree with the evaluator
```

## Scope

This verifies the current source-demo path on this NVIDIA GPU. The
`host-data` demo includes evaluator fallback for statements outside
the lowering subset; passing does not mean every statement executes
on the GPU. The references were committed evaluator baselines, not a
fresh evaluator run during this command.

This run did not measure performance, run the full pre-commit gate,
test the extension facade or world model, or validate other NVIDIA
cards. Another process held about 15.6 GB of GPU memory during the
checks. Earlier NVIDIA measurements are in
[the RTX 5060 Ti report](rtx5060-results.md) and
[the ML report](rtx5060-ml-results.md).

## Documentation validation

The report passed `sw-markdown-checker` and the repository spelling
check. A subsequent `RUSTUP_TOOLCHAIN=nightly just gate` attempt failed
in the extension test `rerun_rebind_and_invalidation` with
`OpenCL: CL_OUT_OF_HOST_MEMORY` (seven of eight session tests passed).
The full gate therefore did not pass; this does not change the
successful standalone source-demo checks above. GPU memory was heavily
occupied, but the cause of the OpenCL error was not established.
