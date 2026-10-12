# Binary array files

Use binary files to bind weights and prepared batches without passing
their items through the extension's text bridge. The CLI recognizes
the header in `--bind NAME=@FILE`; the extension reads a path with
`n gp:b_indFile path`. Text bindings still work.

```bash
just build
just pack-array f64 432,64 w1.txt w1.xta
target/release/xetal-gpu run model.xir --device opencl:0 --bind w1=@w1.xta
```

The converter accepts whitespace-separated items, with an explicit
scalar (`f32`, `f64`, `i64`) and shape (dimensions separated by commas,
or `scalar`). It checks the item count and refuses an existing output
file. `f64` preserves the text input's Float precision; `f32` rounds it
to single precision. i64 values never pass through Float conversion.

In X_eTaL, with a program declaring its weight input first:

```
"gp:" u_se< "Gpu"
n := gp:l_oad []N_GET "model.xir"
n := 1 gp:b_indFile "w1.xta"
n := gp:r_un 0
gp:o_utput 1
```

The path is relative to the host process's working directory. The
file's shape must match every declared dimension. Its scalar type
must match too, except f32 storage may widen to a declared f64 input.
The device schedule still determines the GPU's arithmetic width.
Files of another scalar type are refused; this is no implicit cast.

## Format version 1

All integers and payload items are little-endian. Arrays are row-major.

| Offset | Bytes | Meaning |
| ------ | ----- | ------- |
| 0 | 8 | ASCII magic and version: `XTLARR01` |
| 8 | 1 | Scalar tag: 1 f32, 2 f64, 3 i64 |
| 9 | 1 | Rank, 0 through 9 |
| 10 | 6 | Reserved, all zero |
| 16 | 8 per axis | Dimensions, unsigned 64-bit integers |
| after dimensions | 4 or 8 per item | f32, f64 or signed i64 payload |

A scalar has rank zero and one item. Empty arrays have a zero
dimension and no payload. The reader checks the version, tag, rank,
reserved bytes, dimensions, arithmetic overflow and exact payload
length before decoding items. Truncation and extra bytes are errors.
The shared Rust implementation is `xetal_gpu_xir::binary`; it has no
OpenCL dependencies.

## Measurements and limits

`just binary-bench --require-device` prepares the pinned world model,
checks binary CLI results on the interpreter and devices, then times
text and file bindings through the same extension. It also checks a
million-element sum. `--output FILE` selects another report destination;
the default is the Arch report. The gate runs its correctness checks
with `--check`, without rewriting saved measurements.

On the RTX 5060 Ti, world-model binding fell from 460 ms to 13 ms,
and the whole process from 1057 ms to 385 ms. Million-element binding
fell from 411 ms to 10 ms. Outputs agree exactly through both routes.
See [the measurements](rtx5060-binary-results.md) for the intervals and
single-sample limits. This measures input binding and scalar outputs,
not a million-item output round trip.

Each file binding reads and decodes the array on the host. Each run
still builds kernels, allocates device buffers and uploads the inputs.
Outputs still cross the text bridge. Resident weights, compiled kernel
sessions and native binary array calls remain separate work.
