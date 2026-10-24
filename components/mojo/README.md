# xetal-gpu-mojo

The Mojo backend: from a checked XIR program to Mojo source, built as a
shared library by the Mojo compiler, loaded with `dlopen`, and run on
this machine's GPU through Mojo's `DeviceContext` (Metal on Apple
silicon, CUDA on NVIDIA, HIP on AMD; the emitter does not know which).
A second backend of the same IR beside `components/opencl`, ported from
`modular-fork/rust-demo`, the first proof that Rust can write Mojo a
person could keep. It reads the same schedule (plan A6: the element
widths), so one program and one schedule run on either backend.

```
xetal-gpu run FILE.xtl --device mojo [--time]      # lower, emit, build, run on the GPU
xetal-gpu kernel FILE.xtl --backend mojo           # the Mojo source it would build
```

## The toolchain

`scripts/mojo.sh` (`just mojo`) installs the MAX nightly pinned in
`MOJO_VERSION` with `uv` into the gitignored `work/mojo/.venv` and
links `bin/mojo`; the `max` wheel carries the Mojo compiler, its
standard library and the `max.gpu` package the kernels import. The
host finds the compiler as `XETAL_GPU_MOJO`, else `bin/mojo` of the
repository holding the tool, else `mojo` on the path, and says what
to run when none is there. On macOS the Metal toolchain is needed
(`xcodebuild -downloadComponent MetalToolchain`); on Linux an NVIDIA
driver 580 or newer, or `MODULAR_NVPTX_COMPILER_PATH` pointing at a
CUDA toolkit's `ptxas` (rust-demo's `doc/porting-arch-rtx5060.md`).
Mojo on NVIDIA goes through the CUDA driver: no CUDA is written here,
but the without-CUDA claim of this repository rests on the OpenCL
backend, and every result names its backend.

## What the emitter decides

- **Fusion.** Consecutive elementwise values (map, select, cast) of
  one length become one kernel, `stageN`, one thread per element, each
  value a variable, a buffer read as `b_name[i]` (`[0]` for a single
  value extended over the stage), a scalar constant a literal. Only
  the values an output or a later step reads are written to buffers.
- **Types.** The schedule's widths: Float as `Float32` (`Float64`
  when asked), Int as `Int64` (`Int32` when asked), Bool as `Bool` in
  registers and `Int32` 0 or 1 in buffers. `idiv` and `mod` are
  Mojo's `//` and `%` (floored, as X_eTaL's), a Float to Int cast takes
  `floor`, a number to Bool is `!= 0`.
- **Reductions.** A whole-array reduction is one kernel per operation
  and element type (`reduce_add_float32`): a block folds 256 items as
  a tree in shared memory and writes its partial; a pass per 256-fold
  until one value is left (8 items are one pass, 70,000 are three).
  The fold order differs from the evaluator's right fold, so Float
  results agree within a tolerance, Ints exactly.
- **Axes and array structure.** Axis reductions use one thread per result
  and fold a possibly strided line from the right, as the evaluator does.
  Tables, take/drop along any axis, ravel and arbitrary transposes use
  indexed reads. Inner products use one thread per output with a right
  fold over products. These are straightforward kernels; tiled matrix
  multiplication is still pending. Different array lengths split fused
  stages, with intermediate buffers carrying values between them.
- **The ABI.** The generated file exports `xir_init` (the Mojo
  runtime, 1 when the build targets a GPU), `xir_device_info`
  (`api=metal;name=Apple M1 Max`) and `xir_run(inputs, outputs)`: the
  program's inputs in order, then the constants the host uploads
  (every constant with more than one written item, so a weight matrix
  is never source text), host pointers to `f64` Float, `i64` Int, and
  `i32` Bool arrays,
  converted to the scheduled device width inside. Ints beyond 2^53
  retain every bit when the schedule selects i64. A build is cached
  under `work/mojo-cache/` by the source's hash and the compiler, so a
  program builds once (about 2.6 s on the M1 Max) and loads after.
- **Not yet:** lookup, rotate, reshape, the TopK IR operation and
  concat are refused by name, and the OpenCL backend runs them.
  The Accel top-k demo already works because its source lowers to
  axis reductions, tables and masks. The refusals are pinned in
  `tests/expected/refused.txt` so the remaining coverage is explicit.

## Checked

`tests/demos.rs` lowers every demo in `libs/*/demos/`: a taken one
has its Mojo source pinned in `tests/expected/<name>.mojo`
(`XETAL_BLESS=1 cargo test` rewrites after a reviewed change), a
refused one its reason. With the toolchain present, each taken demo
is built and run on this machine's GPU and must give the evaluator's
baseline within 1e-5; without it, the test says it skipped that part.
On the Apple M1 Max (Metal, Mojo 1.2.0.dev2026101305, 2026-10-16)
saxpy, threshold, reduce-sum, pipeline and vector-add agree with the
evaluator; a build takes about 2.6 s, a run about 3.5 ms including
the device context, uploads and downloads.

On the RTX 5060 Ti (CUDA driver 615.78.08, the same Mojo pin,
2026-10-16), the same five demos agree. Cold builds take 4.25 to
4.78 seconds; first library calls take 389 to 593 ms, repeated calls
in the same process 0.050 to 0.094 ms. These are single samples with
different startup and warmup costs, not a throughput comparison
between the cards. [The RTX report](../../docs/rtx5060-mojo-results.md)
includes outputs, total times and reproduction commands.

On the RTX 5060 Ti, 2026-10-23, eleven source demos pass: the original
five plus axis-reduce, host-data, matmul, tiny-net, topk and transpose.
The mixed host/device output of host-data is compared in source order.
Additional interpreter comparisons cover exact i64 values beyond 2^53
through bound inputs and uploaded constants, Bool outputs and tables,
floored casts, strided reductions on every axis of a rank-3 array,
singleton reductions, later-axis slices, empty results, and i32 schedules.
These are correctness checks; the October 16 timings above describe
the earlier five-demo implementation.

On the Apple M1 Max (Metal, the same Mojo pin, 2026-10-23) the same
eleven demos and the three semantics tests pass. A cold build takes
1.0 to 1.2 s (topk 1.7 s), a run 3 to 5 ms (topk 8 to 10 ms), the
whole command 50 to 60 ms with the library cached:
[the M1 Max report](../../docs/m1max-mojo-results.md).

## An example to show

The tiny network is X_eTaL source with weights read as data. Show its
evaluator output, its generated Mojo, and its GPU result:

```sh
just demo-lib Accel tiny-net
just mojo-kernel libs/Accel/demos/tiny-net.xtl
just run-mojo libs/Accel/demos/tiny-net.xtl
```

The last command reports the device and build/run times. The same source
also runs through OpenCL; no backend choice appears in the model itself.
Compilation is cached after the first build. These small examples show
correct translation and execution, not a speed advantage over the CPU.
