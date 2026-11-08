# xetal-gpu-cli

`--bind NAME=@FILE` accepts text or a checked binary array file.
`xetal-gpu pack f64 2,3 INPUT.txt OUTPUT.xta` converts existing text
weights; it refuses to replace an existing output. f32 and i64 are
also supported. See [binary arrays](../../docs/binary-arrays.md).

`xetal-gpu`, the X_eTaL-gpu tool.

```
xetal-gpu check FILE.xir                 parse and check: every value with its type and shape
xetal-gpu devices                        OpenCL devices (opencl:N) and the default Mojo GPU
xetal-gpu run FILE.xir [--device cpu|opencl:N|mojo] [SCHEDULE]   run; each output printed as xetal prints it
    [--bind NAME=1,2,3 | --bind NAME=@FILE]...     an input's items (a file: whitespace-separated)
    [--repeat N] [--refresh NAME]...          reuse a session; optionally reupload named inputs each repeat
    [--time]                                 on a device: where the time went (setup, upload, kernels as the device timed them, launches, download), to standard error
xetal-gpu print FILE.xir                 the program in its canonical text form
xetal-gpu kernel FILE.xir [SCHEDULE]     the OpenCL C 1.2 source it would run
xetal-gpu explain FILE.xir [SCHEDULE]    the plan: buffers, kernels, launches, in words
    SCHEDULE: --float f32|f64 (f32)  --int i32|i64 (i64)  --work-group N (256)
xetal-gpu version
```

`just build` builds it in release mode (`target/release/xetal-gpu`);
`scripts/check-equiv.sh` (in the gate) runs every demo in
`libs/*/demos/` through it, lowered from its source (`.xtl` files
are lowered by `components/lower` first; `lower FILE.xtl` prints the
XIR), and compares with the evaluator's baseline. A statement the
device cannot take is evaluated by the pinned `xetal` and bound as
an input (`lower` says so on standard error), or printed in its
place; `explain FILE.xtl` ends with the X_eTaL line each kernel came
from.

`--device mojo` uses the same source and schedule through the Mojo
backend. Its `--time` separates build, per-kernel device intervals, call
and total time; repeated calls keep the library loaded but upload every
input and allocate buffers each time. The per-launch timers synchronize
and add profiling overhead. OpenCL repeats retain device buffers and
upload only refreshed inputs. See [Mojo](../mojo/README.md).

```
$ xetal-gpu check libs/Accel/demos/reduce-sum.xir
%a : i64 [8] = const (8 items)
%two : i64 [] = const (1 item)
%b : i64 [8] = map mul %two %a
...
$ xetal-gpu run libs/Accel/demos/reduce-sum.xir
108
24
408
4.5
3.1875
```

```
$ xetal-gpu explain libs/Accel/demos/pipeline.xir
schedule: Float as float, Int as long, work-group 256
buffers:
  b_x: long 8 items, %x i64 [8]; constant, uploaded by the host
  ...
kernels:
  k0: 8 elements, one work-item each; computes %xw %s %pos; writes %pos
  reduce_add_long: a tree reduction in local memory, one partial per work-group; computes %sum
  ...
launches:
  1: k0(b_x, b_w, b_pos) global 8 local 8: 8 elements (1 group of 8)
  2: reduce_add_long(b_pos, b_sum, 8, local 2048 bytes) global 256 local 256: %sum: add 8 items to 1 partial (pass 1)
  ...
```

```
$ xetal-gpu devices
opencl:0  Apple M1 Max (GPU, Apple; OpenCL 1.2), 32 compute units, work-groups to 256, 53084 MB, no fp64
$ xetal-gpu run libs/Accel/demos/pipeline.xir --device opencl:0
32
16.625
```

`--device cpu` is the reference interpreter (`xetal-gpu-xir`);
`kernel` and `explain` are the emitter (`xetal-gpu-opencl`);
`devices` and `--device opencl:N` are the runtime
(`xetal-gpu-runtime`), which takes the same schedule flags.

Use xetal-gpu memory FILE.xir to report logical/physical buffer counts,
planned payload bytes, launches and guards without allocating GPU memory.
The just ray-memory recipe builds the CLI and defaults to the ray diagnostic.


## Resident frame batches

`lower-function FILE --param NAME:TYPE[:DIMS] --uses IMPORTS --regions`
exposes the existing typed-function lowering API. Parameters remain runtime
inputs, rather than numeric file reads that can fold into constants. DIMS
is comma-separated; omit it for a scalar. No language syntax is added.

`run FILE.xir --bind frameStart=0 --repeat 31 --advance frameStart=6
--save-repeats DIR --device opencl:0 --time` prepares a scene once, advances
the bound scalar Int by six between executions, uploads only changed
inputs and saves every repeat as batch-NNNN.out. Each execution can render
several frames in one mapped region. CPU supports the same progression;
Mojo refuses these batch options explicitly. Duplicate/non-Int/non-scalar
inputs and Int overflow fail. Timings include saved-output serialization;
only the first repeat includes GPU setup. ASCII output is currently a
host-side throughput cost, not GPU-to-GPU traffic.
