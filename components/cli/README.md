# xetal-gpu-cli

`xetal-gpu`, the X_eTaL-gpu tool.

```
xetal-gpu check FILE.xir                 parse and check: every value with its type and shape
xetal-gpu devices                        the OpenCL devices found (opencl:N)
xetal-gpu run FILE.xir [--device cpu|opencl:N] [SCHEDULE]   run; each output printed as xetal prints it
    [--bind NAME=1,2,3 | --bind NAME=@FILE]...     an input's items (a file: whitespace-separated)
xetal-gpu print FILE.xir                 the program in its canonical text form
xetal-gpu kernel FILE.xir [SCHEDULE]     the OpenCL C 1.2 source it would run
xetal-gpu explain FILE.xir [SCHEDULE]    the plan: buffers, kernels, launches, in words
    SCHEDULE: --float f32|f64 (f32)  --int i32|i64 (i64)  --work-group N (256)
xetal-gpu version
```

`just build` builds it in release mode (`target/release/xetal-gpu`);
`scripts/check-equiv.sh` (in the gate) runs every twin in
`libs/*/demos/` through it and compares with the evaluator's
baseline.

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
