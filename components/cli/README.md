# xetal-gpu-cli

`xetal-gpu`, the X_eTaL-gpu tool.

```
xetal-gpu check FILE.xir                 parse and check: every value with its type and shape
xetal-gpu run FILE.xir [--device cpu]    run; each output printed as xetal prints it
    [--bind NAME=1,2,3 | --bind NAME=@FILE]...     an input's items (a file: whitespace-separated)
xetal-gpu print FILE.xir                 the program in its canonical text form
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

`--device cpu` is the reference interpreter (`xetal-gpu-xir`); the
OpenCL devices (`--device opencl:N`, `xetal-gpu devices`, `kernel`,
`explain`) come with the opencl and runtime components (plan saga 1,
steps 4 and 5).
