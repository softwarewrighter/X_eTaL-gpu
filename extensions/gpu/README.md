# The gpu extension

X_eTaL programs that run XIR programs on a GPU themselves: a native
extension in X_eTaL-extensions' package format (ABI V1), loaded by
its bridge host `xetal-x`. Why an extension, and when offloading pays
through the bridge, is in
[`docs/macros-and-extensions.md`](../../docs/macros-and-extensions.md)
(X1: OpenCL is a C-ABI library; products and models gain, elementwise
work does not).

| Path | What |
| ---- | ---- |
| `extension.toml` | the package: name `gpu`, ABI 1, the facade, the library's stem |
| `rust/` | the native library (`xetal-ext-gpu`, a cdylib on the pinned SDK and this repository's runtime), its own Cargo workspace, with tests through the raw ABI |
| `lib/Gpu.xtl` | the facade: the native functions with X_eTaL names and types |

```bash
just xetal-x        # the pinned X_eTaL-extensions and its xetal-x
just gpu-ext        # build the library beside xetal-x
```

## Native functions

ABI V1 functions take at most two arguments and give one result, so
the extension keeps a session per thread: load a program, bind its
inputs one by one, run it, read its outputs one by one.

| Function | X_eTaL type | What |
| -------- | ----------- | ---- |
| `devices` | `Unit -> Char` | the OpenCL devices, one line each (`opencl:N`) |
| `load` | `Char -> Int` | an XIR program, by its text: parsed and checked; the number of inputs |
| `bind` | `Num a => Int -> a -> Int` | input n (from 1, in program order) gets an array of its declared shape; an Int input takes whole numbers only |
| `schedule` | `Char -> Int` | the schedule for later runs, as the text of a `schedules/*.toml` (empty: the defaults) |
| `run` | `Int -> Int` | run on `opencl:d`, or on the reference interpreter for -1; the number of outputs |
| `output` | `Num a => Int -> a` | output i (from 1) of the last run: Ints as Ints, Floats as Floats |
| `explain` | `Unit -> Char` | the last run's plan in words |
| `kernel` | `Unit -> Char` | the last run's OpenCL C |

Errors name what is wrong: the program's line, the input and its
shape, the schedule's field, the device's limit.

## Tests

`rust/tests/session.rs` calls every function through the raw ABI, as
the loader does: the descriptor; a session on the interpreter
(a matrix-vector product and its sum); Ints returned as Ints; the
errors; and a session on device 0 with a tiled schedule, skipped
with a message when there is no device. The gate runs them with
format and clippy (`scripts/gpu-ext.sh --check`).
