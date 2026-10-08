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

## The facade

```
"gp:" u_se< "Gpu"
"ac:" u_se< "Accel"
a := 64 64 r_eshape (r_ange 4096) / 4096
c := a gp:p_roduct a                   # on opencl:0
(a ac:m_atmul a) gp:n_ear c            # ok: the evaluator agrees within f32
```

Run it with `scripts/xx run prog.xtl` (or `just run-x prog.xtl`):
`xetal-x` with this package loaded and this repository's libraries
on `XETAL_PATH`.

| Function | Type | What |
| -------- | ---- | ---- |
| `gp:d_evices @` | `Unit -> Char` | the devices |
| `gp:l_oad t` | `Char -> Int` | load an XIR program's text |
| `n gp:b_ind a` | `(Num a, Num b) => a -> b -> Int` | bind input n |
| `gp:s_chedule t` | `Char -> Int` | a schedule's TOML text, `""` for the defaults |
| `gp:r_un d` | `Num a => a -> Int` | run on `opencl:d`, -1 the interpreter |
| `gp:o_utput i` | `Num a => a -> Float` | output i as Floats |
| `gp:o_utputInts i` | `Num a => a -> Int` | output i as Ints (exact to 2^53) |
| `gp:e_xplain @` | `Unit -> Char` | the last plan in words |
| `gp:k_ernel @` | `Unit -> Char` | the last plan's OpenCL C |
| `sa gp:p_roductProgram sb` | `a -> b -> Char` | the XIR text of a product for those shapes |
| `sx gp:d_enseProgram swb` | `Int -> a -> Char` | the XIR text of a dense layer (Accel's `ac:d_ense`) |
| `a gp:p_roduct b` | `(Num a, Num b) => a -> b -> Float` | `a '+ '* i_nner b` on `opencl:0` |
| `x gp:d_ense wb` | `(Num a, Num b) => a -> b -> Float` | the layer x W + b on `opencl:0` |
| `w gp:n_ear g` | `(Num a, Num b) => a -> b -> Char` | `ok` when g agrees with w within a relative 1e-5 |

The first nine are one line each of X_eTaL-extensions' binding macro
(`"r_un : int -> int" ffi:b_ind< "gpu/run"`); the rest are ordinary
X_eTaL. The program writers are functions, not macros: an XIR
program is data handed to the device, not code for the evaluator.

Values cross the bridge as text, so a device's f32 result comes back
widened (`1.100000023841858` for 1.1): compare it with `gp:n_ear`,
not by printing it.

## Tests

`rust/tests/session.rs` calls every function through the raw ABI, as
the loader does: the descriptor; a session on the interpreter
(a matrix-vector product and its sum); Ints returned as Ints; the
errors; and a session on device 0 with a tiled schedule, skipped
with a message when there is no device. The gate runs them with
format and clippy (`scripts/gpu-ext.sh --check`).

`tests/` holds the facade's reg-rs baselines (`scripts/test-ext.sh`,
`just test-ext`): `types` pins the facade's types; `basics` runs the
program writers and sessions on the interpreter, exactly; `bad-shape`
pins the error for an input of the wrong shape; `products-device`
compares products at every rank pairing, a 64 by 64 product untiled
and in 16 by 16 tiles, and a dense layer on `opencl:0` with Accel's
results (each line `ok`). A `*-device` test is skipped, with a
message, where there is no device.
