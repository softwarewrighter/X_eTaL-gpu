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
| `lib/Gpu.xtlm` | the accelerate macro, `gp:a_ccel<` |

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
| `bind_file` | `Int -> Char -> Int` | input n gets a checked binary array read from a host file path |
| `schedule` | `Char -> Int` | the schedule for later runs, as the text of a `schedules/*.toml` (empty: the defaults) |
| `run` | `Int -> Int` | run on `opencl:d`, or on the reference interpreter for -1; the number of outputs |
| `rerun` | `Int -> Int` | reuse compiled kernels and device buffers; upload inputs rebound since the last run |
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
| `n gp:b_indFile path` | `Num a => a -> Char -> Int` | bind input n from a binary array file; only the path crosses the bridge |
| `gp:s_chedule t` | `Char -> Int` | a schedule's TOML text, `""` for the defaults |
| `gp:r_un d` | `Num a => a -> Int` | run on `opencl:d`, -1 the interpreter |
| `gp:r_erun d` | `Num a => a -> Int` | reuse the session on this device, uploading changed bindings |
| `gp:o_utput i` | `Num a => a -> Float` | output i as Floats |
| `gp:o_utputInts i` | `Num a => a -> Int` | output i as Ints (exact to 2^53) |
| `gp:e_xplain @` | `Unit -> Char` | the last plan in words |
| `gp:k_ernel @` | `Unit -> Char` | the last plan's OpenCL C |
| `sa gp:p_roductProgram sb` | `a -> b -> Char` | the XIR text of a product for those shapes |
| `sx gp:d_enseProgram swb` | `Int -> a -> Char` | the XIR text of a dense layer (Accel's `ac:d_ense`) |
| `a gp:p_roduct b` | `(Num a, Num b) => a -> b -> Float` | `a '+ '* i_nner b` on `opencl:0` |
| `x gp:d_ense wb` | `(Num a, Num b) => a -> b -> Float` | the layer x W + b on `opencl:0` |
| `w gp:n_ear g` | `(Num a, Num b) => a -> b -> Char` | `ok` when g agrees with w within a relative 1e-5 |

The native bindings are one line each of X_eTaL-extensions' binding macro
(`"r_un : int -> int" ffi:b_ind< "gpu/run"`); the rest are ordinary
X_eTaL. The program writers are functions, not macros: an XIR
program is data handed to the device, not code for the evaluator.

Values cross the bridge as text, so a device's f32 result comes back
widened (`1.100000023841858` for 1.1): compare it with `gp:n_ear`,
not by printing it.

For large inputs, `gp:b_indFile` avoids that text transfer by reading
a checked binary array directly. On the RTX 5060 Ti the world model's
input binding took 13 ms from files versus 460 ms as text, with the
same exact counts. `gp:r_un` uploads each run; `gp:r_erun` keeps buffers resident and
uploads only changed bindings. Loading another program, changing the
schedule or switching devices rebuilds the session.
See [binary arrays](../../docs/binary-arrays.md) for conversion and
the file format, and [the report](../../docs/rtx5060-binary-results.md).

## The accelerate macro

`lib/Gpu.xtlm` (imported with the facade: `"gp:" u_se< "Gpu"` brings
both files under one alias) defines `gp:a_ccel<`, the user's
framework: a macro hands a function's text to a runner. MC10 makes a
macro a function from source text to source text, so

```
u:d_ense := "float float" gp:a_ccel< "{ x wb -> ac:r_elu x ac:d_ense wb }"
```

expands (`xetal expand`, pinned in `tests/expand-accel-macro-device.out`) to

```
u:d_ense := ({ x wb -> gp:o_utput 1 gp:a_fter gp:r_un 0 gp:a_fter 2 gp:b_ind wb gp:a_fter 1 gp:b_ind x gp:a_fter "{ x wb -> ac:r_elu x ac:d_ense wb }" gp:l_ower "FILE|float float" })
```

Read from the right, as X_eTaL evaluates: the function's text goes to
the runner (`gp:l_ower`, with the file it is written in, whose
`u_se<` lines are its imports, and the arguments' kinds), the
arguments are bound, the program runs on `opencl:0`, the result is
read back. `gp:a_fter` returns its left after its right, so nothing
is bound that is never read. The runner (the native library) lowers
the text through `components/lower` for the arguments' shapes when
they are bound, once per set of shapes, and keeps the session;
`xetal-x`'s X_eTaL evaluates everything else. The same macro with a
runner that offers load-from-source, bind and run would accelerate
on anything.

The kinds say how each argument is taken (`"float"` or `"int"` per
parameter; `""` for all Floats); an Int array a function looks up by
(`ac:e_mbed`) is bound as `int`. The function is one or two named
arrays to one array, on one line, without quotes.
`tests/accel-macro-device.xtl` runs a dense layer at two shapes
(the second relowers, a repeat reuses), an embedding lookup with an
int argument, and softmax of attention scores, each `ok` against the
evaluator's own value. `demos/accelerate.xtl` is the same in demo
form, three functions used and checked, and the live site's landing
page shows each of its definitions beside what `xetal expand` makes
of it, generated from the pinned `xetal-x` at build time.

## The offload demo

`demos/offload.xtl` (`just offload`) computes the same work with the
evaluator and through this extension on `opencl:0`, checks they
agree, and times both. On the Apple M1 Max:

| Work | Evaluator on the Mac | Apple M1 Max through the extension | |
| ---- | -------------------- | ---------------------------------- | - |
| 512 by 512 product | 14545 ms | 462 ms | 31 times faster |
| a million items, `x + x * w`, positive parts summed | 188 ms | 2447 ms | 13 times slower |

On the RTX 5060 Ti in the Arch Linux machine (`docs/rtx5060-results.md`),
the same demo: the product 979 ms against 24994 ms in that host's
evaluator (25 times faster), the million items 1440 ms against 254 ms
(5.7 times slower).

Of the product's 462 ms on the GPU side, loading the program and
binding the two inputs took 289 ms, the run 12 ms (compiling the
kernels included), reading the result back 127 ms: the bridge's text
is nearly all of it. That is the prediction of
`docs/macros-and-extensions.md`: products gain even through the
bridge, elementwise work loses, and a native hook with binary arrays
(ask G8) would remove most of the cost. The demo prints its timings,
so it has no baseline; `tests/offload-device.xtl` pins what holds on
any machine with a GPU (the results agree, and a 256 by 256 product
is faster on the GPU).

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
results (each line `ok`); `offload-device` checks the GPU's 256 by
256 product agrees and is faster than the evaluator's. The demos run
with `XETAL_DEMOS=1 scripts/test-ext.sh`. A `*-device` test is skipped, with a
message, where there is no device.
