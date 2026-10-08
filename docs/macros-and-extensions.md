# Macros and extensions in X_eTaL-gpu

Where this repository should use X_eTaL's two extension mechanisms
beyond ordinary `.xtl` libraries, macro libraries (`.xtlm`) and
native extensions (X_eTaL-extensions' ABI V1), and where it should
not. Written 2026-10-07 against X_eTaL 75e6a5c (this repo's pin),
X_eTaL-libraries, X_eTaL-ML and X_eTaL-extensions as they are on
that date.

## The rule

A macro or an extension needs a justification. Ordinary X_eTaL is
the default; each candidate below is judged by three tests, and one
must hold:

| Test | Holds when | Mechanism it can justify |
| ---- | ---------- | ------------------------ |
| (a) concision | the thing cannot easily be expressed concisely in `.xtl` code | a macro (new notation that expands to ordinary X_eTaL) |
| (b) speed | it would run too slowly in `.xtl` code, measured, at the sizes that matter | an extension (native code behind a typed facade) |
| (c) a foreign library | X_eTaL has no way to reach a third-party library with a C ABI | an extension |

Two more rules follow from what a macro is (X_eTaL MC10, MC23): a
macro is a function from the source text left and right of its call
to new source, run before the program is type-checked. So:

- **A macro is only for producing X_eTaL code.** When the result is
  data (an XIR program sent to a device, a kernel's text, a report),
  an ordinary function computes it at run time, and test (a) is
  answered by that function, not by a macro.
- **A macro sees text, never types or shapes.** It cannot know that
  `x` is a Float vector of 1024 items; anything that depends on that
  waits for ask G3 (static shapes) or happens at run time.

## Verdicts

| # | Candidate | Kind | Test | Verdict |
| - | --------- | ---- | ---- | ------- |
| X1 | The GPU runtime as a native extension, `Gpu` (`gp:`): an X_eTaL program sends an XIR program and its input arrays to an OpenCL device and reads the outputs back | extension | (c) yes; (b) yes for products and models, no for elementwise work (measured below) | **do it**: a new saga, gpu-extension, here; testable on this Mac now |
| X2 | Its facade written by `ffi:b_ind<` (X_eTaL-extensions' `Ffi.xtlm`) | existing macro | (a) already justified there: one signature line per native function instead of five lines of channel code | **reuse**, do not write another |
| X3 | Linear algebra (X_eTaL-extensions' planned `linalg`, nalgebra) as the CPU reference for products | extension | (b) would hold, but the reference must be the evaluator (plan A2) | **no**: it would replace the specification with another implementation |
| X4 | XIR text from an Accel expression or a model spec (`"784 128 relu 10 softmax"`, X_eTaL-ML's Net grammar) | function | (a) holds against writing XIR by hand, but the result is data | **an ordinary `.xtl` function** in the `Gpu` library's X_eTaL half, not a macro (saga 3 and the model saga) |
| X5 | An offload macro: `"x w" gp:o_ffload< "x ac:p_ipeline w"` writes the CPU call and the GPU call of one expression, so it is written once | macro | (a) partly: the expression is written once instead of twice; but the macro must parse X_eTaL text and cannot see types (G3), and it duplicates the lowering X_eTaL owns (G2) | **defer**: ask X_eTaL first; revisit only if G2 stays unplanned after X1 lands, and then for a closed grammar (a chain of `ac:` calls on named arrays) |
| X6 | Choosing the device at compile time | existing system macro | none needed: `@ c_fg< "gpu"` with `xetal --cfg gpu` (MC25) already gives a program a compile-time fact | **use `c_fg<`**; write no macro |
| X7 | Schedules (work-group, widths, tiles) written in the source | macro | fails A6 and research7 section 11: the schedule is data, not semantics | **no** |
| X8 | The Accel exports as macros | macro | (a) fails: each export is one short expression (`{ x w -> l:s_um l:r_elu x + x * w }`) | **no** |
| X9 | Table-driven equivalence tests | existing macro | (a) already justified in X_eTaL-libraries' Check: `k:c_ases<` names each case by its own text | **reuse** when a test program needs named cases (pin X_eTaL-libraries as X_eTaL-ML does); not needed yet |
| X10 | Writing the `.xir` twins by macro | macro | a macro gives source to the program, not files; the twins are data | **no**: the twins go when G2 lands; until then X4's function can write the simple ones |

In short: one extension (X1) is justified now, on test (c) and, for
the work that matters, test (b); no new macro is justified now; two
existing macros (`ffi:b_ind<`, `c_fg<`) and possibly a third
(`k:c_ases<`) are used as they are.

## What exists in the peer repositories

### Macros

| Where | Macros | Their justification |
| ----- | ------ | ------------------- |
| X_eTaL, `lib/System.xtlm` (no import) | `i_f<`, `u_nless<`, `e_ach<`, `f_ormat<`, `d_bg<`, `a_ssert<`, `p_anic<`, `t_odo<`, and the compiler-only `i_nclude<`, `c_fg<`, `f_ile<`, `l_ine<`, `e_rror<` | what only the compiler knows, or what nearly every program uses (MC24) |
| X_eTaL-libraries | `d:d_ate<` (Dates), `py:p_oly<` (Polynomials), `g:g_raph<` (Graphs), `b:f_ields<` (Bits), `cs:c_olumns<` (Csv), `k:c_ases<` (Check) | notation checked when the program is compiled, constants worked out once (bit offsets, a polynomial's coefficients), names defined from data (columns, graph nodes) |
| X_eTaL-ML | `net:n_etwork<`, `net:m_odel<`, `net:p_arams<`, `net:s_hapes<` (Net) | a whole network from one spec line, its parameter count at compile time, a load-time shape check per layer |
| X_eTaL-extensions | `ffi:b_ind<` (Ffi) | a facade function from a native function's signature, one line each |

Two limits matter here. An expansion may not import a library
(MC23, `macro-import`), and a macro's text cannot name the alias the
caller chose for another library (X_eTaL-ML's ask M11), so Net writes
`nn:` and asks the reader to import NN under that alias. Any macro
this repo writes later would inherit both.

### Extensions

- **ABI V1** (`X_eTaL-extensions/docs/abi-v1.md`): a shared library
  exports `xetal_extension_v1`, a descriptor of typed functions of
  arity 0 to 2. Values are Bool, Int, Float, Text, or a dense,
  row-major array of rank 0 to 9 (Bool, Int as i64, Float as f64,
  Char), passed in binary; panics are contained.
- **The bridge** (`docs/bridge.md`): X_eTaL cannot call native code
  yet (their ask E1, X_eTaL Saga 23, planned, not started), so
  programs run with `xetal-x`, X_eTaL's CLI whose store routes paths
  `ext:EXTENSION/FUNCTION` to extensions. Arguments go out with
  `[]N_PUT` and results come back with `[]N_GET` as text: the binary
  ABI is reached through a text channel, and the text is the cost.
- **Packages**: `extensions/NAME/` with `extension.toml` (name,
  version, `abi = 1`, facade, library stem), `rust/` (a cdylib on
  `xetal-ext-sdk`), `lib/NAME.xtl` (the facade). `xetal-x --ext DIR`
  loads any package directory, so a package can live in this
  repository and be built against a pinned X_eTaL-extensions, its
  `xetal-ext-sdk` crate taken by path from a clone in `work/`.

## Measurements

On this Mac (Apple M1 Max, release builds), timed inside one
process with Clock's `ck:t_ime`, so start-up and file reading are
left out:

| Work, a million Floats | Time |
| ---------------------- | ---- |
| `'+ r_/ x` in X_eTaL | 90 ms |
| the pipeline `'+ r_/ (x + x * x) m_ax 0.0` in X_eTaL | 152 ms |
| a sum in Rust over the bridge (out as text, one number back) | 207 ms |
| the array out to Rust and back over the bridge (`hx:e_cho`) | 401 ms |

| Matrix product, X_eTaL `'+ '* i_nner` | Time | Per multiply-add |
| ------------------------------------- | ---- | ---------------- |
| 128 by 128 | 198 ms | 94 ns |
| 256 by 256 | 1540 ms | 92 ns |
| 1024 by 1024 (estimated, the same rate) | about 100 s | |

From step 6 of saga 1 (`just sizes`, whole processes): the GPU path
for a million-item pipeline takes 0.24 s, of which about 0.15 s is
compiling the kernels.

What this says about test (b) for X1:

- **Elementwise work and single reductions: no.** The data crosses the
  bridge as text at about 2.5 million Floats per second each way,
  which is slower than X_eTaL computing the answer itself. Through
  the bridge, a GPU can only lose here; with a binary native hook
  (E1) it would be roughly even.
- **Products and models: yes.** A product does n^3 multiply-adds on
  2 n^2 inputs and n^2 outputs. At 1024, X_eTaL needs about 100 s;
  the bridge moves the 3 million Floats in about 1.2 s; the GPU
  computes the product in well under a second. The crossover is near
  n = 128, where X_eTaL takes 0.2 s and the bridge 0.02 s. A
  Jev-like model (saga gpu-models) is all products.
- So the extension's demos and timings should be products and model
  layers, and its documentation should say plainly that elementwise
  offload through the bridge is slower than staying in X_eTaL.

Test (c) holds for X1 regardless of speed: OpenCL is a C-ABI library
(Apple's `OpenCL.framework`; on Linux the ICD loader `libOpenCL.so`
and a vendor driver), and X_eTaL has no way to call it.

### Checked after building it

The `Gpu` extension (saga gpu-extension) measured on this Mac, in one
X_eTaL program (`extensions/gpu/demos/offload.xtl`):

| Work | Evaluator | GPU through the bridge |
| ---- | --------- | ---------------------- |
| 512 by 512 product | 14545 ms | 462 ms (load and bind 289, run 12, read back 127) |
| a million items, elementwise and summed | 188 ms | 2447 ms |

Both verdicts hold: the product is 31 times faster even through the
text bridge, the elementwise work 13 times slower, and the bridge is
nearly all of the GPU side's time.

## X1 in detail: the `Gpu` extension

What a program would look like:

```
"gp:" u_se< "Gpu"
"ac:" u_se< "Accel"
w := 1024 1024 r_eshape ...                 # weights, read from a file
x := 1024 1024 r_eshape ...
prog := gp:m_atmulProgram 1024 1024 1024    # XIR text, by an ordinary function (X4)
y := gp:r_un1 prog (x gp:p_air w)           # on opencl:0, the default schedule
"cpu and gpu agree:"
(x ac:m_atmul w) gp:n_ear y                 # the evaluator stays the reference
```

Native functions, each one line of `ffi:b_ind<` in `lib/Gpu.xtl`:

| Facade | Signature (Ffi kinds) | What |
| ------ | --------------------- | ---- |
| `gp:d_evices @` | `unit -> text` | the devices, as `xetal-gpu devices` lists them |
| `gp:l_oad t` | `text -> int` | check an XIR program (its text); the number of inputs, or an error naming the line |
| `n gp:b_ind a` | `int float -> int` | bind input n (in program order) to an array of any rank |
| `gp:r_un d` | `int -> int` | run on `opencl:d` (or the interpreter for -1); the number of outputs |
| `gp:o_utput i` | `int -> floats` | output i, reshaped |
| `gp:e_xplain @` | `unit -> text` | the last plan, as `xetal-gpu explain` prints it |

The extension keeps one session per thread (the loaded program, the
bound inputs, the last outputs), because ABI V1 functions take at
most two arguments and give one result. The ordinary X_eTaL half of
the library (`gp:r_un1`, `gp:p_air`, `gp:m_atmulProgram`, `gp:n_ear`)
is written in `.xtl` over those six.

Where it lives: here, as `extensions/gpu/` (a package directory:
`extension.toml`, `rust/` a cdylib on this repo's runtime and
X_eTaL-extensions' `xetal-ext-sdk`, `lib/Gpu.xtl`). X_eTaL-extensions
is pinned like X_eTaL (`XETAL_EXTENSIONS_COMMIT`, cloned into
`work/xetal-extensions/`, which also builds `xetal-x`). Nothing in
X_eTaL-extensions changes; it may list the package later.

Limits to state in its README: Float arrays travel as f64 and the
device computes in the schedule's width (f32 on this Mac); Int
outputs come back through the `floats` kind and are exact only up to
2^53; the text channel is the cost; everything here becomes faster,
without changing a program, when X_eTaL's native hook lands (E1),
because only `Ffi.xtlm`'s expansion changes.

## Asks that follow

To X_eTaL (added to `docs/xetal-asks.md`):

- **G8, the native hook with binary arrays** (X_eTaL-extensions' E1,
  X_eTaL Saga 23): the measured bridge cost (401 ms per million
  Floats out and back) is what keeps offload of elementwise work from
  paying; with ABI V1's binary arrays called directly it would not.
- **G3, static shapes**, is restated as the reason an offload macro
  (X5) cannot be written well.

To X_eTaL-extensions (recorded here; this repo does not change it):

- an `ints` argument kind and a `bool` result kind for `ffi:b_ind<`,
  so an Int array goes out without passing through `f_loat`;
- results of more than one array per call, or the session pattern
  above documented as the way to return several;
- `xetal-x` looks for native libraries beside the path it was started
  from without resolving a symlink, so a repository that links
  `bin/xetal-x` to the binary finds none (found 2026-10-07; this repo
  writes a wrapper script instead); resolving the path would fix it.

## Plan changes

- A new saga, **gpu-extension**, after gpu-algebra: pin
  X_eTaL-extensions; the `gpu` package; the facade by `ffi:b_ind<`;
  the X_eTaL half (X4's program writers); tests with `xetal-x` on this
  Mac; a demo where a 512 by 512 product is computed by the evaluator
  and by the GPU from the same X_eTaL program, timed both ways.
- Saga gpu-hardware moves to the **Arch machine with a newer GPU**
  (coming soon): nothing in the code changes (the OpenCL library is
  loaded at run time, `check-equiv.sh` and `check-sizes.py` use every
  device found); the machine needs `ocl-icd` and the vendor's OpenCL
  (`opencl-nvidia` for NVIDIA, `rocm-opencl-runtime` for AMD) and
  `clinfo`; with fp64 there, `--float f64` schedules become testable
  and the Float tolerance can be tightened.
