# Asks for X_eTaL

What this repository needs from X_eTaL (`../X_eTaL`) that X_eTaL does
not have yet, and bugs it uncovers. This repo does not change X_eTaL:
each ask is filed here (and taken to `../X_eTaL`), the work here uses
the workaround noted below or waits, and the workaround is removed
when the ask lands in the X_eTaL commit this repo pins
(`XETAL_COMMIT`, now 1998414).

Each entry: status (open, filed, landed, dropped), kind (feature, bug
or speed), what here needs it, why, a minimal repro or example, and
the workaround in use. The G numbers are this repository's; X_eTaL's
own ledger (`docs/asks.md`, generated from `docs/asks.toml`) may
number them differently.

The first four asks are the **blockers**: `docs/research7.txt`
describes an accelerator IR and a lowering that belong in X_eTaL
(research7: "the contract belongs to X_eTaL"), and as of 2026-10-06
neither is planned there (`../X_eTaL/docs/plan.md` has no saga for
it; `docs/wish-list.md` lists "Parallel and GPU kernels" as an XL
wish). Until they land, this repository carries a provisional IR of
its own, `components/xir`, written to research7's description, and
hand-lowers each example (`libs/Accel/demos/*.xir` beside each
`.xtl`), so that the GPU side can be built and tested now; the
provisional IR is to be replaced by X_eTaL's, not kept.

| # | Status | Kind | Ask | Needed by | Workaround |
| - | ------ | ---- | --- | --------- | ---------- |
| G1 | filed (X_eTaL's ledger GPU1: new, not yet planned, as of 27b3fd0, 2026-10-12) | feature | **An accelerator IR in X_eTaL** (`components/accel/xetal-accel-ir`, research7 section 2 and "What belongs in XIR"): a typed dataflow DAG of Input, Const, elementwise ops (add, sub, mul, div, min, max, neg, abs, comparisons, and, or, not, select, exp, log, cast), Map/Zip over arrays with scalar extension, Reduce along an axis, later Dot/Matvec/Matmul and lookup; values carry a scalar type and a static shape; a textual form that round-trips; no knowledge of OpenCL, CUDA or Verilog | everything: this repo is a backend of that IR | `components/xir`, a provisional IR in this repo written to the same description (its text form is in `components/xir/README.md`), to be replaced |
| G2 | filed (GPU2: new, not yet planned; needs GPU1) | feature | **Core to accelerator IR lowering** (`xetal-accel-lower`, research7 "accelerator extraction"): given a typed Core program (or one top-level function), either the IR for the acceleratable region or a reason it is not acceleratable (recursion, dynamic shapes, I/O, nested arrays, general higher-order functions). Needs static shapes: see G3 | every example: today each `.xtl` is lowered by hand into a `.xir` twin | hand-written `.xir` files, each checked for equivalence against the pinned evaluator's output of the `.xtl` (`scripts/check-equiv.sh`) |
| G3 | declined by X_eTaL (GPU3: T7, array types are rank-erased and shapes are checked at run time; no annotations in v0). So the lowering (G2) will specialize per shape when a program is bound, as this repo's twins already do: every shape is known when the inputs are bound | feature | **Shapes known statically** (X_eTaL-ML's M12 asked the same for shape-correctness): the lowering needs each array's shape at compile time, at least for a function's arguments as given by the caller (the kernel is specialized per shape), and the scalar type `Int` versus `Float` versus `Bool` (the inferred types give the latter) | G2; the OpenCL kernels are generated per shape and element type | the `.xir` twins state shapes; the evaluator is the reference |
| G4 | open (blocker; now the form this repo's lowering reads, 2026-10-13) | feature | **A stable Core IR dump** as the contract until G1 and G2 exist: `xetal core FILE` prints s-expressions (`(let b (app2 #* 2 a))`), undocumented and unversioned; a documented, versioned form (JSON, or the text as is with a grammar) would let this repo prototype the lowering out of tree without waiting for G2 | a possible bridge (`xetal core` to `.xir`) while G2 is unplanned | none used: the bridge is not built (plan A4); the hand-written twins stand in |
| G5 | planned in X_eTaL (GPU5: Saga 23, host bindings and native packages) | feature | **Bind arrays into a program and read results back without text** (X_eTaL-demos' ask, X_eTaL-ML's M5): the host runtime moves X_eTaL arrays to device buffers and back; today the only way to get an array into or out of the evaluator is source text and printed output | the end-to-end `xetal run --device opencl` of research7 section 10, once lowering exists | the examples carry their data as literals and data files; results are compared as printed text with a Float tolerance |
| G6 | open | feature | **Device selection at the command line** (research7 section 11: `xetal run --target opencl FILE`), a compiler and runtime concern, not language semantics; a backend registered by a downstream crate | the one-command demo | `xetal-gpu run` takes the `.xir` twin; `xetal run` the `.xtl`; the equivalence script runs both |
| G8 | planned in X_eTaL (GPU8: Saga 23) | feature | **The native hook, with arrays passed in binary** (X_eTaL-extensions' E1, X_eTaL Saga 23, planned): today extensions are reached through `xetal-x`'s text channel, measured here at 401 ms for a million Floats out and back, slower than X_eTaL's own 152 ms pipeline on them (`docs/macros-and-extensions.md`) | the `Gpu` extension (saga gpu-extension): only products and models gain through the bridge | run with `xetal-x`; `gp:b_indFile` binds prepared binary input files without passing their items through the text channel (`docs/binary-arrays.md`); outputs still use text |
| G9 | open | feature | **Tuples in the lowering** (`docs/tuple-plan.md`, T5): when the accelerator IR and the lowering (G1, G2) are designed, a function's tuple parameter (`{ (x, w) -> ... }`) should become several IR inputs and a tuple result several IR outputs, so a kernel with several results needs no extra calls | a model layer with several outputs; every program with several inputs | inputs bound one by one and outputs read one by one through the extension |
| G7 | open | docs | **Fold orders specified** (found 2026-10-07): `r_/` and `'+ '* i_nner` fold from the right, and `'+ r_/_12` of a matrix folds down the columns first and then the column results, not the ravel; none is written in `docs/reference.md`, and Float results depend on it (a 64 by 64 product's sum differed in the last digits until this repo matched it). Also **Float printing specified**: the equivalence checks compare the evaluator's printed Floats (shortest round-trip, `3.0`, `0.30000000000000004`) with an f32 device's results; the rule for printing (and for very large or small values, `1e21`) is not written down in `docs/reference.md` | `scripts/check-equiv.sh` | Floats are compared numerically within a relative tolerance (1e-5 for f32, exact for Int) |
| G10 | open | docs | **Declare the minimum Rust version** (found on Arch, 2026-10-12): building pinned 1998414 with rustc 1.88 fails at `xetal-prim-types/src/sig.rs:57`, an `if let` match guard stabilized in Rust 1.95; document the requirement and declare `rust-version` so the build fails with a clear toolchain requirement | fresh builds of the pinned evaluator and the extension host | select the installed newer toolchain with `RUSTUP_TOOLCHAIN=nightly` on this host; no vendored source edits; details in `docs/rtx5060-results.md` |
| G11 | open | feature | **A crate API for source to Core** (found 2026-10-13, A4b): this repo lowers X_eTaL source to its IR in process through the pinned X_eTaL's crates (a path dependency on the clone in `work/xetal`): parse a program or an expression, load its libraries, give the Core of each definition and statement; the API those crates expose today is whatever `xetal core` happens to use and is not promised | `components/lower` (saga 4, 0n to 0q) | the path dependency, re-pinned deliberately; what breaks on a repin is fixed here |
| G12 | open | docs | **The macro contract stated** (MC10, `docs/combinators-macros-and-trains.md`: a macro is an ordinary function from source text to source text): this repo's `gp:a_ccel<` relies on receiving a function's source text unchanged and on its result being spliced as written; please state it as a promise in `docs/reference.md` | the accel macro (saga 4, 0p) | none; the facade's tests pin the behavior |
| G13 | open (the user, 2026-10-16) | feature | **Stronger lint-time, macro-time and compile-time checks**, so that what this repository now catches at run time (on the host before an upload, or on the device as a wrong answer) is reported before a program runs; the cases are listed in the details below: a Float literal no device width can hold (`1e300`), a written position outside its table, an empty array reaching an operation that cannot take one, a reshape to a shape computed from data, a macro call whose argument form the macro does not accept, a relative `[]N_GET` path, a `## >>` example whose output has a blank line | every program lowered here: the lowering refuses what it cannot translate with a reason, but a wrong value (NaN from an overflow, a clamped position) is found only by comparing with the evaluator | the XIR checker (static shapes, written positions), the emitter (an empty constant's buffer decided at plan time) and the runtime (`check_f32_range`, `check_positions` at bind and rebind) check on this side, at the latest point that knows the shapes; the gate's equivalence checks against the evaluator catch the rest |

## Details

### G1 and G2: the IR and the lowering are X_eTaL's

research7 (the user's planning conversation, archival) places the
accelerator IR and the eligibility analysis in X_eTaL:

```
softwarewrighter/X_eTaL
        |  typed Core IR
        v
 accelerator representation/API
        |
        +-----------------------------+
        v                             v
softwarewrighter/X_eTaL-gpu    hardwarewrighter/X_eTaL-fpga
```

The IR "should know nothing about OpenCL, CUDA, Verilog, Yosys, Gowin,
Tang Nano, NVIDIA"; its job is "here is the computation", and the
downstream repositories answer "here is how this device realizes
that computation". What this repository needs from it, concretely,
is in `components/xir/README.md`: the provisional IR is the
request, written as code and tests. When X_eTaL's lands, the plan is:
`components/xir` becomes a thin adapter (X_eTaL's IR to this repo's
schedule), `components/lower` (which since 2026-10-13 lowers X_eTaL
source to this IR through X_eTaL's own crates, the hand-written
twins deleted) is retired, and the equivalence script runs `xetal`
for both sides.

A suggested first shape of the lowering, from research7 ("Don't
compile all X_eTaL"): accept statically known scalar types and ranks,
pure functions, arithmetic, comparisons, scalar extension, `e_ach`
and elementwise built-ins, `r_/` reductions, soon `i_nner`; reject
recursion, dynamic allocation, nested arrays, dynamic shapes, I/O,
native extensions, general higher-order functions. The evaluator
stays the golden reference.

### G3: static shapes

The lowering specializes a kernel to its argument shapes (an OpenCL
work-item per element; a reduction's workgroup size). X_eTaL's types
carry the scalar type, not the rank or the shape (X_eTaL-ML's M12
asked for shape ascriptions). The smallest useful form: the lowering
is given the argument shapes by the caller (the values the program
binds), and the IR records them.

### G4: the Core dump as a bridge

`xetal core` prints, for `a := 1 2 3 4` then `2 * a`:

```
(let a (array 1 2 3 4))
(eval (app2 #* 2 a))
```

This is enough to prototype the lowering for the integer subset out
of tree, but it is an internal form, so this repository does not
build on it (architecture decision A4 in `docs/plan.md`). If the
X_eTaL side prefers that this repo prototype the lowering, a
documented dump is the smallest ask.

### G13: what was found at run time, and where it could be found earlier

X_eTaL is statically typed to prevent run-time errors (the user,
2026-10-16: "when a runtime issue is found, suggest how it could be
detected or reported at compile time or lint time"). Each issue this
repository met while lowering X_eTaL-ML's programs, with the check
that exists here now and the earlier check asked of X_eTaL. The
common enabler is G3: with shapes in the types, most of these are
type errors; without them, a lint over literals and a check over
written shapes still cover the frequent cases.

| Found at run time | Here now | Asked of X_eTaL |
| ----------------- | -------- | --------------- |
| `1e300` as a sentinel in X_eTaL-ML's top-k (`sm:t_op`): infinity in f32, so a row became NaN on the device (2026-10-13) | the runtime refuses a Float beyond f32's range at upload, naming the fix; X_eTaL-ML shifts by a finite function of the data now | lint: a Float literal beyond the range of a 32-bit float (about 3.4e38), with the message that a device computes in f32 by default; or, once G3 lands, a width in the type (`Float32`) so the literal is a type error |
| a position outside its table in `s_elect` (lookup, 2026-10-12): the device cannot refuse one, the kernel clamps | the XIR checker refuses a written position outside the table statically; the runtime refuses a bound input's stray position before the upload (`Plan::position_checks`), with the evaluator's message | compile time: positions as a type bounded by the table's axis when both are written (G3's shapes make the bound known); lint: a written position vector against a written table |
| an empty constant (`0 2` reshape of nothing, the first accumulator of the world model's `c_at` loop) reached the OpenCL runtime, whose buffers hold at least one item, and the host indexed its first item (2026-10-16) | the lowering folds `c_at` with an empty side known statically to the other side; the emitter decides the empty constant's one-item buffer when it builds the plan (compile time for the device); the runtime refuses a constant that does not fit its buffer by name instead of indexing | compile time: an empty array known from the written shape (`0 2 r_eshape`, `0 t_ake`) is a type with G3's shapes, and an operation that cannot take one (`r_/` without an identity, `f_irst`) is a type error; lint: the written cases |
| a reshape to a shape computed from data (`nn:s_oftmax`'s `h:r_ows`, `('* r_/ -1 d_rop s_hape m) c_at -1 t_ake s_hape m`): the lowering saw a node, not a shape (2026-10-16) | the lowering folds a reduction of a shape-like vector (a shape's product) to a static shape, so `r_eshape` takes it | compile time: a shape is a kind, not an Int vector, when G3 lands: `s_hape` gives one and `r_eshape` takes one, and arithmetic on it stays static; until then nothing: this is the lowering's concern |
| a macro call parsed as a strand: `f "text" gp:l_ower "spec"` made two string literals after a function a strand, and `@ m:x< "text"` needs both sides (accel-macro, 2026-10-13) | the facade's expansion is pinned by a test; the macro uses `[]R_EJECT` for an argument it cannot take | macro time: the expander reports an argument form the macro does not accept with the macro's name and the call's span (today the error is the expansion's, about the expanded text); the contract itself is G12 |
| `[]N_GET` resolves a relative path against the working directory, not the file (run-from-source, 2026-10-13) | the lowering reads relative to the program's directory, and the demo inlines its weights | lint: a relative path in `[]N_GET`, or the rule stated in `docs/reference.md` (relative to the file, as `u_se<` finds libraries) |
| a `## >>` example whose output has a blank line (rank 3) ends the example early (doc-test, 2026-10-12) | the example was reshaped to rank 2 | lint: an example whose expected output contains an empty line, with the rule that a blank `##` ends an example; or a fence form for multi-block outputs |
| a program's own tolerance check, `1e-6 > 'm_ax r_/ ... a_bs (u:s_ample e) - expected` (X_eTaL-ML's denoiser), gives 0 on an f32 device, where the difference is 8.5e-6 (2026-10-16) | the benchmark script compares that line on the interpreter only and says so in `docs/ml-results.md`; X_eTaL-ML is asked for a tolerance f32 meets (the world model's check passes) | lint: a tolerance literal below about 1e-5 compared against a Float computed from data, with the message that a device computes in f32; with a width in the type (G3), a comparison of a `Float32` against a literal it cannot resolve is a type error |
| `error[rebind]` and `Int / Float` under the new pin | already compile-time errors, as they should be: nothing to ask | nothing |

