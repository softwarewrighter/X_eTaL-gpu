# Asks for X_eTaL

What this repository needs from X_eTaL (`../X_eTaL`) that X_eTaL does
not have yet, and bugs it uncovers. This repo does not change X_eTaL:
each ask is filed here (and taken to `../X_eTaL`), the work here uses
the workaround noted below or waits, and the workaround is removed
when the ask lands in the X_eTaL commit this repo pins
(`XETAL_COMMIT`, now 75e6a5c, X_eTaL main of 2026-10-06).

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
| G1 | open (blocker) | feature | **An accelerator IR in X_eTaL** (`components/accel/xetal-accel-ir`, research7 section 2 and "What belongs in XIR"): a typed dataflow DAG of Input, Const, elementwise ops (add, sub, mul, div, min, max, neg, abs, comparisons, and, or, not, select, exp, log, cast), Map/Zip over arrays with scalar extension, Reduce along an axis, later Dot/Matvec/Matmul and lookup; values carry a scalar type and a static shape; a textual form that round-trips; no knowledge of OpenCL, CUDA or Verilog | everything: this repo is a backend of that IR | `components/xir`, a provisional IR in this repo written to the same description (its text form is in `components/xir/README.md`), to be replaced |
| G2 | open (blocker) | feature | **Core to accelerator IR lowering** (`xetal-accel-lower`, research7 "accelerator extraction"): given a typed Core program (or one top-level function), either the IR for the acceleratable region or a reason it is not acceleratable (recursion, dynamic shapes, I/O, nested arrays, general higher-order functions). Needs static shapes: see G3 | every example: today each `.xtl` is lowered by hand into a `.xir` twin | hand-written `.xir` files, each checked for equivalence against the pinned evaluator's output of the `.xtl` (`scripts/check-equiv.sh`) |
| G3 | open (blocker) | feature | **Shapes known statically** (X_eTaL-ML's M12 asked the same for shape-correctness): the lowering needs each array's shape at compile time, at least for a function's arguments as given by the caller (the kernel is specialized per shape), and the scalar type `Int` versus `Float` versus `Bool` (the inferred types give the latter) | G2; the OpenCL kernels are generated per shape and element type | the `.xir` twins state shapes; the evaluator is the reference |
| G4 | open (blocker) | feature | **A stable Core IR dump** as the contract until G1 and G2 exist: `xetal core FILE` prints s-expressions (`(let b (app2 #* 2 a))`), undocumented and unversioned; a documented, versioned form (JSON, or the text as is with a grammar) would let this repo prototype the lowering out of tree without waiting for G2 | a possible bridge (`xetal core` to `.xir`) while G2 is unplanned | none used: the bridge is not built (plan A4); the hand-written twins stand in |
| G5 | open | feature | **Bind arrays into a program and read results back without text** (X_eTaL-demos' ask, X_eTaL-ML's M5): the host runtime moves X_eTaL arrays to device buffers and back; today the only way to get an array into or out of the evaluator is source text and printed output | the end-to-end `xetal run --device opencl` of research7 section 10, once lowering exists | the examples carry their data as literals and data files; results are compared as printed text with a Float tolerance |
| G6 | open | feature | **Device selection at the command line** (research7 section 11: `xetal run --target opencl FILE`), a compiler and runtime concern, not language semantics; a backend registered by a downstream crate | the one-command demo | `xetal-gpu run` takes the `.xir` twin; `xetal run` the `.xtl`; the equivalence script runs both |
| G8 | open | feature | **The native hook, with arrays passed in binary** (X_eTaL-extensions' E1, X_eTaL Saga 23, planned): today extensions are reached through `xetal-x`'s text channel, measured here at 401 ms for a million Floats out and back, slower than X_eTaL's own 152 ms pipeline on them (`docs/macros-and-extensions.md`) | the `Gpu` extension (saga gpu-extension): only products and models gain through the bridge | run with `xetal-x`; offload only compute-heavy work |
| G9 | open | feature | **Tuples in the lowering** (`docs/tuple-plan.md`, T5): when the accelerator IR and the lowering (G1, G2) are designed, a function's tuple parameter (`{ (x, w) -> ... }`) should become several IR inputs and a tuple result several IR outputs, so a kernel with several results needs no extra calls | a model layer with several outputs; every program with several inputs | inputs bound one by one and outputs read one by one through the extension |
| G7 | open | docs | **Fold orders specified** (found 2026-10-07): `r_/` and `'+ '* i_nner` fold from the right, and `'+ r_/_12` of a matrix folds down the columns first and then the column results, not the ravel; none is written in `docs/reference.md`, and Float results depend on it (a 64 by 64 product's sum differed in the last digits until this repo matched it). Also **Float printing specified**: the equivalence checks compare the evaluator's printed Floats (shortest round-trip, `3.0`, `0.30000000000000004`) with an f32 device's results; the rule for printing (and for very large or small values, `1e21`) is not written down in `docs/reference.md` | `scripts/check-equiv.sh` | Floats are compared numerically within a relative tolerance (1e-5 for f32, exact for Int) |

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
schedule), the hand-written `.xir` twins are deleted, and the
equivalence script runs `xetal` for both sides.

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
