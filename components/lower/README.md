# xetal-gpu-lower

The lowering: X_eTaL source to an XIR program, through the pinned
X_eTaL's own crates. `xetal_program::load` assembles a program with
its libraries (found beside it and on `XETAL_PATH`) and lowers it to
X_eTaL's Core; this crate walks that Core and builds the XIR program
the device runs. Nothing is parsed here, and no library body is
copied: the same Core the evaluator runs is what gets translated.

```
xetal-gpu lower libs/Accel/demos/tiny-net.xtl    # the XIR program
xetal-gpu run libs/Accel/demos/tiny-net.xtl --device opencl:0
xetal-gpu explain libs/Accel/demos/tiny-net.xtl  # which X_eTaL became which kernels
```

Every `xetal-gpu` command takes an `.xtl` file as well as an `.xir`
one; the `.xtl` is lowered first. The crate depends by path on the
clone in `work/xetal` (`just xetal` makes it), so a fresh checkout
builds the components only after that; the pin is `XETAL_COMMIT`, and
what a repin breaks here is fixed here (ask G11 in
`docs/xetal-asks.md`).

## How it translates

The translator is a small evaluator over Core. A value is one of:

- **static**: known now. A literal, a strand, a file read with
  `[]N_GET` and `n_umbers`, a shape (`s_hape`, `t_ally`, `f_irst`,
  `c_at` of shapes), a count, a range. A static value is also marked
  *shape-like* (a shape, a count, Int arithmetic on those) or *data*
  (a strand, a range, a reshape, a file).
- **a node**: a value of the XIR program being built, which the device
  computes.
- a function (a library or local lambda with its environment), a
  primitive with the functions it has taken (`'+ r_/` is `r_/`
  holding `+`), a primitive waiting for its right argument, `p_ower`
  waiting for its count, text, unit, a tuple.

A primitive applied to arguments:

- folds, when every argument is static and shape-like and the
  primitive is Int arithmetic, a take or drop, or a select of shapes
  (so `f_irst -1 t_ake s_hape m` and `(2 s_elect s_hape x) - 2` are
  numbers the program never sees);
- builds data statically when it is a constructor (`r_eshape` of a
  strand, `r_ange`, `o_ffsets`, `f_loat` of a static, `n_umbers`), so
  a program's weights and batch become constants written in once,
  under the name the program gave them;
- emits an XIR operation otherwise, its static arguments written in
  as constants: arithmetic and comparisons with scalar extension (a
  literal adapts to the type it meets, so `x m_ax 0` on Floats is
  `max` with `0.0`; `/` gives a Float, so Ints are made Floats
  first), `e_xp`, `l_og`, `n_eg`, `a_bs`, `f_loat`, `f_loor`, `r_/`
  with and without an axis (all axes is the whole), `'+ '* i_nner`,
  `t_able` with `l_eft`, `r_ight` or an arithmetic function, `t_ake`
  and `d_rop` along any axis, `r_avel`, `r_eshape`, `o_-`, `o_\`
  and `o_\_jk`, `t_ranspose`, `s_elect` along any axis.

A library function is a closure inlined at each call; `let` and
`eval` statements run in order, every `eval` an output. `p_ower` with
a static count unrolls. What is outside the subset is refused with a
reason naming the X_eTaL construct: a guard on a device value, a
variable that changes, `w_here` (a shape the data decides), `s_\`,
`^`, `c_at` of device arrays (X_eTaL-ML's ask GA7 asks for the last
three), `i_nner` with other functions than `+` and `*`.

## What the device cannot take, the evaluator computes

A top-level statement that is refused does not fail the program
(`lower_file_full`, what `xetal-gpu run` uses): the pinned `xetal`
runs the program up to that statement, in the program's directory
with its libraries, and

- for a binding (`v := s_ort ...`), prints the value's shape and
  items, which come back as an **input** of the XIR program under the
  binding's name, bound with the program; the device computes
  everything that follows from it;
- for a printed statement (`w_here v > 4`), prints it, and that text
  is printed in its place among the device's outputs
  (`Lowered::printed` is the order).

So data preparation stays X_eTaL, however it is written, and the
accelerated part is whatever the subset covers. The attempt's partial
values are rolled back first, so nothing dead reaches a kernel. Each
evaluator run goes through a temporary file beside the program
(`.xetal-gpu-host-PID.xtl`, removed after); `xetal` is `bin/xetal`
of the repository containing the program or the working directory,
or `XETAL_BIN`. The cost is one evaluator run per such statement
(two for a printed one), fine for a demo, to be replaced by the
session's binary binding when a model's batch is large.

`xetal-gpu explain FILE.xtl` ends with a `from:` section naming,
for each kernel, the X_eTaL lines whose values it computes.

## A function for the runner

`lower_function(text, params, dir, uses)` lowers one function given
as its source (`{ x w -> ... }`) for arguments of the given types:
the function is defined as `u:a_ccel` in a program of its own with
the imports `uses` (the calling file's `u_se<` lines), applied to one
input per parameter, and its result is the program's output. This is
what the gpu extension runs for `gp:a_ccel<`: the arguments' shapes
are known when they are bound, so the lowering is specialized to
them and cached by the runner per set of types.

## Checked

`components/lower/tests/demos.rs` lowers every demo in
`libs/*/demos/` and runs it on the reference interpreter: the output
must be the evaluator's baseline, exactly. `scripts/check-equiv.sh`
(in the gate) does the same through `xetal-gpu run`, on the
interpreter and on every device found (Floats within 1e-5), and
`components/opencl/tests/twins.rs` pins each lowered program's
kernel source and plan. The twelve hand-written twins were this
translator's oracle until it passed all of them (2026-10-13); ten of
the twelve lowered programs plan launch for launch as the hand twins
did, and the other two differ because the source recomputes a value
the twin shared, or spells top-k as its composition.

## Not yet

The models of X_eTaL-ML need `c_at` of arrays, `'= t_able` and `^`
(GA7); `s_\`; tuples as several inputs and outputs; `m_ap` over rows.
A statement inside a function that is outside the subset still fails
the whole function (only top-level statements fall back to the
evaluator).
