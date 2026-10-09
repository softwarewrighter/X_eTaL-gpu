# Tuples in X_eTaL-gpu

Whether this repository uses X_eTaL's tuples, where they would help,
what blocks each use, and when to adopt them. Written 2026-10-08
against X_eTaL 1998414 (this repository's pin) and X_eTaL-extensions
7bdd827.

## Status

No code here uses tuples, and nothing so far needs them. They are
available to code that runs on the pinned `xetal`, but not to code
that runs under `xetal-x`.

X_eTaL's tuples (Saga 39, decisions TU1 to TU12 in
`../X_eTaL/docs/lang-choices.md`):

- A tuple is written `(w, b)` and taken apart only by a pattern, in
  a binding (`(w, b) := s`) or a lambda parameter
  (`{ (w, b) x -> ... }`).
- A tuple is not an array: it has no shape, and a tuple where an
  array is expected is a type error before the program runs.
- The higher-order built-ins take tuples as they are, so `p_ower`
  repeats a step on a tuple state.

Checked with the pinned `xetal`:

```
      u:f_ := { (d, a) b -> d + a + b }
      (1, 2.0) u:f_ 3.0
6.0
      u:g_ := { m -> (s_hape m, '+ r_/_12 m) }
      u:g_ 2 3 r_eshape r_ange 6
(2 3, 21)
      u:s_tep := { (w, b) -> (w + 1, b * 2) }
      3 'u:s_tep p_ower (1, 1)
(4, 8)
```

Under `xetal-x` (X_eTaL-extensions' bridge host, which embeds X_eTaL
v0.1.0) the same `t := (2, 3.5)` stops with
`error[unexpected-char]`.

## Where code runs

What can use tuples depends on which X_eTaL runs it:

| Code | Runs on | Tuples |
| ---- | ------- | ------ |
| `libs/Accel` (library, tests, example programs) | the pinned `xetal`, and also `xetal-x` (the gate checks every Accel program prints the same under it) | not until `xetal-x` has them, while Accel must stay usable from extension programs |
| `extensions/gpu/lib/Gpu.xtl` and programs that use the extension | `xetal-x` only | no |
| the model programs of saga gpu-models (`examples/jev/`) | the pinned `xetal` | yes |
| the docs (`xetal doc`) | the pinned `xetal` | yes |

## Where tuples would help

| # | Use | What it gives | Needs | Blocked by |
| - | --- | ------------- | ----- | ---------- |
| T1 | A device chosen per call: `(d, a) gp:p_roductOn b` | `gp:p_roduct` and `gp:d_ense` always run on `opencl:0`, because a function takes at most two arguments; a pair on the left adds the device without a session setting | a tuple pattern in a facade function | `xetal-x`'s X_eTaL v0.1.0 (ask X-1 below) |
| T2 | All outputs of a run at once: `(y, s) := gp:r_unAll 0` | a program with several outputs is read one call at a time (`gp:o_utput 1`, `gp:o_utput 2`) | a tuple result from a native function | ABI V1 has no tuple value, and the binding macro has no tuple kind (ask X-2 below); also X-1 |
| T3 | A model's weights as one value: `(e, wq, wk, wv, w1, w2) := weights` and `x m:f_orward weights` | the Jev-like model has many weight arrays; a tuple keeps them as one argument instead of a function per layer | tuple patterns in the model program | nothing: the model runs on the pinned `xetal` |
| T4 | Training state through `p_ower`: `n 's_tep p_ower (w, m, v, k)` | an optimizer's state (weights, moments, a step count) carried as one value, which X_eTaL-ML asked tuples for (M13) | tuples through `p_ower` (TU9) | nothing, if saga gpu-models trains in X_eTaL; it plans a Rust trainer instead, so this is optional |
| T5 | Several inputs and outputs in the lowering | a function `{ (x, w) -> (y, s) }` maps directly onto an XIR program with inputs `%x %w` and outputs `%y %s` | X_eTaL's accelerator IR and lowering (asks G1, G2) accepting tuple patterns as inputs and tuple results as outputs | G1 and G2, not planned in X_eTaL yet (ask G9 below) |

Not uses:

- **Accel's exports.** Each takes one or two arrays and gives one;
  `ac:d_ense` already packs the bias into the weights as NN does. A
  tuple would not make any of them shorter.
- **XIR itself.** The IR already has any number of inputs and
  outputs; tuples are an X_eTaL-side way of naming them.
- **Arrays of tuples on the device.** TU8 keeps tuples out of arrays
  in v1; a GPU wants a tuple of arrays anyway, which is what T5
  describes.

## Asks

To X_eTaL-extensions (recorded here and in
`docs/macros-and-extensions.md`; this repository does not change it):

- **X-1, a newer X_eTaL pin.** `xetal-x` embeds X_eTaL v0.1.0, which
  has neither tuples nor `h:` names. Pinning X_eTaL 1998414 or later
  unblocks T1, lets the Gpu facade move its private helper to `h:`
  (as `xetal migrate` proposes), and lets Accel use tuples where they
  shorten a program.
- **X-2, tuples across the ABI.** ABI V1 reserves tag 8 for records
  and has no tuple value; the binding macro's kinds have no tuple
  either. A tuple value (its parts as values) and a tuple kind in
  `ffi:b_ind<` would give T2.

To X_eTaL (added to `docs/xetal-asks.md` as G9):

- **G9, tuples in the lowering.** When the accelerator IR and the
  lowering (G1, G2) are designed, a function's tuple parameter should
  become several IR inputs and a tuple result several IR outputs
  (T5), so a kernel with several results needs no extra calls.

## The rule for adopting them

Tuples are used where they make a program shorter or remove a session
setting or a second call, as with macros and extensions
(`docs/macros-and-extensions.md`), and nowhere else:

1. Code that runs under `xetal-x` (the Gpu facade, extension demos,
   Accel while extension programs import it) does not use them until
   X-1 lands and `XETAL_EXTENSIONS_COMMIT` moves to that pin.
2. Code that runs only on the pinned `xetal` (the models of saga
   gpu-models) may use them from now on: T3 first, T4 if training
   moves into X_eTaL.
3. When X-1 lands: T1 (a device per call), the `h:` helper, and a
   check that every Accel program still prints its baseline under the
   new `xetal-x`.
4. When X-2 lands: T2.
5. When G1 and G2 land: T5 in the adapter from X_eTaL's IR.
