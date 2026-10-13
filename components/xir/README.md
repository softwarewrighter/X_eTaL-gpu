# xetal-gpu-xir

The `binary` module reads and writes portable f32, f64 and i64 array
files with checked shapes and payload lengths; it has no device
dependencies. See [binary arrays](../../docs/binary-arrays.md).

The provisional accelerator IR of X_eTaL-gpu (XIR): typed arrays,
elementwise maps with scalar extension, a select by mask,
reductions, casts; its text form, a checker that gives every value
its type and shape, a reference interpreter that computes as the
X_eTaL evaluator does, and printing as `xetal` prints.

It stands in for the accelerator IR that belongs in X_eTaL
(`docs/xetal-asks.md`, G1) and is written to `docs/research7.txt`'s
description. It knows nothing of OpenCL, CUDA or Verilog, and nothing
of X_eTaL's syntax: a program reaches it as text, today written by
hand beside each `.xtl` example (the twins in `libs/Accel/demos/`),
tomorrow by X_eTaL's lowering (G2). When that lands this crate
becomes an adapter and the twins go.

## The text form

```
# a comment runs to the end of the line
%a = input i64 [8]                   # bound by the host at run time
%b = const i64 [8] 1 2 3 4 5 6 7 8   # every item, row order (or one item for all)
%k = const f64 [] 2.0                # a single value
%c = map add %a %b                   # elementwise; a rank-0 argument extends
%m = map gt %c %b                    # comparisons give bool
%s = select %m %c %b                 # cond ? a : b, item by item
%r = reduce add %s                   # a whole array to one number
%q = reduce max axis=2 %m            # along one axis (1 is the first): rank n to n - 1
%f = cast f64 %r                     # between scalar types, same shape
%p = matmul %a %w                    # a's last axis with w's first ('+ '* i_nner)
%t = take -1 %wb                     # n t_ake along the first axis (negative: from the end)
%d = drop -1 %wb                     # n d_rop along the first axis
%v = ravel %t                        # r_avel: every item as one vector
%s = table right %o %v               # a 'f t_able b: every pairing; f a two-argument map, left or right
%tt = transpose %a                   # o_\: the axes reversed; transpose 3 1 2 %t is 3 1 2 t_ranspose t
%ro = rotate axis=3 -1 0 1 %t        # -1 0 1 o_-_3 t: one copy per amount along a new first axis
%dk = drop -1 axis=4 %ro             # -1 d_rop_4: along any axis (take likewise)
%rs = reshape [9 4] %dk              # r_eshape: the items in row order, repeated as needed
%em = lookup %ids %table             # ids s_elect table: rows at positions (Ints from 1), in ids' shape
%tk = topk 2 %m                      # a bool mask of each row's 2 largest (k ac:t_opk m); topk 2 axis=1 down the columns
output %c                            # what the program prints, in order
output %f
```

| Part | Form |
| ---- | ---- |
| scalar type | `i32`, `i64` (X_eTaL's Int), `f32`, `f64` (Float), `bool` |
| shape | `[]` one value, `[8]` a vector, `[2 3]` a matrix (rows first) |
| map operations | `add sub mul min max neg abs` (any number); `div exp log` (Floats only: X_eTaL's `/` gives a Float, so an Int is cast first); `idiv mod` (Ints, floored as `d_iv` and `m_od`); `eq ne` (any one type, give bool); `lt le gt ge` (numbers, give bool); `and or not` (bool) |
| reduce operations | `add mul min max`, folded from the right as `r_/` is; the whole array (`r_/_12`, or `r_/` of a vector) without `axis=`, one axis with it (`axis=2` is `r_/_2`, `axis=1` is `r_/` of a matrix) |
| structure | `take n`, `drop n` (along the first axis, or `axis=k`; a negative n from the end; no padding), `ravel`, `reshape [s]` (the items in row order, repeated as needed; an empty argument has nothing to repeat), `table f` (APL's outer product; `left` and `right` spread a vector along a new axis, as `'l_eft t_able` and `'r_ight t_able` do in X_eTaL-ML's NN) |
| rotate | `rotate n` (`n o_- a`, a positive n toward the front) or `rotate axis=k n1 n2 ...` (`n o_-_k a`; several amounts give one copy per amount along a new first axis, as X_eTaL does) |
| topk | `topk k %a` (along the last axis; `axis=j` for another): a bool mask of each line's k largest items, the largest taken out k times with its ties, as X_eTaL-ML's `sm:t_op` and its router choose (so every item at or above the line's k-th largest distinct value; all of a line with fewer); one operation for the composition's 4k passes. There is no argmax operation: X_eTaL-ML's `nn:a_rgmax` is a row max, a spread, a mask and a min of positions, four kernels the emitter already has |
| lookup | `lookup %i %a` (`i s_elect a`: the items along the first axis at the Int positions i, from 1; the result is a's shape with that axis replaced by i's shape, so a batch of token ids picks a batch of rows) or `lookup axis=k %i %a` (`s_elect_k`; along a later axis i is a vector or a single value, as X_eTaL has it); a position outside the axis is refused: written ones by the checker, bound ones by the runtime before the upload, the interpreter as the evaluator does |
| transpose | `transpose` (`o_\`: the axes reversed) and `transpose p` (`p t_ranspose`: axis i of the argument becomes axis p[i], 1 the first, each listed once; `o_\_23` is `transpose 1 3 2`); a vector or a single value is unchanged |
| cast | Int to Float exactly, Float to Int toward negative infinity (`f_loor`), Bool to 0 or 1, a number to bool when not 0 |

Rules: a value is defined once, before it is used; a map's arguments
share one scalar type and one shape, except that a single value
(`[]`) extends to every item; `select` takes a bool condition;
`reduce` needs rank 1 or more and gives rank 0, or, along an axis,
leaves that axis out (an empty axis has no value); the program has at
least one `output`.

## Semantics (X_eTaL's)

The interpreter is the reference the GPU is compared with:

- Int is i64 (wrapping), Float is f64; at `i32` and `f32` every
  result is rounded to that type, so the interpreter at f32 models a
  device computing in single precision.
- `idiv` and `mod` are floored: `-7 idiv 2` is -4, `-3 mod 2` is 1,
  `3 mod -2` is -1; division by zero is an error.
- A reduce folds from the right: the last item first, then each
  earlier item combined with the result so far; the whole of a matrix
  is folded down its columns first, then across the column results,
  as `'+ r_/_12` does (found by probing: not the ravel's fold) (`'+ r_/ 0.1 0.2 0.3`
  is `0.1 + (0.2 + 0.3)`, which prints `0.6`; the left fold would
  print `0.6000000000000001`).
- Printing follows `xetal`: a Bool as 1 or 0; a vector's items
  separated by one space; a matrix with right-aligned columns; a
  higher rank plane by plane with one blank line between planes,
  two between rank-3 blocks, and so on (probed to rank 5); a
  Float in full decimal with the shortest digits that read back, and
  always a point (`3.0`, `0.00000015`, `1000000000000000000000.0`,
  `-0.0`, `inf`).

## Use

```rust
use xetal_gpu_xir::{check, format, run, text};
let program = text::parse(src)?;          // Error names the line
let checked = check(program)?;            // Error names the value
let outputs = run(&checked, &inputs)?;    // inputs: HashMap<String, Array>
for a in &outputs { println!("{}", format::show(a)); }
```

`text::print` gives the canonical text back (`parse(print(p)) == p`).
The tests in `src/` cover parsing errors, the typing rules, the
arithmetic, the fold order, casts, narrow types and printing; the
equivalence script (`scripts/check-equiv.sh`) proves every twin
against the evaluator's baseline, exactly.

## Not yet

A map still extends only a single value, as
X_eTaL does (a vector across the rows of a matrix is a shape error
there too).
