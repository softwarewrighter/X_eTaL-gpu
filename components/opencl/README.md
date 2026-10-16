# xetal-gpu-opencl

The schedule and the OpenCL C 1.2 emitter: from a checked XIR program
to kernel source and a launch plan. Nothing here touches a device;
the runtime component executes a plan, and this crate's tests pin
the source and the plan of every demo in `libs/*/demos/`, lowered
from its X_eTaL source by `components/lower`
(`tests/expected/<name>.cl` and `.plan`; `XETAL_BLESS=1 cargo test`
rewrites them after a reviewed change).

```
xetal-gpu kernel FILE.xir [--float f32|f64] [--int i32|i64] [--work-group N]
xetal-gpu explain FILE.xir [the same]
```

## The schedule

The schedule is data, separate from the IR (plan A6): how the device
represents each scalar type and how big a work-group is.

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `float` | `f32` (`float`) | X_eTaL's Float on the device; `f64` asks for `double` (`cl_khr_fp64`) |
| `int` | `i64` (`long`) | X_eTaL's Int on the device; `i32` asks for `int` |
| `work_group` | 256 | work-items per work-group, a power of two (the reduction tree needs it) |
| `tile` | 0 | inner products, and transposes of the last two axes, in T by T tiles in local memory; 0 untiled |

Bool is `int` (1 or 0, what a C comparison gives).

A schedule can be a file, `schedules/<device>.toml`, with the same
fields (each optional, the defaults otherwise) and two for the
reader, `device` and `tested`; an unknown key is an error, and
`validate` names the field that does not fit (a work-group that is
not a power of two, a tile too large for the work-group). The
runtime refuses a schedule its device cannot run the same way,
naming the field and the value to use.

```
xetal-gpu run FILE.xir --device opencl:0 --schedule schedules/apple-m1-max.toml --tile 16
```

| File | For | Run there |
| ---- | --- | --------- |
| `apple-m1-max.toml` | this repository's Mac; the defaults | yes |
| `nvidia-rtx5060-ti.toml` | Arch Linux's RTX 5060 Ti, 16 GB; work-groups of 256, 16 by 16 tiles | yes, Float as f32 |
| `nvidia-rtx5060-ti-f64.toml` | the same RTX 5060 Ti with Float as double | yes, Float within 1e-12 |
| `nvidia-modern.toml` | Turing to Blackwell (GTX 1660 Super, RTX 3060 and 3060 Laptop, RTX 3090, RTX 4070 Laptop, RTX 5060 Ti, A2): work-groups of 1024, 16 by 16 tiles | not yet |
| `nvidia-modern-f64.toml` | the same, Float as `double` | not yet |
| `nvidia-legacy.toml` | the Kepler Quadro K4200 and the Maxwell and Pascal Teslas (M10, P4) | not yet |

## What the emitter decides

- **Fusion.** Walking the program in order, consecutive elementwise
  values (map, select, cast) of one shape become one kernel, one
  work-item per element: each value is a C variable in the kernel's
  body, read as `b_name[i]` when it comes from a buffer, as a literal
  when it is a single-value constant. Only the values an output or a
  later kernel reads are written to buffers; the rest never leave
  registers. A reduction ends the group (so kernels follow program
  order: an elementwise value written after a reduce starts a new
  kernel even when it could have joined an earlier one).
- **Reductions.** One kernel per operation and element type
  (`reduce_add_float`, `reduce_max_long`): each work-item loads one
  item (or the identity), the work-group folds them as a tree in
  local memory, and work-item 0 writes the group's partial. The plan
  launches it again on the partials until one value is left: 8
  items are one launch, a million are three (4096 partials, 16, 1).
  Two partial buffers alternate between passes. The fold order
  differs from the evaluator's right fold, so Float results agree
  within a tolerance, Ints exactly.
- **Reductions along an axis.** One kernel per operation and element
  type (`reduce_axis_add_long`): one work-item per result item, each
  folding its line of the array (`len` items, `inner` apart) from the
  right, as the evaluator does, so the result is the evaluator's
  exactly in the device's width. Along the only axis of a vector it is
  the tree above. No local memory yet: a row of a million items is
  one work-item's loop.
- **Inner products.** `matmul` (a's last axis with b's first, any
  ranks) is one work-item per result item, its products summed from
  the last to the first as `'+ '* i_nner` sums them (`matmul_float`).
  With `--tile T` (a power of two, T * T at most the work-group) a
  work-group computes a T by T block, staging blocks of a and b in
  local memory; the blocks go from the last to the first and each
  block's products likewise, so the fold order is the same.
- **Structure.** A `take` or `drop` is one slice kernel per element
  type (`slice_float`): of every line along its axis, the items kept,
  one work-item each (along the first axis that is one contiguous
  run); a `ravel`, and a `reshape` to the same count, is the same
  buffer, no launch; a `reshape` to another count cycles the items
  (`cycle_float`) or fills a single value; a `rotate` is a gather
  kernel of its own with its amounts as a constant table in the
  source, one work-item per result item; a `table` is a kernel of
  its own, one work-item per result item reading `a[k / nb]` and
  `b[k % nb]` (a literal for a single value).
- **Lookups.** A `lookup` is a gather kernel of its own, one
  work-item per result item, reading its position from the positions'
  buffer (a literal for a single written one), with the table's lines
  and strides baked in. The device cannot refuse a position outside
  the axis, so the kernel clamps it, and the host refuses it first:
  the plan lists the input buffers a lookup reads as positions
  (`Plan::position_checks`) and the runtime checks them when an
  input is bound or rebound, with the evaluator's message; positions
  written in the program are checked by the checker.
- **Top-k.** A `topk` is one kernel per k and element type
  (`topk5_float`), one work-item per line: it keeps the line's k
  largest distinct values in private memory as it reads the line,
  then marks the items at or above the last of them; two passes
  where the composition (a row max, a spread, a mask, a subtraction,
  k times) makes 4k. Measured on the Apple M1 Max on a retrieval-sized
  matrix of 256 queries by 100,000 scores, the 5 largest per row: the
  composition 190 ms (16 launches), the kernel 122 ms, the scoring
  product that makes such a matrix 92 ms (host launch times, which on
  this Mac are the real ones; Apple's profiling counters say 4.6 and
  2.9 ms). One work-item per line is what bounds it, 256 work-items
  for 25.6 million items; a work-group per line is the next step if
  retrieval shows the need.
- **Windows, measured.** A convolution's 3 by 3 windows as Conv
  builds them (two rotations, four drops, a reshape) are six gathers
  on the device. For a batch of 64 pictures of 28 by 28 on the Apple
  M1 Max, the six kernels plus an 8-filter product take 0.055 ms of
  device time against 0.014 ms for one transpose plus the same
  product; the launches cost 6 ms either way. One windows operation
  (im2col) would save about 0.04 ms per batch here, so it is not
  added; it is worth revisiting when a CNN's time shows it, or on a
  card where the ninefold traffic costs more.
- **Transposes.** A `transpose` that moves axes is a gather kernel
  of its own (`transpose3`), one work-item per result item: its
  position is taken apart with the result's dimensions and put
  together with the argument's strides, both baked in as constants.
  One that moves nothing (a vector, a single value, `1 2 t_ranspose`)
  is the same buffer, no launch. With `tile = T`, a transpose of the
  last two axes (`o_\` of a matrix, attention's `K^T`, `o_\_23`)
  goes through local memory in T by T tiles
  (`transpose_tiled16_float`), read and written along the rows, the
  block arithmetic in `int` (a 64-bit division is emulated on a
  GPU and cost more than the gather saved). Measured on the Apple M1
  Max, 4096 by 4096 Floats, the kernel's own time: the gather 0.10
  ms, tiled 0.046 ms (8 or 16 alike), a plain copy 0.011 ms; at 2048
  by 2048, 0.018 against 0.012 ms. A rank-3 reversal (`o_\` of
  `[64 256 256]`) is not a swap of the last two axes and stays the
  gather, 0.021 ms. The Apple schedule file stays untiled (it is the
  defaults); the NVIDIA files tile, so their transposes go through
  local memory, to be measured there.
- **Empty values.** A device buffer holds at least one item, so an
  empty constant (`0 2 r_eshape` nothing, a loop's first
  accumulator) is planned as one zero that nothing reads; the plan
  thereby promises the runtime that every constant has its buffer's
  items or exactly one, and the runtime refuses a plan that breaks
  it by name rather than indexing (found 2026-10-16 with the world
  model's dream loop; the lowering folds such a `c_at` away first).
- **Semantics kept.** `idiv` and `mod` are floored
  (`xetal_idiv_long`, `xetal_imod_long`), Float to Int casts take
  `floor`, a number to Bool is `!= 0`, `abs` of a `long` avoids the
  unsigned `abs`.
- **Sizes.** An elementwise launch has one work-item per element,
  the global size rounded up to the work-group (bounded by the
  element count rounded to a power of two); a reduction launch has
  one work-group per partial.

The generated source for `libs/Accel/demos/threshold.xir` is one
kernel of ten C statements for the whole program; `explain` lists
every buffer (what it holds, who uploads it), every kernel (which
values it computes and writes) and every launch with its sizes.

## Not yet

A tree per line for very long axes, and where the weights live
(host or device) as a schedule field (the model saga).
