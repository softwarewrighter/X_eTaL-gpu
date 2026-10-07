# xetal-gpu-opencl

The schedule and the OpenCL C 1.2 emitter: from a checked XIR program
to kernel source and a launch plan. Nothing here touches a device;
the runtime component executes a plan, and this crate's tests pin
the source and the plan of every twin in `libs/*/demos/`
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
| `tile` | 0 | inner products in T by T tiles in local memory; 0 untiled |

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
| `nvidia-modern.toml` | the coming Arch machine's newer NVIDIA GPU: work-groups of 1024, 16 by 16 tiles | not yet |
| `nvidia-modern-f64.toml` | the same, Float as `double` | not yet |
| `nvidia-legacy.toml` | Kepler, Maxwell, Pascal cards | not yet |

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
- **Structure.** A `take` or `drop` keeps contiguous rows, so it is
  one copy kernel (`copy_float`) from an offset; a `ravel` is the same
  buffer read as a vector, with no launch; a `table` is a kernel of
  its own, one work-item per result item reading `a[k / nb]` and
  `b[k % nb]` (a literal for a single value).
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
