# Accel

The acceleratable subset of X_eTaL: elementwise arithmetic, masks and
a select, reductions. Each function is one whole-array expression in
the subset research7 (`docs/research7.txt`) says an accelerator
should accept first: static scalar types, pure functions, arithmetic
and comparisons item by item with scalar extension, a select by
mask, reductions. Nothing recursive, nested, dynamic or effectful.

```
"ac:" u_se< "Accel"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `ac:` is the
recommended alias.

## Conventions

- `ac:s_um`, `ac:l_argest` and `ac:d_ot` collapse a vector to one
  number; `ac:r_owSums`, `ac:c_olSums` and `ac:r_owMax` work along an
  axis of a matrix (`r_/_2` along each row, `r_/` down the columns).
- A function works on Ints or Floats alike where the type says
  `Num a`; `ac:t_hreshold` is Float-only because a Bool mask converts
  only to the type of the literal it meets (`x * f_loat x > t`), and
  `ac:k_eep` takes an Int mask for Int arrays.
- A select is a multiplication by a mask, so a negative Float that
  is masked out comes back as `-0.0` (what `-0.75 * 0.0` is); a GPU
  computes the same.
- The evaluator is the specification: `tests/` and `demos/` carry
  the outputs the GPU must reproduce (Ints exactly, Floats within a
  tolerance, since a device computes in f32).

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `x ac:v_add y` | `Num a => a -> a -> a` | the sum, item by item |
| `a ac:s_cale x` | `Num a => a -> a -> a` | every item times a |
| `ac:r_elu x` | `Num a => a -> a` | x where positive, 0 elsewhere |
| `t ac:t_hreshold x` | `Float -> Float -> Float` | x where x > t, 0.0 elsewhere |
| `m ac:k_eep x` | `Num a => a -> a -> a` | x where the mask m is 1, 0 elsewhere |
| `ac:s_um x` | `Num a => a -> a` | the sum of a vector |
| `ac:l_argest x` | `Num a => a -> a` | the largest item |
| `x ac:d_ot y` | `Num a => a -> a -> a` | the inner product of two vectors |
| `ac:r_owSums m` | `Num a => a -> a` | the sum of each row (`'+ r_/_2 m`) |
| `ac:c_olSums m` | `Num a => a -> a` | the sum of each column (`'+ r_/ m`) |
| `ac:r_owMax m` | `Num a => a -> a` | the largest item of each row |
| `x ac:p_ipeline w` | `Num a => a -> a -> a` | the sum of the positive items of x + x * w: multiply, add, select, reduce |

## Examples

From `../tests/basics.xtl`:

```
      1 2 3 4 ac:v_add 10 20 30 40
11 22 33 44
      3 ac:s_cale 1 2 3
3 6 9
      ac:r_elu 1 -2 3 -4
1 0 3 0
      0.5 ac:t_hreshold 0.25 0.5 0.75 1.0
0.0 0.0 0.75 1.0
      (1 0 1 0) ac:k_eep 5 6 7 8
5 0 7 0
      ac:s_um 1 2 3 4
10
      ac:l_argest 3 9 2
9
      1 2 3 ac:d_ot 4 5 6
32
      1 2 3 4 ac:p_ipeline 1 -2 1 -2
8
      m := 2 3 r_eshape 1 20 3 4 5 60
      ac:r_owSums m
24 69
      ac:c_olSums m
5 25 63
      ac:r_owMax m
20 60
```

## Demos

The example programs, each the reference for a GPU run:

- [`demos/vector-add.xtl`](../demos/vector-add.xtl): `c := a + b`,
  Ints and Floats; the first kernel, one work-item per element.
- [`demos/saxpy.xtl`](../demos/saxpy.xtl): `a x + y`, a scale then
  an add, which the GPU fuses into one kernel.
- [`demos/threshold.xtl`](../demos/threshold.xtl): the select by
  mask three ways: a threshold, relu, an arbitrary Int mask.
- [`demos/reduce-sum.xtl`](../demos/reduce-sum.xtl): research7's
  GPU PoC 0 (`a`, `b := 2 a`, `c := a + b`, `+/ c`), the largest
  item, an inner product.
- [`demos/axis-reduce.xtl`](../demos/axis-reduce.xtl): row sums,
  column sums and row maxima of a matrix, Ints and Floats; one GPU
  work-item per result.
- [`demos/pipeline.xtl`](../demos/pipeline.xtl): the acceptance
  pipeline, multiply, add, select, reduce, as one expression.

## Provenance

Written for X_eTaL-gpu, after `docs/research7.txt` ("GPU PoC #0
should be ridiculously small"; "the first end-to-end acceptance test:
input arrays, multiply, add, threshold/select, reduce").
