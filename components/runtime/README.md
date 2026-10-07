# xetal-gpu-runtime

The OpenCL host runtime: find the devices, execute a plan on one of
them, read the results back.

```
xetal-gpu devices                                  # opencl:N, name, kind, version, units, work-group, memory, fp64
xetal-gpu run FILE.xir --device opencl:N [SCHEDULE] [--bind NAME=...]
```

What `execute` does, in order: picks the device (`opencl:N` counts
every device of every platform, in the order `devices` lists them);
refuses a plan that needs double precision on a device without it,
or a work-group larger than the device or the kernel allows (saying
which `--work-group` to pass); builds the plan's source with the
device's OpenCL compiler (its build log is the error when it
refuses); allocates every buffer of the plan in the element type the
schedule chose and uploads inputs and constants converted to it (an
X_eTaL Float to `float`, an Int to `long`, a Bool to `int`); runs the
launches in the plan's order on an in-order queue, each with its
global and local sizes; reads the outputs back and widens them to
the program's types (`float` to f64, `int` to Bool where the value is
a Bool).

The OpenCL library is loaded at run time (opencl3's `dynamic`
feature), so the tool builds and runs on a machine without OpenCL,
where `devices` reports none. The runtime's own tests run on device
0 when there is one and print that they were skipped otherwise.

## Platforms

- **macOS**: Apple's OpenCL (1.2, deprecated but present) is found
  without setup. This Mac: `Apple M1 Max (GPU, Apple; OpenCL 1.2), 32
  compute units, work-groups to 256, no fp64`, so the default
  schedule (Float as `float`) is the one that runs here.
- **Linux, old NVIDIA cards**: the driver's OpenCL (`libnvidia-opencl`)
  and an ICD loader (`ocl-icd`, package `ocl-icd` or
  `ocl-icd-libopencl1`); `clinfo` should list the card before
  `xetal-gpu devices` can. Kepler (K80) needs the 470 driver series,
  Maxwell and Pascal the 5xx series; both report OpenCL 1.2 (3.0 on
  newer drivers) and have fp64. The gpu-hardware saga records what each generation
  does.

## Sizes and timings

`just sizes` (`scripts/check-sizes.py`) generates Ints and Floats
into `work/sizes/`, runs an Accel program on them with the pinned
`xetal` (sum, largest, inner product, the pipeline, a Float sum and
inner product) and the same computation as an XIR program with
inputs bound to the same files, on the interpreter (exactly) and on
every device (Floats within 1e-5). The gate runs the two smaller
sizes and a 64 by 64 product. A size `mN` is an N by N product (Ints
in -9..9, exact; Floats in [0, 1)), its sum and largest item checked,
run on the device untiled and in 16 by 16 tiles. On this Mac,
whole-process times (each includes reading the data files, which
dominates the evaluator's small cases; the OpenCL runs include
compiling the kernels; speed is not a goal yet):

| work | xetal (s) | interpreter (s) | Apple M1 Max (s) | M1 Max, 16 by 16 tiles (s) |
| ---- | --------- | --------------- | ---------------- | -------------------------- |
| 1024 items | 0.07 | 0.00 | 0.08 | -- |
| 65536 items | 0.14 | 0.01 | 0.09 | -- |
| 1048576 items | 1.21 | 0.10 | 0.16 | -- |
| 64 by 64 product | 0.13 | 0.00 | 0.08 | 0.07 |
| 256 by 256 product | 3.48 | 0.04 | 0.08 | 0.09 |
| 512 by 512 product | 28.75 | 0.45 | 0.14 | 0.13 |

At 512 the GPU is about 200 times faster than the evaluator for the
whole program, as `docs/macros-and-extensions.md` expects for
products; tiling makes no difference yet at these sizes on this
device.

A million-item reduction is three launches (4096 partials, 16, 1);
the Float sums agree with the evaluator's f64 right fold within
1e-5 although the device adds `float`s as a tree.

## Checked

`scripts/check-equiv.sh` runs every twin on every device found and
compares with the evaluator's baseline: Ints exactly, Floats within
a relative 1e-5 (the device computes in `float` and reduces as a tree,
the evaluator in f64 from the right). On this Mac all five twins
agree, and `-0.0` comes back as `-0.0`, as the evaluator prints it.
