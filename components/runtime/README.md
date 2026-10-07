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
  newer drivers) and have fp64. Saga 4 records what each generation
  does.

## Checked

`scripts/check-equiv.sh` runs every twin on every device found and
compares with the evaluator's baseline: Ints exactly, Floats within
a relative 1e-5 (the device computes in `float` and reduces as a tree,
the evaluator in f64 from the right). On this Mac all five twins
agree, and `-0.0` comes back as `-0.0`, as the evaluator prints it.
