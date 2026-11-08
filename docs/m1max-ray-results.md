# Ray tracing on Apple Silicon: the M1 Max against the RTX 5060 Ti

Measured on the Mac (Apple M1 Max, 10 cores, 64 GB, macOS 26.5,
Apple OpenCL 1.2, no fp64) on 2026-11-07, at X_eTaL `abdd371`,
X_eTaL-libraries `8bd438d` and this repository at `6d4ce92`, with the
structured OpenCL regions backend (`--regions`) and the pinned
evaluator. The NVIDIA side is the curated evidence the Arch Linux lane
committed under `site-assets/ray/` (RTX 5060 Ti, driver 615.78.08);
nothing on that side was re-run here. The generated work lives under
`work/ray-*-m1max/` and is not tracked.

## What was run

| Step | Command | Result on the M1 Max |
| --- | --- | --- |
| Pinned evaluator, libraries, tool | `just xetal`, `just xetal-libraries`, `just build` | built; the lowering must be built after the X_eTaL checkout moves, not concurrently |
| Committed diagnostic on the device | `just ray-check-device opencl:0` | 16 triangle IDs exact, 36 linear and display RGB lines within 1e-5 against the evaluator baseline; 15.2 s for the whole recipe |
| 16 by 8 logo pair, same generated source on CPU and GPU | `just ray-image-pair --output work/ray-pair-m1max/logo` | passes: linear RGB within 2.5e-7, display RGB within 2.0e-7, identical RGB8 pixels |
| 640 by 256 logo on the GPU | `just ray-render-regions --width 640 --height 256 --repeat 3` | renders; see the timings below |
| 1280 by 512 logo on the GPU | `just ray-render-regions --width 1280 --height 512 --repeat 3` | renders; compared pixel for pixel with the RTX still |
| 640 by 256 logo on the CPU | `just ray-hires-cpu --libraries-ref 8bd438d... --width 640 --height 256 --jobs 8 --angle 0.0` | 476.4 s wall, 16 strips of 16 rows on 8 workers |
| 2 by 2 glass fixture at f32 | `just ray-render-regions --scene glass --float f32 --width 2 --height 2` | compiles (about 10 minutes in Apple's Metal compiler service), then the Glass library's own guard refuses: `Glass interface vectors must be unit length` |

The recipes call `python3`; they were run with a `uv venv` first on the
PATH so that the system interpreter was not used.

## Quality: the same pixels, with f32 edge cases

The 16 by 8 pair is bit for bit the curated RTX pair: the Apple
`logo.tif`, `logo.png` and `logo-cpu.tif` are byte-identical to
`site-assets/ray/image-pair/{gpu,gpu,cpu}.tif/png`, from the same
source digest (`5a660e...`), and the CPU comparison agrees to the same
order (RTX 3.4e-7 linear, Apple 2.5e-7).

At 1280 by 512 the Apple render and the RTX still
(`site-assets/ray/logo-gpu/still.tif`) have identical headers and
differ in 53 of 655,360 pixels:

| Where | Pixels | What differs |
| --- | --- | --- |
| High-contrast silhouette edges (neighbor contrast at least 20) | 21 | one side of an edge resolved the other way |
| Isolated pixels along the vane's shadow boundary, rows 255 to 282 | 32 | a lit orange pixel on one card is in shadow on the other (for example 234,152,87 against 72,47,29) |

Nothing differs anywhere else; the two images are indistinguishable to
the eye. These are shadow and hit tests that land within f32 rounding
of a surface, and the two vendors' f32 arithmetic (fused multiply-add
contraction, division and square-root rounding) decides them
differently. Across vendors f32 is not bit-exact unless the emitter
asks for it; the schedule could carry a strict f32 option (no
`FP_CONTRACT`, correctly rounded divide and sqrt, the OpenCL
`-cl-fp32-correctly-rounded-divide-sqrt` build flag), and a kernel
built with it on both cards would show whether the 53 pixels go away.
That is schedule data, not a change to the scene (plan A6, A7).

The same comparison on one machine, the Apple GPU at f32 against the
Apple CPU evaluator at f64 for the matched 640 by 256 logo, differs in
36 of 163,840 pixels, of the same kind. Mixed precision, as the
performance rules require it to be labeled.

## Speed: resident renders, cold setup, and the CPU

The logo, f32, one centered sample per pixel, angle 0, the second and
later resident calls of one prepared session:

| Image | RTX 5060 Ti (curated) | M1 Max (this run) |
| --- | --- | --- |
| 16 by 8, cold first call | 5074.8 ms (setup 5064.6) | 11878.0 ms (setup 11764.6) |
| 16 by 8, resident total | 4.1 ms | 73.8 ms |
| 640 by 256, cold first call | not recorded | 3150.5 ms (setup 2856.6) |
| 640 by 256, resident total | not recorded | 209.0, 221.1 ms (two calls) |
| 1280 by 512, cold first call | 5851.1 ms (setup 5003.9, device 738.9) | 4397.3 ms (setup 3405.6) |
| 1280 by 512, resident total | 1213.4 ms (device 890.9, download 87.4) | 890.5, 909.4 ms (host launches 876.8 to 901.9, download 6.8 to 13.1) |

Read these with three cautions:

- The programs are not the same version. The RTX stills were rendered
  from the display-only program by the backend at `97cad81` with a
  modified worktree (142 kernels, 189 to 196 launches, 24 guards). The
  backend at `6d4ce92` plans the identical 16 by 8 source as 1269
  kernels, 1506 launches and 112 guards, and the 1280 by 512 source
  as 1513 launches. The pixels agree, the resident time at 1280 by 512
  is comparable, but the launch count grew about eight times between
  the two commits. The structural acceptance in `ray-performance.md`
  (a small kernel set independent of geometry) should be re-checked at
  the current commit before either card's number is quoted.
- Apple's OpenCL event timing does not account for the kernels: it
  reports 14 ms of device time for a 1280 by 512 render whose host
  launch sequence takes 877 ms, and the launch sequence blocks on
  every guard flag read, so the host number bounds the real execution.
  On the RTX the device and host numbers agree (891 against 1126 ms).
  Compare resident totals, not the device column, across the two cards.
- At 16 by 8 the M1 Max is eighteen times slower than the RTX (73.8
  against 4.1 ms) while at 1280 by 512 it is about a quarter faster
  (890 against 1213 ms). The small image measures per-launch overhead
  (about 50 us a launch on Apple's driver, times 1506 launches); the
  large one measures tracing. Fewer launches would help Apple far more
  than NVIDIA.

Cold setup is Apple's `cl2Metal` translation of the generated C: 11.8
s the first time a program is seen and 2.9 to 3.4 s afterwards in this
session, against a steady 5.0 s on the RTX. Apple returns no reusable
binary (`docs/m1max-opencl-handoff.md`), so every process pays it.

## CPU against GPU on the same machine

The first matched native-size comparison at the pinned libraries, the
640 by 256 logo with identical geometry, camera, angle, one sample and
display settings:

| Side | Implementation | Time |
| --- | --- | --- |
| CPU | pinned `xetal` evaluator, f64, 16 full-camera strips of 16 rows on 8 workers | 476.4 s wall; 3518.9 strip-seconds; each strip 214 to 226 s including evaluator setup and text output |
| GPU | M1 Max through OpenCL regions, f32 | 4.5 s for the whole CLI process; 3.15 s first call; 209 to 221 ms resident |

The resident GPU call is about 2,200 times faster than the CPU wall
time, and the whole GPU process about 100 times. This is an f64
tree-walking evaluator against an f32 device, so it is a mixed-precision
comparison and says nothing about the evaluator's per-pixel cost (each
strip's 220 s includes setup and printing 16 rows of Floats). A strict
backend speedup needs the XIR interpreter at f32 on the CPU as the
other side, which `xetal-gpu run --device cpu` provides and this run
did not time.

The 16 by 8 CPU comparison process took 122.9 s on the M1 Max against
168.7 s on the Arch host for the same program: the Mac's faster single
core and SSD, not anything about the GPUs.

## Glass

The glass scene cannot run on this device as the libraries stand.
Apple's OpenCL has no fp64, and Glass pins its unit-vector, cosine and
Fresnel checks at relative 1e-10 and 1e-12 (`libs/Glass/src/Glass.xtl`
lines 37, 53 and 66 at `8bd438d`), which no f32 vector can meet (f32's
relative epsilon is 6e-8). The f32 attempt compiled, then stopped at
the first guard, correctly. Two routes, neither in this repository:

- RL11 in `docs/xetal-libraries-ask.md`: Glass takes its tolerance from
  the data's precision or as a parameter, so an f32 render validates at
  about 1e-6. The RTX f64 checkpoint (2 by 2, cap 4: 7.0 ms resident, 424 s
  uncached compile) stays the reference.
- G13's lint, which already lists a tolerance literal below about
  1e-5 compared against a Float computed from data: this is another
  instance, in a library rather than a program.

## What this establishes

The RayTrace path through structured OpenCL regions runs on the M1
Max from the same sources as on the RTX 5060 Ti with the same pixels
at small size and 53 f32 edge pixels at 1280 by 512. The Mac can
produce the Apple column of the matched CPU/GPU benchmark today for
the opaque scenes; the glass column waits on the library's tolerance.
The launch-count growth between the two backend commits is the one
open question for the NVIDIA numbers as well.
