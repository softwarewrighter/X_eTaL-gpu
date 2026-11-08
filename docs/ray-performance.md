# Ray-tracing performance choices

## Goal and measurement rules

Deliver a native high-resolution image and recorded video, with measured
CPU versus GPU render time for the same X_eTaL algorithm. Target a
1280 by 512 or larger antialiased logo still before rendering a sequence.
The existing 160 by 64 movie is a low-resolution CPU preview.
The current GPU backend runs generated OpenCL compute kernels; it has
no RTX hardware traversal backend. These measurements therefore compare
our shared software tracer on CPU/GPU, without claiming RT-core use.

Hold geometry, camera, resolution, samples, seed, interaction budget and
display settings fixed across paired runs. Verify raw pixels before timing.
Record the CPU implementation and precision: the evaluator and XIR
interpreter are different CPU implementations. A Float64 evaluator versus
a Float32 GPU is a mixed-precision comparison and must be labeled; use
matching precision for a strict backend comparison.

Separate cold loading/lowering/build, geometry preparation, uploads,
warm rendering, GPU event intervals, readback and image/video encoding.
Report repeated medians and spread, including speedup or slowdown. A
single observed wall time during other work is not a benchmark. Keep
single-frame latency and total animation throughput separate. The sum of
concurrent frame wall times is not a measured serial baseline; run the
serial baseline independently rather than claiming speedup from that sum.

## Target execution model

The scene and image fit the device. A 1024 by 1024 Float32 RGB output
is 12 MiB; a 3840 by 2160 output is about 95 MiB. A hundred thousand
host kernel launches are a lowering limitation, not a requirement of
ray tracing. Arithmetic grows with rays, primitives and interactions;
that does not justify a dispatch per array operation or per primitive.

The primary fix is execution granularity: compile many independent ray
calculations into a small number of kernels. Each ray keeps its local
intersection/path state in registers and executes geometry/interaction
loops on the device. General map/loop/reduction fusion belongs in this
GPU backend; the tracing math remains shared X_eTaL library code. Do not
introduce a hand-written OpenCL renderer or new language syntax.

Use image tiles, for example 64 by 64 or 128 by 128 pixels, with smaller
hardware work-groups inside them. Many tiles can execute concurrently;
they can also be batched in one dispatch. Full-frame camera coordinates
and stable global pixel/sample IDs prevent projection changes or tile
seams. Shadows and secondary rays still query the full scene. CPU tiles
run the same shared computation. Repeating the old fine-grained launch
graph separately for every tile would increase overhead.

Keep scene data, cached geometry, path state and sample accumulations on
the GPU. Update only changed camera/vane parameters; transform geometry
on the device when needed. A bounded ring of GPU frame buffers can retain
completed frames while subsequent frames render. Read back a finished
frame only when export or a CPU consumer needs it. Overlap that readback
and TIFF/video encoding with later rendering using event dependencies.
Ring capacity is scheduling data, not a promise to retain an unlimited
animation in memory.

On one GPU, intermediate traffic uses device-local memory bandwidth.
The current path already keeps intermediate arrays on the device; its
problem is frequent global-memory spills plus launches and blocking
reads of tiny guard flags. Those flag bytes are small, but their round
trips synchronize the pipeline. Buffer reuse reduces capacity; it does
not remove the traffic or synchronization.

Validation must remain safe: check before invalid arithmetic/indexing on
the device, collect errors with deterministic failure ordering, and check
the resulting error status at a batch/frame boundary. Shared scene/input
preflight may run once per changed scene. Do not simply postpone all
checks until after unsafe work or discard guards to reduce launch count.

The target is a small fixed set of tracing kernels per phase, independent
of primitive count, with no intermediate CPU round trips. A wavefront
pipeline may use several kernels per interaction; it must keep queues
and ray state resident. Compare this with a fused per-ray loop rather
than assuming either wins. Retain an explicit CPU reference and compare
raw pixels, tile order, edge tiles, sample identity, frame-ring wrap and
error recovery before timing.

Current status: transient allocation reuse and resident prepared sessions
exist. Prepared execution still reads outputs each call and guards still
block the host; fused tracing loops, resident frame-result handling and
asynchronous export are subsequent work. RL10's bounded library queries
are complementary; they do not replace the backend granularity fix.

## Current measured constraints

The old two-triangle audit has 23,770 IR lines and 1,585 ordered guards.
Incremental checking and reuse of identical immutable SSA expressions
reduce the pristine diagnostic to 4,541 lines and 310 guards. This is
about an 81 percent structural reduction, not a measured render speedup.
The optimized default OpenCL plan still has 2,233 kernel launches for
only sixteen primary rays. Ordered guards require blocking flag reads;
launch and synchronization costs can dominate such a small workload.

A one-pixel probe of the actual 322-triangle logo produces 258,581 IR
lines and 16,817 guards even after expression reuse. Geometry preparation
falls back explicitly for r_ev and g_rade; no larger GPU render is claimed.
This makes bounded vectorized primitive batches a concrete library ask
(RL10), alongside local buffer/synchronization work.

The public library nearest-hit sweep holds one triangle/ray batch and
selected previous hits. It already avoids allocating every ray against
every triangle simultaneously. However, its static primitive loop is
unrolled by the current lowering, so IR size and launch count grow with
the scene. Geometry batching alone does not fix this compiler expansion.

## Optimization order

| Choice | Benefit | Measurement or constraint |
| ------ | ------- | ------------------------- |
| Incremental checking and expression reuse | Avoid rechecking/cloning the entire IR per value; share repeated calculations and identical guards | Keep guard ordering/messages and rebound-input failures; compare generated work and real output |
| Resident prepared sessions | Reuse compiled kernels, geometry/material buffers and allocations | Separate preparation from warm frames; measure changed-input upload and readback |
| Ray batches or image tiles | Bound intermediate buffers while giving the GPU enough work | Sweep about 1K, 4K, 16K and 64K rays, subject to memory; preserve global pixel/sample IDs |
| Prepared geometry | Cache triangle edges, normals and bounds rather than rebuilding them per ray | Shared library preparation, refreshed when transforms change; include its cost in cold/per-frame timing |
| Shadow any-hit and inactive paths | Avoid finding a full nearest hit when only opaque visibility is needed; stop finished paths | Preserve transparent-shadow semantics; profile divergence versus compaction overhead |
| Memory layout and coherent tiles | Improve coalesced loads and reuse for nearby primary rays | Backend scheduling keeps logical arrays unchanged; secondary rays may be less coherent |
| Static versus per-frame data | Upload only camera/transforms or changed geometry | Reuse stationary X, floor and materials; do not change scene semantics to avoid transfer |
| Buffer lifetime reuse | Reuse storage after its last reader instead of retaining all intermediates | Measure peak memory; keep pending kernels, guard dependencies and outputs valid |
| More fusion and fewer synchronization points | Reduce launch/flag-read overhead | Preserve the first failing guard before unsafe arithmetic; do not merely delete validation |
| Bounded loop execution | Avoid an IR/kernel expansion for every primitive or interaction | General GPU IR/lowering work here; retain the same shared library algorithm |
| Shared acceleration structures | Reduce primitive tests as scenes grow | Profile bounds/BVH build and traversal; general library work, no renderer-specific language primitive |
| Parallel CPU frames | Improve sequence throughput using independent frames | Sweep worker counts 1/2/4 and memory/core limits; report total throughput separately from per-frame latency |
| GPU batch/frame concurrency | Possibly overlap transfers and compute | Compare one resident queue against small bounded concurrency; extra queues may contend and multiply memory |
| Encoding pipeline | Overlap completed-frame writing/encoding with subsequent computation | Keep encoding outside renderer-only timing; retain uncompressed TIFF masters |

Measure batching before scheduling multiple full frames on one GPU.
One sufficiently large batch can already occupy the device. Concurrent
frames duplicate state, consume memory, and do not guarantee throughput.
Multi-GPU frame distribution is a later independent option, with each
GPU's settings and timings reported explicitly.

## Shared scene and sampling requirements

CPU and GPU execute the same X_eTaL transport. Scheduling choices belong
in this GPU repository; intersection, sampling and any future BVH functions
belong in general libraries. A batch is an execution partition, not a
change in scene or image quality. Do not reset sample identity at a tile
boundary. Parallel frame order must not change deterministic samples.

For animation, a static local-space vane could use instance transforms
and a reusable acceleration structure rather than rebuilding its geometry
for every angle. Adopt that only after shared transformed-ray/normal and
nearest-hit equivalence tests. Keep the X stationary and preserve object
and material identities.

Higher resolution and antialiasing both increase ray work. First inspect
one still at its intended display size, then sweep resolution and samples,
then render the sequence. Do not upscale the low-resolution recording and
present that as a new high-resolution render. Adaptive sampling, denoising,
reduced ray budgets or simpler materials change the quality comparison;
they need explicit matched baselines rather than hidden benchmark shortcuts.

## Reproduce the current diagnostic audit

```sh
just ray-check-device
wc -l work/ray-integration/diagnostic-source.xir
rg -c ' = guard ' work/ray-integration/diagnostic-source.xir
```

The generated files remain ignored work output. The command compares
all diagnostic IDs, linear RGB and display RGB against upstream CPU
outputs. Actual bound-input tests are in components/cli/tests/ray_inputs.rs;
they mutate geometry, origins, directions and materials in a prepared
OpenCL session, check invalid data, then recover with valid inputs.

## Physical buffer reuse

OpenCL prepared sessions now reuse transient allocations when their
logical values have disjoint lifetimes. Inputs and constants stay
resident and distinct. Outputs remain live until readback. Guard reads
extend a condition's lifetime, and a kernel's readers cannot alias its
writers. This relies on the existing in-order queue and write-all kernels.
Repeated runs start at the beginning and overwrite transient values.

| Graph, default Float32/Int64 schedule | Logical buffers | Physical allocations | Unpooled payload bytes | Reused payload bytes |
| ------------------------------------ | --------------- | -------------------- | ---------------------- | -------------------- |
| Two triangles, 16 rays | 2481 | 87 | 280284 | 13212 |
| Full 322-triangle logo, one ray | 142866 | 73 | 1381624 | 83516 |
| Full logo, 4 by 2 rays | 143510 | 78 | 8941960 | 86728 |

These are planned buffer payloads, including minimum dummy allocations;
they exclude driver allocation granularity, compiled kernels, events,
source text and build workspaces. They are not measured total VRAM or
render speedups. The 4 by 2 graph still has 129916 launches and 16817
guards, with 99535 kernel definitions and 39650839 bytes of generated
OpenCL source. Buffer reuse addresses memory, not that remaining launch
and build cost.
The earlier 2230 diagnostic figure counted launches before the final
guard; the full plan has 2233 launches, including display conversion.

```sh
just ray-memory
just ray-memory work/ray-integration/logo-small.xir
```

The report builds a plan without allocating GPU buffers. Text parsing
now indexes SSA names, and the OpenCL emitter indexes shared kernels,
avoiding repeated scans of growing programs/source during large reports.
Existing generated-source and plan snapshots stay unchanged.

The RL7 a2d4de1 trial passed twenty existing CPU baselines and additional
lens/area-light checks. Strict bound-input RayTrace GPU lowering refused
the new data-dependent nearest-query branch. The trial was not adopted;
8bd438d remains pinned. RL10 needs a compatible bounded vectorized path
without that dynamic-shape requirement. Shared validation stays intact.

## Reviewed design references and next implementation

The following primary sources were reviewed on 2026-11-06. These are
architecture references; no external renderer/compiler implementation
is copied or introduced as a dependency.

- [Futhark regular flattening](https://futhark-book.readthedocs.io/en/latest/regular-flattening.html)
  describes choosing an outer parallel map with an inner sequential loop.
  That is the initial model for independent rays with geometry loops.
- [Futhark performance guide](https://futhark.readthedocs.io/en/latest/performance.html)
  explains producer/consumer fusion and the overhead of parallelizing tiny
  arrays. Our three-component vector operations should become local
  arithmetic, rather than independent global reductions/materializations.
- [Halide GPU tutorial](https://halide-lang.org/docs/tutorial/lesson_12_using_the_gpu.html)
  separates the algorithm from CPU/GPU scheduling, fuses work within tiles,
  and retains device data across repeated realizations. This informs tile
  scheduling and resident execution, not the ray-tracing math.
- [PBRT GPU execution models](https://www.pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Mapping_Path_Tracing_to_the_GPU)
  explains fused-kernel versus wavefront tradeoffs. Large fused kernels
  can lose occupancy and suffer divergent execution; wavefront stages
  improve scheduling but add queue/state memory traffic. Neither is an
  automatic winner for all three scenes.

The initial backend prototype will preserve a structured map/loop region
in XIR instead of expanding p_ower across geometry during lowering.
Scene arrays are immutable captures; ray-local fixed-size values become
local/register arithmetic. Generate the repeated body once, with device
loop bounds. Keep this a general backend capability over shared X_eTaL
functions, without renderer-specific native kernels or new core syntax.

First prove fixed-shape loop/map correctness and safe error propagation;
then use the actual library intersection/shading functions. Target a small
fixed dispatch set for the simple opaque prototype, with primitive-count
changes affecting work rather than kernel count/source size. Validate
zero iterations, capture/state types, input changes, first failure,
partial tiles and raw CPU/GPU pixels. A kernel-count regression test must
compare multiple primitive counts. Buffer reuse alone is not acceptance.

Benchmark a fused per-ray baseline against coarse wavefront stages when
material/path divergence warrants it. Report register/private-memory
costs, launches, transfers, cold preparation and warm rendering. Keep
scene, ray state, accumulations and frame buffers resident; add frame-ring
and asynchronous export support after the first correct tracing region.
The first high-resolution CPU still is independent of this backend work.

## Prototype milestones from the design review

The reviewed references motivate a specific first experiment, not a
promise that one large shader is always best:

1. Add general fixed-shape map/loop regions to the local GPU IR, checker,
   CPU interpreter and emitter. Preserve p_ower as a device loop within
   a mapped region rather than expanding its body per primitive.
2. Keep small coordinate vectors and reductions local to a ray. Do not
   turn a three-component dot product into a global reduction launch.
   Keep immutable scene arrays as device captures, not per-ray copies.
3. Preserve safe per-ray checks and deterministic first failure. A
   failing lane must stop before unsafe arithmetic/indexing. Aggregate
   its error status at the batch boundary; removing checks is not valid.
4. Prove generic loops first, then actual library nearest-hit/shading.
   Start with a fused ray-local baseline for the uniform opaque logo.
   Keep geometry count in a device loop bound, without emitting a kernel
   function for every triangle or every intermediate operation.
5. Compare with coarse intersection/shading/shadow/accumulation stages
   for more divergent glass paths. Keep those queues and states resident.
   The choice follows measured register pressure, memory traffic and time.

The first performance acceptance is structural: kernel count must stay
small as geometry grows from a tiny scene to the full logo. Code size
must not grow in proportion to triangle count, and ray-count changes
must change work/grid size rather than duplicate the shader. Then verify
raw CPU/GPU pixels, invalid-input recovery and sample/tile invariance.
Measure cold preparation and repeated warm render calls separately.
Frame-ring/export overlap follows the first correct region; it must not
hold up a visible still. No native renderer or extra language syntax is
part of this plan.

## Structured region prototype: RTX results

The opt-in OpenCL path now retains fixed-shape p_ower bodies in XIR.
Scalar e_ach maps independent items to work-items; nested loops and small
vectors/reductions stay inside that work-item. Immutable captures remain
shared device arrays. Total expressions independent of local state may
be hoisted; guards and potentially failing indexing/division stay ordered.
Each lane stops on its first failed check. One region-boundary read of
lane error codes selects the first failing lane, preserving mapped order.
This first version downloads the lane flags; compact device reduction is
still pending. Existing top-level guards retain their earlier behavior.

The actual shared nearest-hit function was tested with 2, 44 and 322
triangles: 10 launches in every case, with generated source sizes of
44,559, 44,572 and 44,585 bytes. Hit IDs agree exactly with the interpreter;
invalid directions fail and valid rebinding recovers. Full shared shading
also agrees with the CPU batch implementation. Generic checks cover
nested loops, capture changes, empty maps, zero iterations, partial work
groups and failure ordering. Source serialization round trips regions.

A native 1280 by 512 logo now renders on RTX 5060 Ti through the same
RayTrace functions. The complete scene plan has 196 launches, 142 kernel
definitions, 191,125 source bytes and 107,560,988 planned buffer bytes.
The previous 4 by 2 unrolled logo had 129,916 launches and approximately
40 MB of source. These are structural comparisons at different image
sizes, not a matched speedup measurement. Buffers exclude driver/private
memory. The full scene includes geometry/camera preparation and 24 guard
boundaries, beyond the single mapped tracing kernel.

| Observation | Time |
| --- | --- |
| Initial GPU setup | 5003.902 ms |
| Initial upload | 22.333 ms |
| First device execution | 738.948 ms |
| First total GPU call | 5851.123 ms |
| Second resident device execution | 890.943 ms |
| Second resident total, including output readback | 1213.376 ms |
| Second input upload | 0 ms |

These are individual observations, not a statistical benchmark. The CLI
process including load and ASCII output took 10.032 seconds for two runs.
An 8 by 4 image from the exact generated source passed against the pinned
X_eTaL evaluator (maximum display-RGB error 1.6812e-7). That evaluator
process took 43.082 seconds including load/output. No full-resolution
matched CPU speedup is claimed: the published CPU still uses a different
library snapshot and parallel strips.

Run just ray-render-regions --width 1280 --height 512 after just build.
For the smaller matched-source check, add --width 8 --height 4
--compare-cpu. Output includes uncompressed TIFF, PNG, generated source,
plan, memory report and cold/resident timings under work/ray-regions/.
The gallery preserves the observed frame, logs and small comparison.

The backend flag is --regions; ordinary lowering and Mojo are unchanged.
Mojo explicitly refuses a structured region until its emitter supports
one. Local matmul/top-k are currently refused; this is a bounded prototype,
not complete nested-array support. The GPU adapter returns scalar RGB
components because current e_ach results must be scalar. It traces each
pixel three times; removing that duplication requires a validated mapped
cell/multiple-result design. Standalone repeat owns one work-item's whole
state; the outer each supplies ray parallelism. Do not interpret standalone
repeat over a full-frame state as a parallel ray schedule.

Queued optimizations are channel duplication, compact error reduction,
bounded tiles, resident output/frame-ring/export APIs and matched
useful-size timings. Glass and Cornell still integration comes first now
that the GPU path is fast enough for stills. The longer animation remains
pending.
No new language syntax or renderer-native implementation was introduced.

The current image driver additionally returns raw linear RGB and display
RGB from the same program, and --compare-cpu writes TIFF/PNG for both
backends. The preserved 16 by 8 pair agrees to 3.4462e-7 in linear RGB
and 1.8325e-7 in display RGB, with identical quantized pixels. Its CPU
process took 168.695 seconds including setup/text; its second GPU call
was 4.084 ms including readback. These tiny-image observations validate
execution/output only and are not a full-resolution speedup benchmark.
Exporting raw linear data adds an output readback; the earlier published
1280 by 512 timings remain tied to their archived display-only program.


## Glass transport checkpoint

The shared Scene v2/Glass path now runs through opt-in OpenCL regions.
See [glass transport evidence](glass-scene-plan.md#transport-checkpoint-2026-11-07).
The four-interaction 2 by 2 f64 fixture planned 244 kernels, 322 launches,
38 guards, about 24 KB of reusable buffers and 3.34 MB of generated C.
Resident execution was about 7 ms device / 10 ms total; an uncached build
took about 424 seconds. A later cached setup took about 800 ms. These are
correctness experiments, not representative throughput measurements.

The next optimization target is retaining tuple transport state in a
single bounded device loop, followed by vector RGB results from one trace.
Both address shader size and duplicate work; more host batching alone does
not remove cold compilation. Neither optimization requires a new language
primitive or changing the reusable optics library.

## Apple M1 Max

The same regions path runs on the M1 Max through Apple's OpenCL 1.2:
the 16 by 8 pair is byte-identical to the RTX one, the 1280 by 512
logo differs from the RTX still in 53 f32 edge pixels and renders in
0.89 to 0.91 s resident, and the matched 640 by 256 logo takes 476 s
on the Mac's CPU evaluator against 0.21 s resident on its GPU. Apple's
event timing does not cover the kernels, the per-launch overhead is
about 50 us, and glass is refused by Glass's f64 tolerances. The
numbers, the cautions and the launch-count question are in
[m1max-ray-results.md](m1max-ray-results.md).
