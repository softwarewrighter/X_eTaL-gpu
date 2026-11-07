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

## Current measured constraints

The old two-triangle audit has 23,770 IR lines and 1,585 ordered guards.
Incremental checking and reuse of identical immutable SSA expressions
reduce the pristine diagnostic to 4,541 lines and 310 guards. This is
about an 81 percent structural reduction, not a measured render speedup.
The optimized default OpenCL plan still has 2,230 kernel launches for
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
