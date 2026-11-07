# Ray tracing in X_eTaL

## Goal and boundaries

Implement an original ray tracer as ordinary X_eTaL array code. Run the
same source and scene on CPU and GPU, produce uncompressed images, and
measure whether the GPU is faster. The first GPU is the RTX 5060 Ti
(16 GB) on large12. Other OpenCL devices remain possible.

The rendering algorithm belongs in X_eTaL: camera rays, intersections,
nearest-hit selection, normals, lighting, shadow visibility, reflections,
and pixel values. The host supplies arrays, selects the device, manages
execution and writes files. No host implementation of the renderer, no
Blender rendering or Blender extension, and no external rendering engine.
Use the existing source-to-XIR-to-OpenCL path directly. OpenCL compute
does not automatically use RTX RT cores; an RT-core backend is outside
this plan. No CUDA or OptiX dependency is required.

POV-Ray is a reference for capabilities and visual comparison only.
Do not copy its implementation or bundled assets. Avoid GPL/AGPL
implementation dependencies; review new dependencies and asset licenses.
POV-Ray 3.7 onward uses AGPLv3 or later; older releases had custom
licenses. References: [official license](https://www.povray.org/povlegal.html)
and [older license](https://www.povray.org/povlegal-3.6.html).

## Integration progress

RL1 and RL4 have committed deliveries at X_eTaL-libraries c968326.
The committed CPU diagnostic and six other examples match their baselines
under X_eTaL abdd371; a 64 by 64 uncompressed TIFF preview was produced.
RL2 extrusion/glyphs (f28e7ca) and RL3 scene macros (6e58863) are now
pinned at handoff 7bd8f0a. A reproducible CPU logo TIFF/PNG is available
with just ray-render-cpu. The tiny 4x4 diagnostic now agrees on XIR CPU and RTX OpenCL, including
its guards, shadows and one reflection. Larger scenes and changed render
inputs still need integration/performance validation. The readable a/e
fix from 811cadd is now adopted at e7140be for the recorded gallery.
The first gallery animation uses CPU frames, an outward shaft/arrowhead
and the correct pivot through stationary X. Glass and benchmark entries
remain planned; see the recorded-gallery section in ray-integration.md. See [ray-integration.md](ray-integration.md)
for evidence and reproducible CPU checks. Local adoption/source/device/
image work is prioritized before remaining Mojo/ML steps; it does not
wait for the extrusion or scene macros.

## Library deliverables

The approved library requests are RL1 (3D math), RL2 (extrusion and glyph
macro), RL3 (scene DSL and transform groups) and RL4 (ray-tracing functions),
specified in [xetal-libraries-ask.md](xetal-libraries-ask.md). They are
requested locally for owner triage; no upstream completion is claimed.
RL1 and primary RL4 functions are needed before ray-source finishes;
RL2/RL3 and RL4 shadows/reflections are needed before ray-logo. These are
milestone dependencies, not calendar commitments. GPU capability checks
and the image/benchmark harness can progress before the asks land.

Use the combined macro design: one small scene DSL plus a reusable
extrusion macro. Compile-time macros emit numeric geometry/material/group
arrays; ordinary X_eTaL applies the vane's frame transform and renders.
No new language feature is requested. Pin committed library deliveries
before closing the milestone that consumes them.

## Scene and animation

A thick, extruded X stays fixed at the center. The letters `_eTaL`
extend outward as a rigid vane, rotating around the vertical axis through
the X. The camera stays fixed. This interpretation can be adjusted after
a still preview; separate radial letter arms are a possible alternative.
Use original polygon outlines extruded into triangles, with an explicit
underscore, consistent winding and normals. Keep geometry as scene data;
compute animated transforms in X_eTaL. Use contrasting materials, a floor
and one light so shadows and reflections expose the shape.

Start with a sphere/triangle diagnostic scene, then the logo at 640 by
360. Render a seamless revolution with angle 2*pi*frame/frame_count;
omit the duplicate endpoint. Proposed defaults: 240 frames at 30 fps.
The primary artifacts are uncompressed image frames. A compressed video
may be an optional derivative, never the only output.

## Algorithm and representation

Reusable ray-tracing math and scene notation belong in X_eTaL-libraries
(or an existing general library), not permanently in this GPU repository.
Audit existing library exports before proposing a new package. Request
new or changed libraries in [xetal-libraries-ask.md](xetal-libraries-ask.md),
with ordinary functions and especially macro libraries considered before
any language change. A macro should expand useful scene or geometry
notation into ordinary X_eTaL; it must not hide a native renderer.

Keep GPU integration demos, equivalence fixtures, device schedules,
benchmarks and execution support here. Use a reviewed, pinned general
library dependency once available. A local algorithm prototype is a
clearly labeled temporary fixture with an ownership/migration entry in
the library ask ledger, not a second permanent general library. The
owning library follows the standard documented source, tests, pinned
export types and demos layout. Lower the same source on CPU and GPU;
do not write hand-maintained GPU kernels or IR twins.

Represent rays, triangle vertices, material properties and hit records
as fixed-shape arrays. Establish the representation with a tiny evaluator
prototype before promising particular syntax. Audit normalization, square
root, cross/dot products, masks, gathers, nearest-hit reductions and fixed
iteration against the actual lowering. Tuples may be used only where the
pinned execution path supports them. Do not require xetal-x or a native
rendering extension.

Use a pinhole camera and Moller-Trumbore triangle intersections, selecting
the nearest positive hit. Define deterministic tie handling, background,
ray-origin offsets, parallel-ray thresholds and degenerate triangles.
Mask invalid denominators before division; avoid infinity sentinels and
NaNs. Start with ambient plus diffuse lighting and hard shadow rays.
Add one bounded reflection bounce after primary rays and shadows agree.
Use deterministic sampling; antialiasing can follow the first image.

Batch pixels and triangles to bound memory: never require the full
pixels-times-triangles Cartesian product at production resolution.
Preserve nearest-hit state across triangle batches with a deterministic
rule. Chunk sizes and device widths belong to schedules or runner data,
not rendering semantics. Introduce a BVH only after measuring the simple
implementation, with a separate planned step if needed.

A missing lowering or backend implementation of an existing operation
is GPU work here: extend the checker, interpreter, emitter or lowering
as appropriate, using the existing architecture and tests. A missing
reusable function or notation is a library ask first, with macro libraries
preferred for new notation. Do not add a language primitive just because
the ray tracer needs a new algorithm. Reserve `docs/xetal-asks.md` mainly
for demonstrated language bugs, with minimal reproductions; an exceptional
core request must explain why libraries and macros cannot express it.
Do not edit the pinned language checkout or another repository here.

## Image output

Return a row-major height-by-width-by-3 linear RGB Float array. Specify
clamping and conversion to display RGB in shared X_eTaL code. Write
uncompressed RGB TIFF (Compression=1), initially 8 bits per channel,
with explicit dimensions, channel order, orientation and transfer curve.
Keep raw linear Float arrays for numerical comparisons before quantizing.
A minimal binary PPM (P6, uncompressed) is acceptable as a diagnostic,
but TIFF is the delivery target. File encoding belongs to a host utility,
not the renderer. Reject mismatched shapes and invalid output values.
Test TIFF tags, pixel bytes and round-trip decoding independently.

## Correctness and performance

Use the pinned evaluator as the semantic reference, the XIR interpreter
as another CPU path, and OpenCL on the GPU. Name every CPU implementation
in reports: evaluator, interpreter, or compiled CPU backend if available.
Do not describe an interpreter comparison as proof against optimized
native CPU rendering. Both timed paths execute the same X_eTaL algorithm,
scene, sample count, bounce count and image dimensions.

Validate tiny analytical cases (hit/miss, edge, parallel, inside, occluded,
reflection and tie), then small images and several animation angles.
Compare raw pixels with stated absolute and relative tolerances, recording
maximum error and failed-pixel counts. Use exact checks for IDs/masks
where unambiguous; report boundary-sensitive cases explicitly. Start from
the project's f32 tolerance, establish an image-specific absolute bound
from evidence, and record both before accepting a full image. Never use
visual similarity alone or adjust tolerances silently to pass.

Measure CPU wall time and GPU device-event time separately. Also report
host render-call wall time including uploads, launches and readback, and
whole-process time. Report cold compilation/setup, warmups and repeated
runs separately; synchronize before stopping a timer. Exclude file writing
from rendering speedups and report file-writing time separately. Record
resident-scene versus refreshed-scene measurements, precision, CPU thread
count, pins, drivers, schedules, resolution, triangle count, sampling,
bounces, repetition count, median and spread. Only compare matched
precision for a strict backend speedup; label evaluator f64 versus GPU f32
comparisons. Report speedup as CPU time / GPU time, including slowdowns.
Sweep small to larger resolutions; do not assume a crossover exists.

## Second scene: glass spheres and crystals

After the initial renderer/logo work, build the reference-inspired still
in [glass-scene-plan.md](glass-scene-plan.md): clear spheres and faceted
crystals, dark reflective floor, bright panels and shallow focus. This
is a staged extension, not a claim that v1 renders glass. Library asks
RL5-RL7 add compatible v2 optics, geometry, camera and light support;
RL8/RL9 are optional caustic/finishing proposals and do not block the
first glass image. The same X_eTaL source, deterministic samples and
settings run on CPU/GPU and produce uncompressed TIFF. Existing v1
materials, logo milestones and library asks remain unchanged.

## Implementation sequence and acceptance

These are the ray-tracing workstream within the existing active saga;
the agentrail steps carry the `ray-` prefix. Existing Mojo and ML work
remains queued; appending these steps does not claim priority over it.

| Step | Delivery and acceptance |
| ---- | ----------------------- |
| ray-source | Audit general libraries and macros first; request reusable camera/intersection/lighting APIs in the library repo; tiny evaluator baselines and pinned types, with GPU prerequisites planned here as needed. |
| ray-device | Same source lowers and runs on the XIR CPU interpreter and RTX OpenCL; analytical tests and small raw images agree; bounded batching and memory checks. |
| ray-image | Shared display conversion and host uncompressed TIFF writer; deterministic still image through both paths, decoder validation and documented recipes. |
| ray-logo | Original extruded X and `_eTaL` geometry, stationary X and rotating vane; shadows and one reflection bounce; multiple angles agree on CPU/GPU. |
| ray-bench | Reproducible CPU/GPU sweep with correctness checks, cold/warm/device/end-to-end timing and honest speedup or slowdown report on RTX 5060 Ti. |
| ray-animation | Seamless uncompressed TIFF frame sequence, manifest and reproducible commands; sampled-frame CPU/GPU verification and optional video derivative. |

Every step adds appropriate tests and `just` recipes, runs `just gate`,
updates library/component docs and status, commits source and saga state,
completes only its own step and pushes. Ignore generated images and timing
scratch output; commit small numerical fixtures and reports. Publish and
check the live site when its displayed content changes. The planning step
creates this roadmap, not a working renderer or measured performance.

## High-resolution demonstration requirement

The 160 by 64 CPU animation is only the first preview. The user requests
high-resolution images/video and measured CPU versus GPU render timing.
Target a native 1280 by 512 or larger logo still, with shared X_eTaL
antialiasing, before committing to a full revolution at that quality.
Use the same scene, resolution, samples, seed, interaction budget and
explicit precision on both devices; validate raw pixels first. Record
warm render wall time, GPU device intervals, cold setup, transfers and
image/video encoding separately. Show the CPU implementation and measured
speedup or slowdown; do not infer timings from the existing tiny scenes.
GPU source correctness/scaling, matched stills and timings now precede
further glass rendering and the longer animation. Retain TIFF masters.
