# Ray-tracing integration status

## Committed dependencies

XETAL_LIBRARIES_COMMIT pins X_eTaL-libraries 8bd438d, which includes
Vector3 (RL1, 202ed3e), Extrusion (RL2, f28e7ca), Scene (RL3, 6e58863)
and opaque RayTrace (RL4, 128e557), with readable a/e glyphs from
RL2a (811cadd), Scene v2/sphere geometry (RL6, 2c08dfd) and bounded
Glass transport (RL5, 30277f5). scripts/xetal-libraries.sh clones a committed
repository into work/xetal-libraries and refuses to replace a dirty clone.
It never imports the sibling working tree. For an initial local clone,
set XETAL_LIBRARIES_SOURCE to ../X_eTaL-libraries; the source still comes
from Git commits, not staged or untracked files. The libraries use MIT.

The evaluator needs X_eTaL abdd371: the previous 27b3fd0 evaluator does
not recognize the s_qrt builtin used by the new libraries. This is an
adoption/version prerequisite, not a request for new language syntax or
a workaround in library code. Both Rust lockfiles follow the newer
committed language crates; the pinned xetal-x remains independent.

```sh
just xetal-libraries
just ray-check-cpu
```

The CPU check uses this repository's pinned evaluator, the committed
library sources and their own pinned outputs. Its library paths are
absolute so any evaluator subprocess resolves the same libraries.

## CPU evidence, 2026-11-06

Twenty committed examples give their upstream outputs exactly. The
original eleven remain unchanged:

| Example | Output lines |
| ------- | ------------ |
| Vector3 basics | 18 |
| Vector3 pivot demo | 3 |
| RayTrace camera | 11 |
| RayTrace intersections | 19 |
| RayTrace lighting | 12 |
| RayTrace pixels | 7 |
| RayTrace diagnostic demo | 36 |
| Extrusion glyphs | 25 |
| Scene arrays | 16 |
| Scene animation | 11 |
| Scene defaults | 5 |

Nine additional examples cover Scene solids/v2, RayTrace spheres/mixed
hits, and Glass basics/transport/paths/images/furnace. Each matches its
committed baseline exactly; no outputs were reblessed.

The diagnostic exercises a two-triangle, two-material scene with point
lighting, shadows, a partly reflective material and display conversion.
The checks are numerical/text comparisons against the committed library
baselines, not copied host implementations of ray tracing.

A separate 64 by 64 CPU preview was rendered by changing only the
committed diagnostic's image size and pixel range to 64 64 and 4096.
The pixels were computed by RayTrace in X_eTaL. Its last 4096 display-RGB
rows were quantized to RGB8 and encoded on the host as uncompressed TIFF.
The generated artifact is work/ray-integration/diagnostic-cpu.tif; libtiff
(tiffinfo -D) decodes it and reports 64 by 64, RGB, 8 bits per sample,
three samples per pixel and Compression=None. A PNG preview is alongside
it. Neither artifact is tracked, and this is a diagnostic tile rather
than the final logo/glass scene. A permanent image writer and reproducible
render command belong to the queued ray-image implementation.

## Visible CPU logo

```sh
just ray-render-cpu
just ray-render-cpu --width 256 --height 64 --output work/ray-integration/logo-cpu-256
just ray-render-cpu --angle 0.0 --output work/ray-integration/logo-zero-cpu
just ray-test-image
```

The render command requires Python 3 and libtiff's ppm2tiff tool.
It checks for the encoder before starting an expensive render.

examples/ray/logo.xtl declares the scene through the general Scene macro,
rotates only the vane group around its pivot, and renders with RayTrace.
It adapts the upstream MIT Scene logo demo without duplicating any
geometry or rendering algorithm in the host code. The fixed X is silver;
the _eTaL vane is copper. The camera looks toward the extruded letters.
Earlier files named logo-cpu-256 under work/ray-integration used the
old glyph outlines. The recorded gallery adopts RL2a at e7140be and uses
new output paths under work/ray-live/logo; its a/e glyphs are corrected.
The low-resolution preview has no antialiasing and is an initial scene,
not the final animation or glass composition.

The default 64 by 16 image took 40.02 seconds on this host's CPU evaluator.
The 256 by 64 still took 673.97 seconds (about 11 minutes), producing
logo-cpu-256.tif and logo-cpu-256.png. Both image sizes were visually
inspected; the larger TIFF also decoded successfully with libtiff.
The generated logo-cpu.tif has uncompressed RGB8 pixels; tiffinfo -D
successfully decoded it and reported Compression=None. logo-cpu.png is
a convenient compressed preview. Both are under work/ray-integration/.
The command saves its actual generated .xtl, raw RGB .out and a .json
record naming the CPU backend, dimensions, angle, pins and render time.

scripts/ray-render-cpu.py only runs the evaluator, checks RGB output,
quantizes display values to 8 bits and writes a PPM adapter file.
Installed libtiff ppm2tiff -c none encodes the TIFF; its permissive
license is provided by the libtiff package (LICENSE.md on this host).
This avoids carrying a general TIFF encoder here. The PNG preview uses
Python's standard library. TIFF pixels are never compressed. The host does not
trace rays or shade surfaces. Unit checks cover known TIFF tags/payload,
PNG rows/CRCs and rejection of invalid pixel counts/nonfinite/range data.
The gate also checks the eleven pinned library examples and lints the
local scene source. Output files are ignored build artifacts.

The first visible CPU render is available now. At the start of this work,
the planning estimate was 20-40 minutes for a reproducible CPU preview and
roughly 2-4 focused work sessions for the first verified GPU image. The
latter remains uncertain until validation guards and bounded sweeps lower
completely. No GPU timing or speedup is inferred from these CPU images.
Glass rendering additionally needs the pending RL5-RL7 library work.

## Current GPU evidence, 2026-11-06

This diagnostic GPU comparison used library pin 7bd8f0a and language
pin abdd371. The recorded CPU gallery adopts the later glyph correction
at e7140be; each recording retains its own pins and source archive.

The tiny two-triangle RayTrace diagnostic now lowers all three renderer
outputs without evaluator fallback. The pristine-source command below has now passed using the committed
4 by 4 diagnostic directly. An earlier audit used that diagnostic plus
an unrelated scalar; its scalar output was removed from generated XIR
before an independent execution check. The actual renderer outputs
are triangle IDs, linear RGB and display sRGB. XIR CPU execution matches
all 36 evaluator baseline lines exactly. RTX 5060 Ti OpenCL matches all
16 triangle IDs exactly and all color values within tolerance 1e-5.
There is no host rendering path when executing this XIR.

This is the first real GPU ray-tracing result, not a GPU logo or glass
render and not a measured speedup. Generated evidence is under
work/ray-integration/: guard-audit.xir/err, diagnostic-device.xir,
diagnostic-xir-cpu.out and diagnostic-opencl.out. The OpenCL pixels also
encode as diagnostic-opencl.tif (uncompressed RGB8) and a PNG preview;
tiffinfo -D decodes the 4 by 4 TIFF. Their small size is intentional.

```sh
just ray-check-device                 # pristine pinned source, CPU and OpenCL
just ray-check-device mojo            # optional full Mojo renderer comparison
```

The command passed on CPU and RTX OpenCL. It separately lowers the
pristine library diagnostic, requires
all three outputs and no fallback messages, and compares against its
committed evaluator output. Cold lowering and backend builds can be slow;
this is a separate integration command, not part of every gate run.

Ordered scalar validation is now an XIR operation. It executes even if
its result is unused. OpenCL and Mojo compute conditions on the GPU,
read one Bool back, and stop before queuing dependent work on failure.
Guard execution, constant guards, message round trips, repeated runs,
failure followed by valid rebinding, and numeric 0/1 boolean conversion
have CPU/OpenCL/Mojo tests. Numeric values other than 0/1 fail rather
than silently becoming true. General dynamic branches remain refused.

The diagnostic exposed f_irst shape bugs: removing a selected axis must
return a scalar for a vector, a row for a matrix, or a column for _2.
The lowering now preserves those shapes and the empty-axis prototype.
The bounded triangle sweep and bounded light recursion both lower in
this diagnostic. Strict changed-scene/ray-input renderer tests and
larger-image/backend validation remain in ray-source/ray-device.

The audit generated roughly 23,770 XIR lines and 1,585 guards. This is
substantial compilation and synchronization overhead for a tiny image;
large glyph scenes must not be assumed practical yet. Profile and improve
general lowering/validation scheduling while retaining semantics before
claiming a useful GPU speedup or scaling to the full logo.

## Earlier GPU eligibility audits

Before runtime guards landed, the committed diagnostic was not accelerated. Running it
through the source lowering returns:

```text
the program prints nothing the device could compute
```

The initial audit copy with one extra unrelated scalar output lets the lowering
report each renderer refusal instead of stopping on an empty device
program. Initially all three renderer outputs fell back to the evaluator with:

```text
n_ot is outside the subset the device runs
```

The only XIR output in that audit is the unrelated scalar. This is not
GPU-rendering evidence. Merely invoking --device opencl:0 or counting
kernels would give a misleading success criterion here.

### Boolean lowering progress, 2026-11-06

The compiler now translates typed Bool n_ot, & and | using existing XIR
maps. Numeric 0/1 inputs accepted by the evaluator remain outside this
new typed-Bool subset: accepting general numbers would require runtime
validation of their values. Strict function tests cover all truth-table pairs, scalar extension,
changed bound inputs and rejection of incorrect types/shapes. Their truth
tables agree with the pinned evaluator. A composed comparison/mask
expression with two different input batches agrees exactly between XIR,
OpenCL and Mojo on this host's RTX 5060 Ti. This adds no language feature,
rendering algorithm or library API.

Re-running the diagnostic audit now moves the refusal to:

```text
a guard on a value the device computes is outside the subset (the condition must be known now)
```

All three renderer outputs still fall back, and the unrelated scalar
remains the only device output. The boolean improvement is not evidence
of a GPU-rendered image. No validation checks were removed.

RL2 extrusion/glyphs (f28e7ca) and RL3 scene macros (6e58863) are now
adopted at 7bd8f0a in the visible-preview step. The earlier boolean
compiler step retained c968326 until this explicit adoption boundary.

### Shape and array-validation operators, 2026-11-06

Shape comparisons and boolean expressions over metadata now stay static.
A bound array with the correct dimensions passes a shape-only guard during
lowering; incorrect dimensions retain the library's panic message.
This resolves checks of shapes, not checks of the data in those arrays.

m_atch now compares complete equal-shaped arrays and returns one Bool;
different shapes return false. Equal-type numeric arrays, Bool/Int
normalization and static strings are supported. Matching across Float
and Int remains refused explicitly until conversion semantics are covered.
This avoids silently changing large-Int equality by casting to Float.

Bool '& r_/ and '| r_/ now lower using existing numeric reductions and
comparisons. They keep the default first axis, explicit axes and all-axis
semantics. Empty axes use true for all and false for any. Numeric 0/1
inputs were refused at that point; runtime validation now supports them. Known matrix cases and
empty identities agree with the pinned evaluator; tests include NaN
matching, exact failure messages and changing bound inputs. All/any and
whole-array matching agree exactly on XIR, RTX OpenCL and Mojo.

That operator-only RayTrace audit still refused all three renderer outputs
at a data-dependent guard; its only device output is the extra unrelated
scalar. No guard was removed. Runtime guard execution and the sweep audit were the next compiler work;
the current result above records their diagnostic coverage.

At that earlier point, the next local work was runtime guard behavior. Preserve the library's finite,
shape and domain checks; do not strip guards to make a renderer appear
eligible. Shape-only guards now resolve statically, while data-dependent
failures must become runtime checks on both backends. Its primitive sweep
is bounded p_ower; its light sweep is bounded
tail recursion. Their complete eligibility must be tested after the
first refusal is addressed, not assumed from this initial audit.

Strict integration acceptance requires device outputs for the actual
rendered arrays, no renderer fallback reasons, evaluator comparisons,
and changed-bound-input checks proving that new rays/scene data affect
the device result. Keep IDs exact, report Float tolerances and test invalid
inputs retain their specified failures. This is GPU work in this repo;
no new library API or core-language feature is currently requested.

## Work order

The saga now puts language compatibility, committed-library adoption,
ray-source, ray-device and ray-image before the remaining Mojo/ML work.
RL2 extrusion and RL3 scene macros are not prerequisites for the opaque
triangle diagnostic or local lowering work. The existing later logo and
glass milestones retain their own library dependencies. Source/eligibility
work can proceed without waiting for those macros.


## Recorded logo gallery

The new Ray tracing section on the GPU project's landing page links to
[the gallery](https://softwarewrighter.github.io/X_eTaL-gpu/ray-tracing/).
The logo entry provides a recorded WebM, individual PNG frames, render
settings and an uncompressed ZIP containing TIFF masters and each frame's
actual X_eTaL source. Glass and benchmark entries are marked planned;
their thumbnail illustrations are not rendered results.

```sh
just ray-render-cpu --width 96 --height 40 --angle 0 --output work/ray-live/preview
just ray-test-scene
just ray-animate-cpu
just ray-render-cpu --width 320 --height 128 --angle 0 --output work/ray-live/logo/poster
just ray-stage-media
just ray-test-media
just pages
just ray-test-page
```

The animation uses 24 frames at 160 by 64 and 6 fps, with angles
2*pi*frame/24 and no duplicate endpoint. Frames are rendered independently
by the CPU evaluator; four jobs are allowed, with pin/source-aware resume
checks. No browser-side ray tracing or GPU logo timing is claimed. A
320 by 128 still supplies the thumbnail. This first preview uses direct
diffuse light and hard shadows via RayTrace s_hade, without reflections.
Its WebM is a lossy web derivative; its RGB8 TIFF masters have no pixel
compression and their ZIP entries use storage without compression.

The X is centered on the vertical axis through (0,0.5,0). Only the vane
group rotates: the original _eTaL mesh, a thin shaft and a triangular
arrowhead, all pointing outward along local +x. X, post and floor remain
fixed. General Extrusion and Scene functions build and transform all
322 triangles in X_eTaL. The camera remains fixed. The geometry test
checks the X/post/floor, pivot distances, quarter-turn direction and
full-turn return. The library's corrected lowercase contours are used.

Selected media are curated under site-assets/ray/logo so building the
site never reruns a long animation. scripts/test-ray-media.py validates
pins/source hashes, frame angles, PNG CRCs, exact TIFF/PNG master pixels,
uncompressed archive entries and the decoded WebM frame count. A local
Chromium check exercises playback, frame selection and mobile layout.
Actual publication is verified with just check-live after deploying;
prepared files or a branch push alone do not establish live availability.

## Sphere/crystal composition

The library pin now includes RL5/RL6. The published logo recording keeps
its original e7140be metadata and frozen scene. RL7 remains uncommitted
upstream and is not used by this composition.

examples/ray/glass.xtl uses the Scene v2 macro for three analytic
spheres, three closed octahedra, a dark reflective floor and two bright
panels. The fixed camera is (5,2.8,7), looking at (0,0.8,0). The current
materials are opaque/mirror: this diagnostic checks placement, not
transparent glass. Shared Glass transport supplies bounded sampled mirror
paths and emission; no host ray tracer or new language feature is added.

```sh
just ray-test-glass-scene
just ray-glass-composition
just ray-render-cpu --scene glass --width 32 --height 32 --output work/ray-glass/composition-32
```

The scene test checks twelve invariants, including closed outward shells,
floor contacts, mixed global/object/material IDs, and shared dielectric
preflight with six separate, non-overlapping volumes. The diagnostic uses
one sample per pixel, seed 123, interaction cap four, a pinhole camera,
point lights and opaque point-light shadows. Output includes raw display
RGB, generated source, PNG, uncompressed RGB8 TIFF and a settings manifest.
Exposure is one, with clamping and the shared sRGB transfer before RGB8
quantization. No depth of field, caustics, dispersion or bloom is claimed.

Next, integrate transparent materials and validate entry/exit paths before
publishing the glass gallery entry. This composition has not established
GPU eligibility or a CPU/GPU speedup; those require the queued source and
final paired-render checks.

A one-pixel eligibility probe of the same composition currently refuses
at u_nique in shared preflight. Its host fallback then attempts s_hape
on the tuple scene bundle and reports expected array, found tuple.
The generated probe and diagnostic are work/ray-glass/eligibility.xtl
and eligibility.log. This is local lowering/fallback work for ray-source,
not an upstream language/library ask; validation is kept intact.

The user screenshot ~/pixelated.png shows the initial 160 by 64 logo
recording enlarged to roughly 1300 pixels wide. The dominant quality
limit is native render resolution, with no supersampling, rather than
TIFF compression or a browser fault. The 320 by 128 poster improves the
still only. A presentation-quality recording needs newly rendered frames
at higher resolution and shared antialiasing; re-encoding/upscaling the
same pixels does not recover missing detail. Validate an intended-size
still before committing to a longer high-resolution sequence.

Composition diagnostics completed at 32 by 32 in 39.55 seconds and 64 by
64 in 155.63 seconds on the CPU evaluator. These are single observed
render-call wall times during other work, not a paired benchmark. The
64 by 64 TIFF decodes with Compression=None. Bright panels are visible
in the framing and need adjustment for a darker presentation image.
After this checkpoint the user prioritizes high-resolution logo output
and paired CPU/GPU timings; transparent glass integration follows.

## Bound-input renderer checks and lowering cost

The lowerer now checks each new operation incrementally and reuses
identical immutable expressions. The pristine diagnostic remains exact
on XIR CPU and within 1e-5 on RTX OpenCL, with exact IDs. Against the
saved old audit, its IR falls from 23,770 to 4,541 lines, and guards from
1,585 to 310. These are structural counts, not render speedup timings.

components/cli/tests/ray_inputs.rs lowers the actual RayTrace nearest
and render functions with four bound arrays and no evaluator fallback.
It changes geometry, origins, directions and materials, compares raw
GPU colors/IDs to the interpreter, checks zero directions, invalid
materials and nonfinite geometry, and verifies valid rebinding recovers
after each mutation or failure. Prepared OpenCL execution is tested on
the installed device. Full renderer Mojo remains unverified.

The one-pixel 322-triangle logo still produces 258,581 IR lines and
16,817 guards. Geometry preparation explicitly falls back for r_ev and
g_rade. This is not a high-resolution GPU result. RL10 requests bounded
vectorized primitive queries in the general library; local allocation,
fusion and guard-readback costs remain GPU work. See
[performance choices](ray-performance.md) for batching, residency,
parallel-frame tradeoffs and matched CPU/GPU measurement requirements.
