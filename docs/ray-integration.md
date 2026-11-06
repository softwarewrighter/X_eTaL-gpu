# Ray-tracing integration status

## Committed dependencies

XETAL_LIBRARIES_COMMIT pins X_eTaL-libraries 7bd8f0a, which includes
Vector3 (RL1, 202ed3e), Extrusion (RL2, f28e7ca), Scene (RL3, 6e58863)
and opaque RayTrace (RL4, 128e557). scripts/xetal-libraries.sh clones a committed
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

Eleven committed examples give their upstream outputs exactly:

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
The lowercase a is visually unclear in the delivered library outline
(user review); RL2a in xetal-libraries-ask.md requests a focused glyph
correction and visual acceptance checks. Existing previews retain that
known issue until a committed upstream fix is adopted.
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

## GPU eligibility evidence

The committed RayTrace diagnostic is not accelerated yet. Running it
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
inputs remain refused pending value validation. Known matrix cases and
empty identities agree with the pinned evaluator; tests include NaN
matching, exact failure messages and changing bound inputs. All/any and
whole-array matching agree exactly on XIR, RTX OpenCL and Mojo.

The full RayTrace audit still refuses all three actual renderer outputs
at a data-dependent guard; its only device output is the extra unrelated
scalar. No guard was removed. Runtime guard execution is the next local
compiler task, followed by an audit of the bounded primitive/light sweeps.

The next local work is runtime guard behavior. Preserve the library's finite,
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
