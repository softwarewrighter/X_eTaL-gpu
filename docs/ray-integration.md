# Ray-tracing integration status

## Committed dependencies

XETAL_LIBRARIES_COMMIT pins X_eTaL-libraries c968326, which includes
Vector3 (RL1, implementation 202ed3e) and opaque RayTrace (RL4,
implementation 128e557). scripts/xetal-libraries.sh clones a committed
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

Seven committed examples give their upstream outputs exactly:

| Example | Output lines |
| ------- | ------------ |
| Vector3 basics | 18 |
| Vector3 pivot demo | 3 |
| RayTrace camera | 11 |
| RayTrace intersections | 19 |
| RayTrace lighting | 12 |
| RayTrace pixels | 7 |
| RayTrace diagnostic demo | 36 |

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

## GPU eligibility evidence

The committed RayTrace diagnostic is not accelerated yet. Running it
through the source lowering returns:

```text
the program prints nothing the device could compute
```

An audit copy with one extra unrelated scalar output lets the lowering
report each renderer refusal instead of stopping on an empty device
program. All three renderer outputs fall back to the evaluator with:

```text
n_ot is outside the subset the device runs
```

The only XIR output in that audit is the unrelated scalar. This is not
GPU-rendering evidence. Merely invoking --device opencl:0 or counting
kernels would give a misleading success criterion here.

The next local work is generic lowering support for boolean/shape
validation and runtime guard behavior. Preserve the library's finite,
shape and domain checks; do not strip guards to make a renderer appear
eligible. Boolean operators and m_atch appear in the library validation
paths. Its primitive sweep is bounded p_ower; its light sweep is bounded
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
