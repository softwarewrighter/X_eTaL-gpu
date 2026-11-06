# Asks for X_eTaL libraries

## Ownership and request policy

Reusable algorithms belong in the base/library repositories. Prefer
existing libraries, then ordinary library functions and macro libraries;
new notation should normally be a macro library rather than language
syntax. This file records new or changed general library requests from
X_eTaL-gpu. GPU lowering, scheduling, kernels, device management and
execution integration belong here, building on the existing components.

Reserve [xetal-asks.md](xetal-asks.md) mainly for demonstrated language
bugs. A core feature request needs evidence that library and macro
solutions cannot express the requirement. Existing historical asks are
not automatically new requirements for ray tracing. This repository
records requests; it does not modify sibling repositories.

Each concrete ask records status, intended owner, existing APIs checked,
required behavior, CPU example, tests, why reuse is insufficient, macro
versus function rationale, and any temporary implementation and removal
condition. Record an upstream commit when landed and pin it deliberately.

## Current local library audit

| Code | Finding | Ownership |
| ---- | ------- | --------- |
| `libs/Accel/src/Accel.xtl` | Pure array functions: sums, dot/matrix products, masks, row reductions, transposes, embeddings, windows, convolution, argmax/top-k, dense layers and softmax. No device selection or OpenCL calls; these are general algorithms despite the name. | Existing tested GPU subset/reference fixture here. Reuse general library APIs where available; review consolidation with X_eTaL-libraries and X_eTaL-ML before moving exports. Preserve compatibility and baselines; no migration performed by this audit. |
| `libs/Accel/demos/` and `tests/` | Source programs and evaluator baselines that exercise the lowering and device implementations. | GPU integration coverage belongs here, even when the functions come from a general library. |
| `extensions/gpu/lib/Gpu.xtl` | Device/session facade: load, bind, run, output, schedules, kernel/explain and execution helpers. | GPU-specific; stays here. Reusable math should not accumulate in the facade. |
| `extensions/gpu/lib/Gpu.xtlm` | `gp:a_ccel<` generates calls to this GPU runner; potentially reusable runner pattern, currently coupled to `gp:` and its session interface. | Current macro stays here. Generalize elsewhere only if an actual shared runner contract is established. |
| Proposed RayTrace functions and scene macros | Camera/vector geometry, intersections, materials, lighting and scene notation have CPU as well as GPU meaning. | General library repository; this repo owns the acceleration support and demo harness. |
| Image encoding | TIFF encoding is general host I/O rather than GPU semantics. | Reuse suitable non-GPL host tooling/library; keep only the pixel-output adapter and demo commands here. No rendering in the encoder. |

## Concrete requests, 2026-11-06

The user approved the combined scene DSL and geometry macro design.
RL1 through RL4 below are requested here for X_eTaL-libraries owner
triage; they are not claimed to be acknowledged, scheduled or implemented
upstream. Package and export names are proposals, not four mandatory new
libraries. Reuse existing packages where the semantics fit. No language
feature or native rendering extension is requested.

Implementations and original glyph data should follow the owning repo's
MIT license, with provenance for any reused material. Do not introduce
GPL/AGPL implementation dependencies or copy POV-Ray code or bundled
assets; POV-Ray remains a capabilities reference only.

### Audit evidence

Inspected X_eTaL-libraries checkout
`27478ec9a20906871b3e3f3e4c241d3448d113d1` (an audit reference,
not a new dependency pin):

- `libs/Geometry/src/Geometry.xtl`: points are 2-row matrices; distance,
  area, centroid, 2D rotation/moves, convex hull and picture output.
  Its private cross product is a 2D orientation helper, not a public
  batched 3D vector operation. Preserve these existing APIs.
- `libs/Matrix/src/Matrix.xtl`: identity and matrix/vector products
  already exist; reuse these instead of another matrix package.
- `libs/Parsing/`: total parsers for notation macros, with refusal
  diagnostics. Reuse its split/name/number parsing as appropriate.
- `libs/Graphs/src/Graphs.xtlm`: a statement macro validates names,
  assigns numeric IDs and emits ordinary bindings and array literals.
  This is the model for scene names and generated bindings.
- `libs/Polynomials/src/Polynomials.xtlm`: a notation macro computes
  constants during expansion and emits their X_eTaL source.

No scene, extrusion or ray-tracing entry point was identified in the
inspected library sources. The owner should check newer upstream work
before implementing. Existing APIs above are reuse candidates, not asks
for reimplementation. This repo will pin a committed library revision
with explicit dependencies and rerun baselines before adoption.

### Request summary and deadlines

Deadlines are dependency milestones, not calendar promises. Ask status
is open/requested here for every row; an upstream commit is pending.

| ID | Request | Owner | Needed by | Dependencies |
| -- | ------- | ----- | --------- | ------------ |
| RL1 | General batched 3D vector and rigid-transform functions | X_eTaL-libraries, compatible Geometry extension or a small 3D companion | Before ray-source can finish and ray-device can validate the renderer | Existing arithmetic, reductions, trig and Matrix products |
| RL2 | Polygon extrusion functions and extrusion/letter macro | X_eTaL-libraries, general mesh/geometry library | Before ray-logo; independent of the diagnostic still image | RL1 conventions, Parsing; can develop alongside RL3 |
| RL3 | Scene DSL macro with ordinary scene/transform functions | X_eTaL-libraries, proposed Scene.xtl/Scene.xtlm | Before assembling the logo scene in ray-logo; required for the delivered scene workflow | RL1 and the agreed data contract; references RL2 assets |
| RL4 | Ray-tracing functions in ordinary X_eTaL | X_eTaL-libraries, proposed RayTrace.xtl | Primary hits/lighting before ray-source ends; shadows/reflection before ray-logo; frame correctness before ray-bench | RL1; can start with hand-authored numeric scene data before RL2/RL3 land |

There is no upstream wait needed to continue local GPU capability checks,
small explicitly temporary fixtures, the image encoder or benchmark
harness. Final reusable algorithms and macro demos require committed,
pinned library implementations. Missing GPU lowering/backend support for
existing operations is implemented here as local steps, not sent to the
language or library owner as a new primitive request.

### Shared numeric data contract

Agree this contract before RL2/RL3 independently implement their outputs.
The following is the proposed v1 contract, with any owner changes recorded
in this ledger before adoption:

| Array | Shape/type | Meaning |
| ----- | ---------- | ------- |
| triangles | T by 3 by 3 Float | Triangle, vertex, xyz coordinate; stable declaration order |
| materialids | T Int | One-based material row for each triangle |
| groupids | T Int | One-based transform group for each triangle |
| materials | M by 4 Float | Linear red, green, blue, reflection weight; each in 0..1 |
| transforms | G by 4 by 4 Float | Row-major rigid transforms using column-vector mathematics |
| pivots | G by 3 Float | Local group pivots, in scene units |
| camera | 10 Float | Eye xyz, target xyz, up xyz, vertical field of view in radians |
| lights | L by 6 Float | Point-light position xyz and linear RGB intensity |

Use right-handed xyz coordinates with y up and radians throughout. A
positive pi/2 rotation about y sends (1,0,0) to (0,0,-1). Batched points
are N by 3 rows; multiply homogeneous rows by the transpose of the
column-vector transform. Mesh normals are computed from transformed
triangle edges, with a documented winding rule. V1 frame transforms are
rigid; general affine transforms and inverse-normal matrices can follow
only if a real use case needs them.

The macro emits separate numeric arrays, not a GPU-side record, nested
scene graph, text parser or tuple requirement. Names and labels are
compile-time metadata. IDs stay Int; do not encode them in f32 colors or
hit distances. Default identity group is row 1. Keep precision, work-group
sizes, backend choice and device schedules outside scene semantics.
Resolution, ray clipping distances, sampling and bounce count are ordinary
render parameters, supplied separately with documented shapes.

### RL1: 3D math and transforms

Status: open/requested here. Intended owner: X_eTaL-libraries.

Request public pure functions for batched dot/cross, length, safe
normalization, translation and rotation about a pivot. Suggested exports
include d_ot, c_ross, n_ormalize, r_otateY and t_ransform; final names
follow the owning package. Dot maps two matching N by 3 arrays to N
Floats; cross maps them to N by 3. Specify scalar/vector broadcasting
rather than accepting ambiguous shapes. Normalizing zero returns zero;
invalid camera directions are rejected by scene validation before render.

Reuse Matrix products and existing primitives. Keep Geometry's 2D point
layout unchanged; provide an explicit adapter if shared code uses it.
These are ordinary library functions because their inputs, including the
frame angle, can change at run time. Macro libraries may call the same
functions when folding static geometry.

CPU acceptance examples: cross((1,0,0),(0,1,0))=(0,0,1), dot=0;
normalize((3,0,4))=(0.6,0,0.8); normalize((0,0,0))=(0,0,0).
A pi/2 y rotation of (1,0,0) is approximately (0,0,-1); rotating about a
nonzero pivot leaves that pivot fixed. Test batches, shape refusals,
identity/composed transforms and finite results. Publish types, reg-rs
baselines and doc examples. Use array compositions suitable for lowering;
GPU support and tolerances are verified here, not promised upstream.

### RL2: extrusion functions and geometry macro

Status: open/requested here. Intended owner: X_eTaL-libraries.

Request ordinary triangulation/extrusion helpers and a macro, tentatively
ex:e_xtrude<, that accepts a named closed 2D contour and positive depth,
then emits ordinary source constructing its triangle array. Contours use
the existing Geometry layout, 2 by N Float (x row, y row), before
conversion to triangle/vertex/xyz order. The macro
justification is concise geometry notation, deterministic constant
construction and early diagnostics. The underlying functions remain
callable without the macro; do not duplicate the geometry algorithm.

V1 supports simple convex and concave contours with no holes, implicit
closure and a documented winding rule. Extrude symmetrically about z=0
(depth d gives z=-d/2 and +d/2), with front/back caps and outward side
faces. Reject self-intersections, repeated/zero-length edges, collinear
zero-area contours, invalid numbers and nonpositive depth. Document the
geometry tolerance; do not silently turn a malformed outline into its
convex hull. Hole contours, bevels and Boolean solids are later features.

For the logo, request original block/stroke outlines for X, underscore,
e, T, a and L, with deterministic advances/spacing and depth. A glyph may
be composed of closed stroke solids; arbitrary fonts and polygon holes
are not prerequisites. An unsupported character must be named in a macro
refusal. Any bundled outlines must have documented compatible provenance;
do not copy POV-Ray or other renderer assets.

CPU acceptance: extruding a 2 by 1 rectangle to depth 0.5 gives 12
triangles, bounds x=0..2, y=0..1, z=-0.25..0.25 and volume 1. Test normals,
caps, edge pairing, a concave X contour and each supported glyph. Pin
expansion output, compare it with direct function construction and test
all refusal cases. No renderer/device calls appear in the expansion.

### RL3: scene macro and transform groups

Status: open/requested here. Intended owner: X_eTaL-libraries.

Request a small Scene.xtlm macro, tentatively sc:s_cene<, plus ordinary
Scene.xtl constructors and group-transform functions. Prefer one scene
macro and the reusable RL2 geometry macro over a macro for every property.
The scene macro validates named references, assigns stable numeric IDs
and emits the shared arrays. This removes repetitive shape/index assembly
without adding language syntax or coupling the scene to a backend.

Proposed call shape: a statement macro with a binding prefix on the left
and scene text on the right, following Graphs. For example,
"logo" sc:s_cene< SCENE_TEXT generates logotriangles, logomaterialids,
logogroupids, logomaterials, logotransforms, logopivots, logocamera and
logolights. Reserve and document that generated namespace; reject duplicate
labels and do not silently shadow another scene's generated bindings.
The ordinary compiler still checks collisions outside the macro input.

Proposed scene text below is a grammar proposal, not installed syntax:

```text
camera eye 6 4 8 target 0 1 0 up 0 1 0 fov 0.7
light key position 5 8 6 color 1 1 1
material silver color .8 .8 .85 reflection .3
material copper color .8 .3 .1 reflection .2
object x extrude X depth .3 material silver
object tail extrude _eTaL depth .2 material copper
group fixed objects x
group vane objects tail pivot 0 0 0
```

Keep the initial grammar small: numeric literals, named assets/materials,
one camera, point lights and flat groups. Resolve references after parsing,
so declaration order does not cause accidental missing-name errors.
Object and material IDs follow declaration order. Group row 1 is the
reserved identity; declared groups follow from row 2 in declaration
order. An object
belongs to exactly one group, using the identity group if none is given.
Reject unknown/trailing tokens, duplicate names, unresolved references,
invalid camera basis/fov, invalid colors/weights, and multiple group
membership. No arbitrary embedded expression evaluator or nested graph
is needed in v1.

Reuse Parsing for total parsing and []R_EJECT for macro refusals; name
the declaration and bad token. Prefer literal-array expansion requiring
no caller library alias or imports. If runtime helper calls are emitted,
document the required import alias and test it; macros must not inject
imports or assume they can inspect caller types/shapes.

Animation is a Scene.xtl function over the static arrays and a run-time
angle: T(pivot) * Ry(angle) * T(-pivot) for vane, identity for fixed.
No frame loop or rendered pixels are evaluated during macro expansion.
The same function must work when called directly on CPU and lowered here.

Acceptance: pin the valid scene's expansion; compare all emitted arrays
with ordinary constructors; check stable IDs and two scenes in one file.
At angles 0, pi/2 and 2*pi, X vertices stay unchanged, vane vertices keep
their pivot distances and 2*pi returns to the start within tolerance.
Test every refusal class, comments/whitespace and complete input consumption.

### RL4: ray tracing as reusable X_eTaL functions

Status: open/requested here. Intended owner: X_eTaL-libraries.

Request camera-ray generation, triangle intersections, nearest-hit
selection, ambient/diffuse shading, hard shadow visibility, and one bounded
reflection bounce as ordinary X_eTaL functions. These compute values at
run time, so scene macros call no native rendering engine and do not
replace the algorithm. No renderer-specific language primitive is needed.

Use the shared scene arrays and explicit render parameters. Start with
pinhole rays and Moller-Trumbore intersections, two-sided triangle hits,
positive finite near/far clipping distances, deterministic first-triangle
selection on an exact distance tie, and a background color. A miss needs
an explicit validity mask and zero ID; all output numeric fields remain
finite. Keep masks, Int IDs and Float distances in documented separate
array outputs/accessors without requiring new tuple lowering. Safe
masked division must substitute a finite denominator before division.

Define edge inclusion, parallel/degenerate thresholds, ray-origin offset,
shadow ray distance limits, material mixing and reflection termination.
Document the linear RGB output and the shared display conversion used
before TIFF quantization. Keep pixel/triangle batches bounded; avoid a
mandatory full-resolution pixels-times-triangles intermediate. Make
chunking preserve the exact tie rule. No BVH, recursion, global illumination,
texture system or arbitrary bounce depth is a prerequisite.

CPU acceptance: a ray from (0.25,0.25,1) along (0,0,-1) hits triangle
((0,0,0),(1,0,0),(0,1,0)) at t=1 with barycentric u=v=0.25.
Test misses, parallel rays, edges, degeneracy, equal-distance ties,
occlusion, reflection and chunked versus unchunked small scenes. Check
deterministic raw pixel arrays before image-file encoding. The owning
repo supplies documented exports, pinned types and evaluator baselines;
this repo proves the same source lowers and agrees on GPU and CPU, with
stated tolerances and measured timing. Upstream need not own GPU tests.

## Delivery and adoption

Implement in dependency order RL1, the RL4 diagnostic renderer, then RL2
and RL3 for the logo; RL2/RL3 may proceed in parallel with RL4. Commit
ordinary functions, macro expansions, failure tests, documentation and
export types in the owning library repo. Report committed references and
any required base-language revision; do not depend on an uncommitted
checkout or edit the language pin mid-step.

When delivered, record the upstream commit and status here. Add a deliberate
library pin/import recipe and dependency license review in X_eTaL-gpu,
rerun evaluator/type/doc/lint baselines, then run lowered CPU/GPU equivalence.
Any temporary local fixture must name its RL ask, path and removal point;
none exists yet. Replace it with the pinned library before closing the
relevant renderer/logo milestone. No duplicate permanent general library
is introduced here, and no ask is sent automatically to another service.
