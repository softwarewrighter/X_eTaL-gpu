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

### Adoption snapshot, 2026-11-06

RL1 is completed upstream in 202ed3e (handoff 4cd8725), and RL4 in
128e557 (handoff c968326). XETAL_LIBRARIES_COMMIT pins c968326, with
CPU compatibility and GPU eligibility evidence in
[ray-integration.md](ray-integration.md). RL2 is in progress and RL3
queued in the last inspected peer saga. The original requested statuses
below are historical; upstream completion does not imply GPU eligibility.
The concrete n_ot/validation lowering gap is local GPU work, not a new
library or language ask. RL5-RL7 are accepted in the peer working plan
but remain pending; RL8/RL9 remain deferred.

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

## Glass-scene follow-on requests, 2026-11-06

The user requested the reference-inspired glass render described in
[glass-scene-plan.md](glass-scene-plan.md). RL5-RL7 are requested here for
owner triage and are required for that milestone. RL8/RL9 are deferred
refinements, not prerequisites or approved extra packages. All belong in
X_eTaL-libraries as compatible additions to RayTrace/Scene and existing
math, geometry or image libraries where appropriate. No core-language
addition or native renderer is requested. The MIT/provenance and CPU/GPU
ownership rules above apply.

Progress snapshot: the peer working plan accepts RL1-RL4 and the v1
contract, and Vector3 is staged/in progress but not yet committed in the
inspected checkout. Do not consume it as a pin until it lands. RL5-RL9
are new local requests; upstream acknowledgment is not claimed. Existing
RL1-RL4 delivery remains useful and must not wait for these follow-ons.

### Follow-on summary and delivery points

| ID | Library change | Needed before | Status |
| -- | -------------- | ------------- | ------ |
| RL5 | Dielectric materials, refraction/Fresnel, medium state, absorption and bounded multiple interactions | glass-transport can finish | Requested here; owner triage pending |
| RL6 | Analytic spheres, closed crystal helpers, mixed nearest-hit IDs and v2 scene notation | glass-composition can finish; closed-glass checks before glass-transport | Requested here; owner triage pending |
| RL7 | Deterministic pixel/lens/emitter sampling, thin-lens camera and emissive/area lights | glass-camera and final glass-render | Requested here; owner triage pending |
| RL8 | Optional floor caustics via bounded photon transport and radiance estimation | Only a separately approved caustic refinement | Deferred proposal; not a first-render blocker |
| RL9 | Optional RGB dispersion and linear-image bloom | Only an optional finishing pass | Deferred proposal; not a first-render blocker |

### Compatible v2 schema proposal

Keep v1 materials M by 4, camera length 10, triangle arrays and point
lights valid, with their original meanings. A v2 constructor/macro emits
an explicit version and additional separate arrays; a v1 adapter provides
opaque defaults. Do not infer the version by silently reinterpreting an
old column. The library owner should finalize and test this contract
before independent Scene and RayTrace changes land.

| Added array | Proposed shape/type | Meaning |
| ----------- | ------------------- | ------- |
| sceneversion | scalar Int | 2 for the extended contract |
| materialkind | M Int | Opaque or dielectric tag, declared constants |
| optics | M by 5 Float | Transmission blend weight, positive IOR, nonnegative absorption coefficients RGB per scene unit |
| emission | M by 3 Float | Nonnegative linear emitted radiance, may exceed 1 |
| spheres | S by 4 Float | Center xyz and positive radius |
| spherematerialids, spheregroupids | S Int each | Material and rigid transform group IDs |
| triangleobjectids, sphereobjectids | T Int and S Int | Closed-object identity for entry/exit; multiple faces of one crystal share an object ID |
| triangleids, sphereids | T Int and S Int | Unique global primitive IDs in declaration order for stable mixed-geometry ties |
| lens | 2 Float | Aperture radius and positive focus distance; zero radius is pinhole |
| emitters | E by 3 by 3 Float | Rectangle origin, span vector u, span vector v |
| emitterradiance | E by 3 Float | Emitted linear RGB radiance, with orientation and visibility defined |

Opaque v1 defaults: no transmission, no absorption, no emission, pinhole
camera and no analytic spheres or area emitters. Existing opaque reflection
weights retain their meaning. For dielectrics, document an energy-consistent
blend of the opaque and dielectric lobes; the dielectric Fresnel reflected
and transmitted weights sum to one for a lossless interface. Do not add
an independent full-strength reflection on top of full transmission.

IDs remain Int and zero remains the miss/no-medium sentinel. Active-ray
medium IDs, hit kind/local index, throughput, direction, origin and sample
identity are runtime arrays, not encoded inside colors. V1 opaque faces
need no closed-volume identity; v2 dielectric faces require one. All APIs
state shapes, units, defaults and malformed-input diagnostics. Geometry
kind dispatch is ordinary X_eTaL code, not a new language primitive.

### RL5: glass transport and material functions

Extend the ordinary RL4 renderer with reflection and Snell refraction,
Fresnel interface weights, total internal reflection and distance-based
Beer-Lambert absorption. Start with air (IOR 1) and non-overlapping,
non-nested closed glass objects. Track entry/exit using geometric and
oriented normals plus object identity. Refuse unsupported nested or
interpenetrating glass during shared scene preflight; do not guess the
medium from material ID when several objects share one material.

Use finite masked denominators and clamp roundoff only within documented
numeric tolerances. At total internal reflection, return a reflection
path and an explicit invalid-transmission mask without sqrt of a negative
value. Define offsets, near/far clipping and interactions beginning inside
a supported medium. Document transparent shadow treatment: an optional
straight-segment attenuation approximation must be labeled and must not
be presented as refracted caustic illumination.

Multiple interactions must use bounded arrays and a fixed cap, not
unbounded recursion or exponential reflection/transmission splitting.
Prefer deterministic sample-indexed selection between weighted lobes for
production, with the estimator and throughput compensation documented.
A small deterministic branch-both oracle is acceptable for analytical
tests. State the radiance/importance convention and refractive IOR factors
in throughput; sampling probabilities alone are not a radiance model.
Preserve the v1 one-reflection entry point and its baselines.

Acceptance: normal-incidence air/glass IOR 1.5 reflectance is 0.04;
normal transmission preserves direction; Snell transmission for incidence
pi/6 has sin(theta_t)=1/3. An internal glass-to-air ray at pi/3 totally
reflects. Test finite results, Fresnel bounds/energy, entry/exit through a
sphere and slab, correct medium reset, absorption exp(-sigma*d), offsets,
clipping, interaction caps and fixed-seed repeatability. Compare tiny
images at increasing caps and record truncation effects. The GPU repo
handles lowering and execution tests for these existing array operations.

### RL6: geometry and scene macro extensions

Add ordinary analytic sphere intersection and normal functions, plus
closed convex polyhedron constructors, initially an octahedron and box.
Reuse RL2 mesh validation/extrusion and RL1 transforms where suitable;
closed crystals stay triangles. Analytic spheres avoid high triangle
counts and flat-normal artifacts without a renderer-specific primitive.
Rigid group transforms preserve sphere radius; bake static positive
uniform scale into the radius or reject unsupported transforms.

Extend the RL3 macro/constructors for sphere, closed-polyhedron and
v2 dielectric material declarations. Support a finite floor rectangle
using existing triangles. Assign object IDs separately from material IDs
and global primitive IDs separately from per-kind positions. The nearest
hit across spheres and triangles uses distance then smallest global ID,
independent of batches/traversal order. Retain v1 syntax and expansion
behavior with explicit v2 opt-in.

Validate finite/positive radii, mesh closure, winding and degenerate faces.
Macro validation can fold static geometry, but ordinary constructors must
perform equivalent checks for dynamically supplied data. Do not introduce
an arbitrary geometry scripting language or a font/CSG dependency.

Acceptance: unit sphere at origin, ray (0,0,3) along (0,0,-1), nearest
t=2 and normal (0,0,1); ray from its center exits at t=1; test tangent,
miss, unnormalized directions with documented t units, clipping and
transforms. Test mixed sphere/triangle ties and chunk equivalence. Closed
polyhedra have consistent outward normals and edge pairing. Pin macro
expansions and compare with ordinary constructors; v1 goldens stay valid.

### RL7: camera, emitter and deterministic sampling support

Add sample generation indexed by pixel, sample number, interaction,
dimension and seed, with a documented integer algorithm. The sequence
must be independent of thread order, chunk size and device scheduling.
Reuse any existing suitable sampling functions instead of a second RNG.
Expose arrays so CPU/GPU can receive identical sample data; any generator
run during rendering remains X_eTaL, not host rendering code.

Extend the pinhole camera with a thin lens: focus distance along the
camera forward axis, aperture radius in scene units, and uniform-disk lens
samples. Focus is a shared render parameter. Add finite rectangular area
emitters and visible emissive geometry, with sidedness, radiance units,
sampling density and visibility/shadow weights stated. Reuse scene material
emission for reflection panels. No bloom is required for a bright panel
to appear in a glass reflection.

Acceptance: aperture radius zero reproduces the pinhole rays, invalid
radius/focus refuses, rays for one film sample meet its chosen focal-plane
point, and a known nearer/farther object blurs as aperture grows. Integer
sample sequences are exact; Float conversions use documented precision.
Chunked samples match unchunked samples. Refuse zero-area/invalid emitters;
check constant-radiance sampling weights and repeatability, soft-shadow
convergence and emissive reflection through tiny scenes. Pin CPU outputs,
export types, macro expansion and new refusals; GPU comparisons live here.

### RL8: deferred caustic library addition

If the initial image needs concentrated illumination beneath glass,
request a separate light-to-surface photon pass and floor radiance
estimator, using the same RL5 optics and scene. Start with a finite planar
receiver and a bounded photon budget; keep photon positions, power and
IDs as arrays. Both photon tracing and camera shading remain X_eTaL.
Do not treat a texture decal or painted highlight as a physical caustic.

Specify estimator radius/kernel, normalization, sampling, fixed seeds and
memory bounds. Use deterministic chunked accumulation for the small CPU
oracle; GPU reduction order may differ only within recorded tolerances.
A spatial index is a measured follow-on optimization, not a core-language
ask or an implicit prerequisite for the first image. Test a glass slab,
symmetric focusing case, emitted/deposited energy accounting, chunk
agreement and convergence as photon count grows. Queue implementation only
after reviewing the first glass render and its cost.

### RL9: deferred dispersion and image finishing

Optional RGB-channel IOR parameters can create approximate rainbow edges;
label this RGB dispersion, not full spectral transport. Preserve a single
IOR mode and account for the extra path cost. Verify identical channel
IORs reproduce the non-dispersive result, all paths remain finite, and
CPU/GPU use the same channel/sample mapping.

If bloom is desired, reuse or add a general linear-image filter with
threshold, normalized kernel and strength as explicit parameters. Keep
an effect-disabled identity test, constant-image/filter tests, finite
results, documented border handling, exposure order and CPU/GPU agreement.
Apply it to linear pixels before display transfer, in shared X_eTaL.
An optional thin emissive rod is scene geometry; do not label it a
volumetric shaft. Full spectral rendering, volumetric transport and
lens diffraction/starbursts are not requested for this milestone.

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
