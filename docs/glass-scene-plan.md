# Glass spheres and crystals in X_eTaL

## Goal and reference

Produce an original still render inspired by the user's local reference,
`~/block-ray-tracing.jpg`, inspected on 2026-11-06. It shows clear spheres
and faceted glass solids on a dark reflective floor, bright edge highlights,
blurred foreground/background, rainbow accents and long light streaks.
Match the composition and glass appearance approximately; pixel matching
and reproducing every optical effect are not acceptance requirements.
Do not redistribute the reference without established permission, use it
as a texture, or copy another renderer's implementation/assets.

This is a second scene milestone after the initial renderer and logo in
[ray-tracing-plan.md](ray-tracing-plan.md). It preserves one X_eTaL
algorithm and scene for CPU and GPU. All intersections, optical transport,
sampling, accumulation and optional image effects run in X_eTaL. The host
selects the device, transfers arrays, launches work and encodes pixels as
uncompressed TIFF. No Blender, native rendering engine or GPL/AGPL code.

Current RL1-RL4 requests provide 3D math, opaque rendering, extrusion and
scene macros. They do not yet provide glass. General library additions
RL5-RL7 below are required for the glass milestone; RL8/RL9 are deferred
refinements. See [library asks](xetal-libraries-ask.md) for contracts,
acceptance tests and required delivery points. Do not expand the original
logo's requirements or silently change its v1 arrays.

## Scene to build

Use right-handed coordinates, y up, and scene units near an object's
radius. Proposed starting values are adjustable scene data:

| Element | Initial setup |
| ------- | ------------- |
| Floor | Large finite rectangle at y=0, two triangles, dark gray RGB about (0.025,0.03,0.035), opaque reflection weight about 0.35 |
| Central sphere | Center (0,0.85,0), radius 0.85 |
| Front-left sphere | Center (-1.45,0.55,1.1), radius 0.55 |
| Rear-left sphere | Center (-1.6,0.7,-1.4), radius 0.7 |
| Crystals | Three to five closed octahedra or other low-face-count convex polyhedra to the right and rear; varied scales and rigid tilts, resting on the floor |
| Camera | Eye (5,2.8,7), target (0,0.8,0), up (0,1,0), vertical fov about 0.6 radians; focus on central sphere |
| Glass | Clear dielectric, initial IOR 1.5, low absorption, smooth interfaces |
| Background | Dark blue-gray, with controllable reflected bright panels |
| Main light | Strong rectangular emitter behind/right, initially near (3,5,-3), aimed at the objects |
| Fill | Dim broad emitter toward front/left; keep most of the image dark |

Check bounds and object intersections when placing assets. V2 initially
supports air and non-overlapping closed glass objects, not nested or
interpenetrating media. Reserve space between objects and keep their
bottoms at y=0. These numbers are a starting layout, not an assertion
that every proposed radius/position has already been rendered.

Place additional white emissive rectangles outside the camera frame but
visible in reflections. Their broad shapes give readable glass edges.
Use flat face normals on crystals and analytic normals on spheres.
A lightly rough floor is a possible later refinement; a dark mirror
with controlled reflection is sufficient for the first image.

The scene macro should describe these objects, materials, groups, lights
and camera, then emit numeric arrays. Example notation below proposes
extensions to RL3; it is not an implemented API:

```text
material glass dielectric ior 1.5 transmission 1 absorption 0 0 0
material floor color .025 .03 .035 reflection .35
sphere center 0 .85 0 radius .85 material glass
polyhedron shape octahedron material glass
plane height 0 material floor
camera eye 5 2.8 7 target 0 .8 0 up 0 1 0 fov .6
lens radius .025 focus 8
```

An infinite-looking plane may expand to a large finite rectangle using
the existing triangle representation. It does not require an infinite
plane primitive. The final grammar and precise field names belong to
the library owner, with expansions and ordinary constructors agreeing.

## Render stages and local steps

The four glass-prefixed steps are queued after the existing ray steps;
this plan does not reprioritize Mojo/ML work or block the original logo.
Pin committed library deliveries deliberately before using them.

| Step | Work | Acceptance and upstream dependency |
| ---- | ---- | ---------------------------------- |
| glass-composition | Assemble an original low-cost opaque/mirror version with spheres, crystals, floor, camera and bright panels. Check silhouette, framing and placement at 256 by 256. | RL1/RL3/RL4 plus RL6 geometry; CPU/GPU raw pixels agree; diagnostic uncompressed TIFF from both. Glass is not claimed yet. |
| glass-transport | Integrate RL5 dielectric transport and compatible v2 material arrays; test closed-object entry/exit and multiple interactions. | Analytical optics cases and tiny CPU/GPU images pass; clear spheres and crystals transmit and reflect, all values finite; fixed ray budget. |
| glass-camera | Integrate RL7 deterministic thin-lens, emitter and image sampling. Tune focus, broad highlights and shadows at 512 by 512. | Aperture-zero pinhole equivalence, fixed-seed sample checks and CPU/GPU raw-image agreement; record noise/convergence and lens settings. |
| glass-render | Render a polished 1024 by 1024 still on RTX 5060 Ti, matching the CPU algorithm/settings; deliver raw arrays, uncompressed TIFF and benchmark report. | Reduced/full-resolution verification as practical, repeated timing and quality checks, reproducible scene/seed/library pins and commands; no assumption of GPU speedup. |

The composition diagnostic uses v1 point lights and lit white triangles
as provisional reflection panels; RL7 replaces them with emissive/area
lighting before the polished render. RL6 opaque sphere support can land
before RL5 glass transport.

Begin with one sample per pixel and the pinhole camera, then sweep
4, 16 and 64 samples where useful. Start transport tests at two surface
interactions; sweep caps 4, 8 and 12 for glass, recording convergence
rather than assuming one cap is sufficient. Record sample count and
interaction cap in every image manifest and timing row. A cap is a render
parameter and cannot quietly differ between CPU and GPU.

The deliverable milestone is recognizable clear spheres and faceted
crystals on a dark reflective floor, with controlled highlights and
focus blur. Rainbow fringes, floor caustics, atmospheric shafts and lens
starbursts are optional enhancements, not hidden requirements for this
milestone. They must be labeled when approximated.

## Optical transport and execution

Use reflection/refraction directions, Fresnel interface weights and total
internal reflection. Track whether a ray is in air or a particular glass
object, geometric versus oriented normals, throughput and remaining
interaction budget. Apply absorption by distance inside the medium.
Watertight closed objects and documented edge ties are prerequisites.
Refuse unsupported nested/intersecting glass rather than assigning the
wrong medium silently. Glass shadows must not accidentally inherit the
opaque renderer's all-or-nothing rule without being labeled approximate.

Avoid a binary ray tree that doubles at every interaction. Use bounded
arrays of active paths, with deterministic samples selecting reflected or
transmitted paths according to documented weights, or another explicitly
bounded method with the same CPU/GPU estimator. Use sample identity
(pixel, sample, interaction, dimension, seed) rather than thread order.
Multiple scattering events are fixed iterations over arrays; no device
recursion or renderer-specific IR instruction is needed.

Prefer analytic sphere intersections to very dense sphere meshes. Keep
crystals as triangles and unify nearest-hit comparisons across geometry
kinds. Chunk rays and geometry to bound working memory; never allocate
pixels times samples times all geometry times an expanding path tree.
If measurements justify acceleration structures, insert a separately
scoped library/backend step; a BVH is not a language primitive.

GPU eligibility checks must cover ordinary library validation paths as
well as arithmetic. Where preflight validation is necessary, it is shared
X_eTaL run before the accelerated region, not host reimplementation of
the renderer. Report unsupported lowering constructs and add local GPU
support for existing operations. Do not remove validation to make a
program appear GPU-compatible.

## Verification, output and performance

Test Snell directions, Fresnel bounds, normal-incidence reflectance,
critical-angle behavior, entry/exit, absorption, ray offsets and clipping
before a large render. Use identical deterministic samples on CPU/GPU.
Compare raw linear RGB before display conversion; report tolerances,
maximum error, nonfinite counts and pixels outside tolerance. Identify
boundary-sensitive rays instead of concealing mismatches with blur.

Use the existing ray plan's separation of evaluator/XIR CPU reference,
GPU device intervals, full render-call wall time, cold setup/build, data
transfers and file writing. Report CPU implementation, precision, device,
schedules, pins, seeds, interaction caps, sample counts, warmups, repeated
median/spread and CPU/GPU speedup or slowdown. Matching visual quality
requires matching samples and transport settings. Do not infer performance
from the tiny diagnostic scene or compare different image settings.
Report speedup only for resolutions actually timed on both CPU and GPU.
If CPU cost limits full-image verification, label the final image GPU-only
and report the paired benchmark at smaller resolutions without extrapolating
a full-image CPU time. Match precision for strict backend speedups; label
f64-evaluator versus f32-device comparisons separately.

Keep raw linear Float arrays and a manifest alongside an uncompressed
RGB TIFF. State exposure, display transfer and clipping. A 16-bit/channel
TIFF option is useful for final highlights, while raw Float arrays remain
the correctness reference. Inspect the floor/reflections, glass boundaries,
object silhouettes, absence of self-hit artifacts, and focused versus
blurred areas. A manifest should describe output bit depth and all optional
effects. Ignore large render output in git; commit tiny numeric fixtures,
scene specifications and reports. Add just recipes when implementation
lands; commands in this document are not claims of existing tooling.

## Optional refinements

RL8 requests a separately scoped caustic stage: trace photons from lights
through glass and estimate their contribution on the floor. This is more
appropriate for concentrated glass illumination than expecting the basic
camera-ray renderer to discover it efficiently. Keep both photon and
camera calculations in X_eTaL with bounded memory and deterministic tests.

RL9 requests optional RGB dispersion and restrained bloom in shared
X_eTaL. RGB-dependent IOR is an approximation to spectral dispersion and
must be labeled. Thin emissive rods can approximate reference streaks,
but they represent geometry, not simulated volumetric light. Full spectral
transport, participating media, diffraction/starburst simulation, arbitrary
fonts and a general texture system are outside the initial milestone.
Request and queue their implementation only if the first render needs it.

## Algorithm references

These are explanations of capabilities and equations, not implementation
sources to copy. Original X_eTaL implementation and compatible provenance
remain required:

- [PBRT: specular reflection and transmission](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission)
- [PBRT: camera models and depth of field](https://www.pbr-book.org/3ed-2018/Camera_Models/Realistic_Cameras)
- [PBRT: stochastic progressive photon mapping](https://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping)
