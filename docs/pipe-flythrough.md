# Pipe fly-through

An original, actually ray-traced camera flight through hollow pipes on a
plane under a blue sky. The two-pipe layout repeats every 36 scene units.
Pipes span z=0..8 and z=18..26, leaving ten-unit open gaps. A red sphere
stands to screen right at z=14; a gold pyramid stands to screen left at
z=32. The +z camera basis has screen right along -x. Pipe outer radius is
2.2, inner radius 1.9, with center y=2.2, so each pipe rests on the plane.
The camera is at y=2.15, clear of both walls and the ground.

## Rendered animation

The native animation is 640 by 360, 180 frames at 15 fps, twelve seconds.
It uses one sample per pixel, flat triangle normals, direct diffuse light,
hard shadows and a blue environment. It is not antialiased. Additional
subpixel samples and averaging are required for a smoother future render;
video compression is not an antialiasing substitute.

All mesh generation, camera rays, intersections and shading are ordinary
X_eTaL, with shared RayTrace math. The host does input progression,
validation, pixel quantization and media encoding. The private demo
geometry fixes its own radii, orientation and tessellation; it is not a
new general geometry library or a native ray tracer. The scene has 1,142
triangles, including three copies of the periodic cell and a ground plane.

The camera moves at constant speed. Each frame's phase is frame modulo
180, so frame 180 returns to phase zero. That endpoint is independently
rendered on the GPU and its PNG must equal frame zero byte for byte.
Only frames 0..179 are encoded: no duplicate endpoint, cross-fade, reverse
playback or pause is inserted. Both the thumbnail and full-size dialog
use native looping playback. The dialog displays the native video width.
Reduced-motion visitors get paused video with controls.

## GPU batching and evidence

Six frames are evaluated together in one flattened frame/pixel/channel
region. The current adapter traces each pixel three times, once per color
component, as the existing logo adapter does. Static meshes are evaluated
once by the pinned X_eTaL evaluator, then uploaded as immutable data.
Explicit typed function input frameStart prevents constant folding of the
camera phase. A device-valued zero term keeps the index vector mapped in
function lowering; without it static each would expand every pixel call.
This adaptation is visible in the source and adds no language feature.

One prepared OpenCL session handles 31 executions: thirty six-frame video
batches plus a six-frame wraparound check. The scene/kernel buffers remain
resident; only the scalar frame offset changes. The batch outputs remain
on device until that batch is complete, then returned as linear and sRGB
Float arrays. Planned reusable buffers total 182,703,040 bytes, with 34
kernels, 47 launches per batch and 246,511 bytes of generated C. Driver
scratch memory is additional. Larger batches are supported with --batch;
six bounds host ASCII parsing memory as well as device storage.

The final render process took 116.27 seconds for all 31 executions,
including setup, readback and saved ASCII output. Frame encoding adds
host work. These are single RTX 5060 Ti observations, not an isolated
CPU/GPU benchmark. A matching tiny CPU/GPU run has identical RGB8 pixels,
linear error 2.35e-8 and display error 9.50e-8. The camera's small vertical
offset avoids placing the tiny test rays exactly on faceted pipe seams;
ambiguous f32 edge hits can choose adjacent faces, as the Mac report also
records. Geometry checks reject zero-area faces and verify sphere bounds.
An early authoring error in trig-expression grouping was fixed before the
published sequence was rendered.

Every native frame is preserved as uncompressed RGB8 TIFF in six stored
ZIP archives, with exact X_eTaL mesh/function sources, numeric inputs and
compiled XIR. The local render directory also preserves all raw Float
batch outputs. Public metadata records pins, executable/source hashes,
resident timings, the small CPU/GPU fixture and encoded format comparison.
No full-resolution CPU equivalence or antialiased quality is claimed.

## Reproduce

```sh
just build
just ray-pipe-render --width 640 --height 360 --frames 180 --batch 6 --fps 15 --output work/ray-pipe-publish
just ray-pipe-render --width 2 --height 2 --frames 4 --batch 2 --compare-cpu --output work/ray-pipe-publish-validation
just ray-pipe-media --input work/ray-pipe-publish --validation work/ray-pipe-publish-validation
just ray-test-pipes
```

The media task encodes MP4/H.264, VP9/WebM, AV1/WebM, animated WebP and
GIF at the native dimensions and frame rate. Eligibility requires a
successful full decode with the exact frame count/duration and average
PSNR of at least 30 dB against the traced PNG sequence. Unsupported
animated decoders are recorded as ineligible. The smallest qualifying
candidate supplies both animated views; MP4 remains downloadable and is
a video fallback. This is a measured choice among these tested settings,
not a claim of globally optimal compression.

## Portability

Compiler semantics and dependency pins remain unchanged. The CLI exposes
existing typed-function lowering and resident input rebinding; RTX media
lives under site-assets/ray/pipes-rtx. Incoming Apple M1 Max results were
rebased and retained. This video is an executed RTX result, not an Apple
GPU performance or compatibility claim.
