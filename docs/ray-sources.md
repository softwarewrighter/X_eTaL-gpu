# Ray-tracing source guide

Every rendered demo in the [live gallery](https://softwarewrighter.github.io/X_eTaL-gpu/ray-tracing/)
now has a visible Source code section. Both pipe video dialogs also link
directly to geometry and tracing source.

| Render | Authoring scene in this repo | Exact recording source |
| --- | --- | --- |
| Pipe fly-through | [pipe-flythrough.xtl](../examples/ray/pipe-flythrough.xtl) | [geometry generation](https://softwarewrighter.github.io/X_eTaL-gpu/ray-tracing/media/pipes-rtx/geometry.xtl), [GPU tracing function](https://softwarewrighter.github.io/X_eTaL-gpu/ray-tracing/media/pipes-rtx/function.xtl) |
| Native GPU logo | [logo.xtl](../examples/ray/logo.xtl) | [complete executed program](https://softwarewrighter.github.io/X_eTaL-gpu/ray-tracing/media/logo-gpu/scene.xtl) |
| Logo animation | [logo.xtl](../examples/ray/logo.xtl) | [recorded scene template](https://softwarewrighter.github.io/X_eTaL-gpu/ray-tracing/media/logo/scene.xtl); frame-specific programs are in the linked master archive |
| Native CPU logo | [logo.xtl](../examples/ray/logo.xtl) | [equivalent full-frame program](https://softwarewrighter.github.io/X_eTaL-gpu/ray-tracing/media/logo-still/scene.xtl); exact executed strips are in the linked master archive |
| Glass transport preview | [glass-transport.xtl](../examples/ray/glass-transport.xtl) | [complete recorded scene/renderer](https://softwarewrighter.github.io/X_eTaL-gpu/ray-tracing/media/glass-rtx/scene.xtl) |
| Small CPU/GPU validation | [logo.xtl](../examples/ray/logo.xtl) | [identical-source comparison program](https://softwarewrighter.github.io/X_eTaL-gpu/ray-tracing/media/image-pair/scene.xtl) |

Current authoring files can evolve. The served recording files are frozen
with their media and validated against each recording's source hashes.
A separate Browse link opens each recording file on GitHub at its last
committed revision, rather than a moving main URL. Python driver links
are labeled separately: they orchestrate execution and encode returned
pixels; intersections, lighting and geometry remain X_eTaL. Driver links
show the current committed orchestration code; the frozen X_eTaL programs
and recording manifests are authoritative for an older render.

## Library cross-references and versions

The source sections link to current cross-reference pages for RayTrace,
Vector3 and, where used, Scene, Extrusion and Glass. Scene and Extrusion
also have macro-library cross-references. For example:

- [RayTrace shading](https://softwarewrighter.github.io/X_eTaL-libraries/doc/libs-RayTrace-src-RayTrace.xtl.html#l.s_hade)
- [RayTrace camera rays](https://softwarewrighter.github.io/X_eTaL-libraries/doc/libs-RayTrace-src-RayTrace.xtl.html#l.r_ays)
- [Scene macro source cross-reference](https://softwarewrighter.github.io/X_eTaL-libraries/doc/libs-Scene-src-Scene.xtlm.html)
- [Glass transport](https://softwarewrighter.github.io/X_eTaL-libraries/doc/libs-Glass-src-Glass.xtl.html#l.t_race)

Those rendered docs follow the library site's current version and can be
newer than a recorded image. Each source section therefore also links to
the exact library commit from that recording's manifest and to individual
X_eTaL/macro files at that commit. The logo animation, CPU still and newer
GPU recordings use different library revisions; do not substitute the
latest library silently when reproducing an older recording.

The benchmark illustration remains planned and links to its validation
plan. It is not presented as a rendered demo with nonexistent source.

## Verification

`just ray-test-sources` checks built pages for six recording-specific
source groups, actual matching public files, immutable GitHub source
links, the correct recorded library revision, xref links, and source links
in both video dialogs. It runs in the full gate after the site build.
The browser check exercises those dialogs while retaining playback,
wraparound and responsive-layout checks.
