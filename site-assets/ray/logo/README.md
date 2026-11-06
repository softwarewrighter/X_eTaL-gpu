# Recorded logo media

These are curated inputs to the static site, not images rendered by the
browser. The X_eTaL CPU evaluator generated the 24 frames at 160 by 64;
a 320 by 128 still supplies the poster. Scene, engine/library pins,
angles and observed wall times are in manifest.json. scene.xtl is the
frozen scene used by this recording. Geometry, glyphs and scene work
follow this repository's MIT license and X_eTaL-libraries' MIT license.

logo.webm is a lossy VP9 derivative at 6 fps. PNG frames retain the
quantized RGB pixels exactly. masters.zip uses ZIP storage without
compression and contains uncompressed RGB8 TIFF frames, per-frame
X_eTaL sources and metadata, the larger still and the MIT license.
Raw linear/display arrays also remain in the ignored work directory on
the rendering host; they can be reproduced from the pinned sources.

Regenerate using just ray-animate-cpu, then the poster command and
just ray-stage-media described in docs/ray-integration.md. Rendering
requires the pinned evaluator/libraries, Python, libtiff tools and
FFmpeg with VP9 support. Static site builds only copy these media and
do not rerun the renderer. Later repo pin changes do not relabel older
recordings; each recording keeps its own exact source and pins.
