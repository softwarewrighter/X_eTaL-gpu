# Paired CPU/GPU image validation

These 16 by 8 images are a correctness fixture from the exact same
X_eTaL program and pinned libraries, not a presentation render or a
representative performance benchmark. The gallery's separate 1280 by 512
GPU still remains the high-resolution example.

The archive preserves CPU/GPU uncompressed RGB8 TIFFs, lossless PNGs,
linear RGB before display conversion, display RGB before quantization,
exact scene source, dependency pins and observed timing logs. Rendering
and display conversion are shared X_eTaL functions. The host only rounds
returned display RGB to bytes and encodes images. Raw linear values are
not clamped or converted to display values on the host.

Maximum linear RGB error is 3.4461402376173567e-7; maximum display RGB error
is 1.83249907248495e-7. The RGB8 pixel arrays are identical. Independent
checks decode TIFF tags, compression, strip payload and PNG rows, compare
against the raw display output and verify linear/display Float tolerances.

Run just ray-image-pair, then just ray-stage-pair and just ray-test-pair.
CPU comparison includes evaluator setup and text output; the GPU logs
separate initial setup and resident calls. Do not derive a full-resolution
speedup from this small fixture. All scene/library code and the assets are
MIT licensed; LICENSE is included in the uncompressed masters.zip.
