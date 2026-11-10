# Native CPU logo still

The 1280 by 512 PNG and uncompressed RGB8 TIFF contain newly traced
pixels, using one centered sample per pixel. This is a CPU preview,
separate from the earlier low-resolution video and from GPU benchmarking.

The X_eTaL evaluator rendered 32 full-camera strips with four workers.
The manifest records actual wall times, language/library commits and
strip-source hashes. The library snapshot is a2d4de1; it is not an
adoption of that revision as the GPU dependency pin. Worktrees contain
only committed, unmodified library sources. The scene and libraries are
MIT licensed, as recorded by LICENSE in the uncompressed masters.zip.

scene.xtl is the equivalent full-frame program. The exact executed
strip programs and their metadata are in masters.zip; no host renderer
or image upscaling is used. Strip pixels are concatenated without image
math. Antialiasing, when requested, averages linear samples in X_eTaL
before the shared sRGB transfer. This first still uses one sample.

Reproduce with just ray-hires-cpu --libraries-ref followed by the full
library commit from manifest.json. Defaults are 1280 by 512, four workers,
16-row strips and one sample; --samples-axis 2 selects four samples.
The reported sequence wall time includes setup, text output and encoding;
the sum of concurrent strip times is not a measured serial baseline.
