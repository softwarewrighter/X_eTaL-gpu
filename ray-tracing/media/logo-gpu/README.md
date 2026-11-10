# Native GPU logo still

This 1280 by 512 image is a new OpenCL render on the RTX 5060 Ti,
with shared MIT-licensed X_eTaL RayTrace math and one sample per pixel.
The scalar channel adapter traces each pixel three times. Scene geometry
setup may use the evaluator; shading and ray intersections run on GPU.

The second resident run measured 890.943 ms of device work and 1213.376 ms
total, including final output readback. Initial setup took 5003.902 ms.
These are individual observations, not a statistical benchmark. The
manifest preserves full logs, dependency pins, executable hash and the
modified-tree status of the prototype build committed with this asset.
The earlier CPU still used another library revision and parallel strips;
it is not a matched CPU timing baseline for this GPU frame.

The exact same generated X_eTaL program at 8 by 4 passed against the
pinned evaluator, with maximum display-RGB error below 1.7e-7. The archive
includes that small program and both outputs. It does not claim a full
1280 by 512 CPU comparison. TIFF pixels and the ZIP are uncompressed;
PNG is the convenient lossless browser image. No upscaling is used.

Reproduce with just ray-render-regions --width 1280 --height 512.
Use --compare-cpu on a small image to repeat the matched-source check.
The scene source and rendering libraries are MIT licensed; LICENSE is
included in masters.zip.
