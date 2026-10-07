# Saga: gpu-extension

Goal: X_eTaL programs call the GPU themselves, through
X_eTaL-extensions' ABI V1 and its xetal-x bridge, as
docs/macros-and-extensions.md justifies (X1: OpenCL is a C-ABI
library; products and models gain on the device even through the
text bridge; elementwise work does not, and the docs say so).
Testable on this Mac. Nothing in X_eTaL-extensions changes: it is
pinned like X_eTaL and its SDK used by path from a clone in work/.

1. pin-extensions: XETAL_EXTENSIONS_COMMIT, scripts/xetal-extensions.sh
   (clone into work/xetal-extensions, build xetal-x and hello), a
   gate check that xetal-x answers and loads hello.
2. gpu-package: extensions/gpu/ (extension.toml, rust/ a cdylib on
   xetal-ext-sdk and this repo's runtime: devices, load, bind, run,
   output, explain; one session per thread), Rust tests.
3. gpu-facade: extensions/gpu/lib/Gpu.xtl by ffi:b_ind<, its X_eTaL
   half (program writers for products and layers, r_un1, n_ear), reg-rs
   tests run with xetal-x, types pinned.
4. offload-demo: one X_eTaL program computing a 512 by 512 product with
   the evaluator and on the GPU through the extension, compared and
   timed; the bridge's cost stated; docs and retrospective.
