# Saga: gpu-models

Goal: research7's PoC G2: a tiny Jev-like typed-decision model (a
small transformer-like classifier, trained from scratch, its inference
written as X_eTaL array code) whose complete inference runs on a GPU,
with the same decision from the evaluator, the interpreter and the
device. The model is the architecture as X_eTaL code and the weights
as data. XIR operations are added only when the model needs them and
stay general X_eTaL primitives (attention decomposes; it is not an
intrinsic). Testable on this Mac; the Arch machine's newer GPU later.

1. tiny-nn: a complete tiny trained network (two dense layers on a
   small task, a std-only Rust trainer with a fixed seed), weights as
   data files, inference in X_eTaL with Accel, its twin, the same
   decisions on CPU and GPU, also through the gpu extension.
2. jev-model: examples/jev/ model.xtl (embedding lookup, attention as
   array operations, normalization, feed-forward, classifier),
   tokenizer.xtl, config.toml, weights (random until trained); the
   XIR operations it needs, general ones only.
3. jev-train: a trainer for the typed-decision task (requests
   classified as DRAW, ARRAY_QUERY, EXPLAIN, EXECUTE, DOC_SEARCH), f32,
   std-only Rust, fixed seed, the data set generated and committed
   small.
4. jev-gpu: the whole inference scheduled as OpenCL kernels; the same
   prompt, model and weights giving the same decision on CPU and GPU;
   explain shows which X_eTaL operations became which kernels.
