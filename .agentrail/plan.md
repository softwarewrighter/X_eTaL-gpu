# Saga: gpu-algebra

Goal: axes and inner products, the operations a model is made of:
reductions along an axis of a matrix, matrix-vector and matrix
products, a dense layer with ReLU and a softmax by row, and the
schedule as a file per device. Every step keeps the discipline of
saga 1: an .xtl first (the Accel library grows), its baseline, a
hand-lowered twin, the interpreter exact, the GPU within tolerance
(docs/plan.md, saga 2; docs/xetal-asks.md for anything missing).

1. axis-reduce: reduce along an axis of a matrix (r_/_2 and r_/),
   one work-item per row or column; the checker's rank rules; the
   emitter and the interpreter; Accel exports (r_owSums, c_olSums,
   r_owMax); demos and twins; the pinned kernels.
2. matvec: matmul for matrix-vector and matrix-matrix (x '+ '*
   i_nner w): one work-item per output element, then a tiled
   schedule; ac:m_atvec, ac:m_atmul; sizes to 1024 by 1024 in
   check-sizes; twins; kernels pinned.
3. dense-layer: y = relu(W x + b) as X_eTaL (ac:d_ense with the bias
   row, as X_eTaL-ML's NN) and on the GPU; a softmax by row (exp,
   row reduce, divide); twins; equivalence.
4. schedules: the schedule as a TOML file per device (work-group,
   tile, element widths); xetal-gpu run --schedule FILE; explain
   shows it; a file for the Apple M1 Max and templates for the old
   NVIDIA cards.
