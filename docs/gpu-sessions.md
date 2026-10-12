# GPU sessions

Compile once, bind weights once, and run successive input batches in the
same process. A prepared session owns the context, queue, kernels and
all device buffers. Every run reads the outputs back.

## X_eTaL extension

Load a program with `gp:l_oad`, bind its inputs with `gp:b_ind` or
`gp:b_indFile`, then call `gp:r_un 0`. Later calls to `gp:r_erun 0`
reuse compiled kernels and buffers. Rebind the input batch before each
call; unchanged weights remain resident. A changed binding is uploaded
on the next rerun. The session's program fixes all input shapes.

Loading a program, changing the schedule or switching devices rebuilds
the session. `gp:r_un` always starts fresh; `gp:r_erun` can also prepare
the first run. Device -1 uses the reference interpreter. Bindings with
invalid shapes are rejected before replacing existing values.

## CLI

```bash
just build
target/release/xetal-gpu run model.xir --device opencl:0 --repeat 21 --time \
  --bind X=@batch.xta --bind w=@weights.xta --refresh X
```

`--repeat N` prints only the final outputs and emits timing for every
run to standard error. `--refresh X` reuploads the same input bytes on
subsequent calls, to measure transfer costs while weights remain resident.
It does not generate new batches. Without refresh all inputs stay resident.
The CPU interpreter also supports repeated execution and timing.

## Trained-model comparison

```bash
just session-setup
just session-bench --require-device
```

Setup installs pinned NumPy and threadpoolctl into an isolated environment
under ignored `work/`. The NumPy wheel supplies optimized OpenBLAS. The
benchmark prepares the pinned world model, checks exact prediction counts
against the reference interpreter, and compares 1, 6 and 12 CPU threads
with GPU tiles 0, 8 and 16. Twenty-one calls per setting provide medians
after three warmup calls. Setup, resident execution, refreshed-input
execution and break-even calls are reported separately.

See [the RTX results](rtx5060-session-results.md). The small trained model
loses to OpenBLAS with new input transfers at every measured batch size.
Keeping all inputs resident demonstrates reuse but does not demonstrate
useful processing of new requests. Host scalar conversion and input upload
dominate the refreshed GPU runs. The next useful workload is batched
embedding retrieval with a large resident document index: many comparisons
per transferred query, followed by reduction or top-k selection.

The gate checks repeated outputs and timing records on a small batch without
requiring NumPy. Runtime and raw extension tests verify rebinding,
unchanged weights, rejected bindings and program/schedule/device changes.
No disk kernel cache or cross-process residency is provided.
