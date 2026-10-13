# X_eTaL-gpu

<p align="center">
  <b><a href="https://softwarewrighter.github.io/X_eTaL-gpu/">The live site</a></b>
  -- what this is, the measured results, and <a href="https://softwarewrighter.github.io/X_eTaL-gpu/doc/">the documentation</a>
  of every library, example and the GPU facade
</p>

X_eTaL programs on GPUs, old ones included.

[X_eTaL](https://github.com/softwarewrighter/X_eTaL), the eXperimental
Extensible Typed Array Language, states a computation as whole-array
operations: `c := a + b`, `'+ r_/ c`. Those operations already expose
their parallelism, so the same program can run as sequential software
(the X_eTaL evaluator), as thousands of parallel threads (a GPU, here)
or as a circuit (an FPGA, in
[X_eTaL-fpga](https://github.com/hardwarewrighter/X_eTaL-fpga)). This
repository is the GPU part: a small accelerator IR, OpenCL C 1.2
kernels generated from it, a host runtime that moves arrays to a
device and back, and the scheduling that decides how a device
realizes an array operation. The targets are old and new cards
alike, all reached through OpenCL: a Kepler Quadro (K4200) and
Maxwell and Pascal Teslas (M10, P4) that the newest CUDA toolkits
retire, and Turing to Blackwell
(GTX 1660 Super, RTX 3060, RTX 3060 Laptop, RTX 3090, RTX 4070 Laptop,
RTX 5060 Ti, A2), and Apple silicon
(M1 Max, where it is developed, and M3 Pro); the plan's hardware
table lists them.

The goal is the concept, one typed array program executing across
generations of parallel hardware with the same answer, not speed.

## What this is

| Part | What it holds |
| ---- | ------------- |
| `libs/Accel/` | the acceleratable subset of X_eTaL as an ordinary, typed, tested library (`"ac:" u_se< "Accel"`), and the example programs, its demos |
| `components/xir/` | the provisional accelerator IR (typed arrays, elementwise maps, reductions), its text form and a reference interpreter |
| `components/lower/` | the lowering: X_eTaL source to the IR, through the pinned X_eTaL's own crates (libraries inlined, shapes folded, every array operation an IR operation) |
| `components/opencl/` | the schedule and the OpenCL C 1.2 kernel emitter |
| `components/runtime/` | the OpenCL host runtime (devices, buffers, kernels) |
| `extensions/gpu/` | the GPU as a native extension X_eTaL programs call through X_eTaL-extensions' `xetal-x` |
| `components/cli/` | the `xetal-gpu` tool: `devices`, `check`, `run`, `kernel`, `explain` |
| `schedules/` | how each device runs a program (widths, work-group, tiles), one TOML file per device |
| `docs/` | the plan, the asks for X_eTaL, the research notes |

Every example is an X_eTaL program first; the pinned X_eTaL
evaluator is the reference, and the GPU must reproduce its answer:

```
                X_eTaL program (.xtl)
                        |
          +-------------+-------------+
          v             v             v
      evaluator    XIR interpreter   OpenCL on a GPU
          |             |             |
          +-------------+-------------+
                        v
                  the same arrays
```

**What runs on the GPU today, plainly:** X_eTaL source, lowered.
Since 2026-10-13 every example program is lowered from its `.xtl` by
this repository's translator ([`components/lower`](components/lower/README.md)),
which gets the program's Core, libraries included, from the pinned
X_eTaL's own crates and builds the XIR program the device runs. The
checks prove each lowered program gives the evaluator's output,
exactly on the interpreter and within f32's tolerance on the device.
The hand-written `.xir` twins that stood in until then were the
translator's oracle and are gone. X_eTaL-ML's benchmarks and world
model still run through a script that writes their XIR from the
definitions; they move to their own source next (the step
models-from-source). The lowering belongs to X_eTaL in the end (asks
G1 and G2), and what is built here moves there when it does.

## Status

Saga 1 (gpu-foundation) done on 2026-10-06: the process, the pinned
X_eTaL (75e6a5c) and the library tooling (step 1); the Accel
library, the acceleratable subset with its five example programs and
their baselines, the reference the GPU must reproduce (step 2); the
provisional IR with its text form, checker and reference interpreter,
the `xetal-gpu` tool (`check`, `run`, `print`), and the five examples'
hand-lowered twins, each proved equal to the evaluator's output
exactly (step 3); the schedule and the OpenCL C 1.2 emitter, fusing
each elementwise chain into one kernel and laying out reductions as
local-memory trees, with every twin's kernel source and plan pinned
(`xetal-gpu kernel`, `explain`; step 4); the OpenCL runtime: devices
listed, the plan executed on one, results read back, and every
example reproduced on this Mac's GPU (Apple M1 Max, OpenCL 1.2),
Ints exactly and Floats within 1e-5 (`xetal-gpu devices`, `run
--device opencl:0`; step 5); the reductions and the pipeline at
1024, 65536 and 1048576 items, generated data run through the
evaluator and the GPU alike, all agreeing (`just sizes`; step 6).
Saga 2 (gpu-algebra): the macros-and-extensions analysis
and reductions along an axis of a matrix (row sums, column sums, row
maxima) and inner products (`'+ '* i_nner`, untiled and in tiles; a
512 by 512 product in 0.13 s on the Apple M1 Max against 29 s in
this Mac's evaluator), and a tiny two-layer network (dense, ReLU,
dense, softmax) whose output on the M1 Max matches the evaluator's,
and schedules as files per device are done (saga 2, finished
2026-10-07). Saga 3 made the GPU a native extension that X_eTaL
programs call themselves, through X_eTaL-extensions' `xetal-x`: on
the M1 Max a 512 by 512 product runs 31 times faster than in the
evaluator, even through the bridge (`just offload`). Saga 4
(gpu-models, in progress) runs X_eTaL-ML's benchmarks and its world
model on every device here, and adds the operations a transformer
needs: transpose (a gather, or the last two axes in tiles; attention
scores and a layer's weight gradient as Accel exports) and the
convolution's windows (rotations along an axis, drops, a reshape, as
X_eTaL-ML's Conv builds them; a small convolution as one product)
and the lookup (`s_elect`: a batch of token ids picking rows of an
embedding table, the ids all a prompt uploads), and the choices,
argmax from the operations it had and a top-k mask as one operation. On 2026-10-12 the same code was run on
the RTX 5060 Ti (16 GB) in the Arch Linux machine, in its own lane:
every check passes there too, with its own schedules
(`schedules/nvidia-rtx5060-ti*.toml`) and reports. See
[`docs/plan.md`](docs/plan.md) for the roadmap and the architecture
decisions.

The committed world model (ML fac586c) now passes the gate on the
RTX too: the evaluator, interpreter, GPU and extension return the
same counts, 1897 and 348. Its kernels take 0.237 ms, the CLI process
317 ms, and the extension's text binding 458 ms before a 197 ms run.
The evaluator's 15.12 s includes the complete demo and dream rollout;
the GPU twin measures batched inference and counts only. See the
Arch reports below for the timing boundaries.

Binary file binding now avoids transferring input arrays as text:
the same world-model counts, with binding at 13 ms versus 460 ms
on the RTX. See [binary arrays](docs/binary-arrays.md) for the CLI
converter and `gp:b_indFile`, and
[the measurements](docs/rtx5060-binary-results.md).

Prepared [GPU sessions](docs/gpu-sessions.md) now keep compiled kernels
and bound weights resident across calls. The [OpenBLAS batch sweep](docs/rtx5060-session-results.md)
compares fresh-input inference against optimized CPU arithmetic. This small
model loses when input conversion and upload are included; resident-only
timings are faster at larger batches but repeat identical data.

## Results by GPU

Two GPUs so far, more to come; every number names its card. The
evaluator's time is its own host's, so each column compares a GPU
with the evaluator beside it, not the cards with each other (speed
is not the goal). Whole programs include starting the process,
reading the data and compiling the kernels; the extension's times
include the text bridge.

| Work | Evaluator on the Mac | Apple M1 Max | Evaluator on the Arch host | RTX 5060 Ti |
| ---- | -------------------- | ------------ | -------------------------- | ----------- |
| 512 by 512 product, whole program (`just sizes m512`, 16 by 16 tiles) | 28.75 s | 0.13 s | 55.12 s | 0.30 s |
| 512 by 512 product through the extension (`just offload`) | 14545 ms | 462 ms (31 times faster) | 24994 ms | 979 ms (25 times faster) |
| a million elementwise items, summed, through the extension | 188 ms | 2447 ms (13 times slower) | 254 ms | 1440 ms (5.7 times slower) |

Against tuned CPU code, not the evaluator: the trained world model
through a kept session (weights resident, a new batch uploaded each
call), beside NumPy's BLAS on the same host (Accelerate on the Mac,
OpenBLAS with the best of 1, 6 or 12 threads on the Arch host). Both
give the counts 1897 and 348.

| World model, batch | NumPy on the Mac | Apple M1 Max | NumPy on the Arch host | RTX 5060 Ti |
| ------------------ | ---------------- | ------------ | ---------------------- | ----------- |
| 1 state | 0.02 ms | 1.54 ms (0.01 times) | 0.07 ms | 0.21 ms (0.35 times) |
| 1,936 states | 1.29 ms | 2.51 ms (0.51 times) | 1.94 ms | 2.97 ms (0.65 times) |
| 8,192 states | 4.68 ms | 5.50 ms (0.85 times) | 8.64 ms | 10.53 ms (0.82 times) |

Neither GPU beats the CPU's BLAS on this small model at any batch
size: the kernels take a fraction of a millisecond, and the rest is
the call's fixed cost (on the M1 Max about 1.3 ms of launches even
for one state) and the batch's upload. The gap closes as the batch
grows; a model with more work per uploaded byte is where a GPU
starts to pay, which the model sagas are for.

The full tables: the M1 Max's in `components/runtime/README.md`,
[`extensions/gpu/README.md`](extensions/gpu/README.md) and
[`docs/ml-results.md`](docs/ml-results.md) (X_eTaL-ML's benchmarks
and its world model); the RTX 5060 Ti's in
[`docs/rtx5060-results.md`](docs/rtx5060-results.md) and
[`docs/rtx5060-ml-results.md`](docs/rtx5060-ml-results.md); the
session comparisons in [`docs/m1max-session-results.md`](docs/m1max-session-results.md)
and [`docs/rtx5060-session-results.md`](docs/rtx5060-session-results.md)
([how they are made](docs/gpu-sessions.md)).

## Build

Prerequisites:

- [Rust](https://rustup.rs) (1.95 or newer for the pinned X_eTaL; `just
  xetal` must have run before `cargo build`, since `components/lower`
  depends on the pinned X_eTaL's crates in `work/xetal`) and
  [`just`](https://github.com/casey/just)
  (`brew install just` or `cargo install just`)
- an OpenCL runtime for the GPU parts: on macOS it is built in
  (Apple's OpenCL 1.2); on Linux the vendor's ICD (for old NVIDIA
  cards, the driver's `libnvidia-opencl`) and the ICD loader
  (`ocl-icd`)
- for the gate (maintainers): `reg-rs` and `sw-markdown-checker`

```bash
just                                 # list the tasks
just xetal                           # clone and build the pinned X_eTaL (bin/xetal)
just eval "'+ r_/_2 2 3 r_eshape r_ange 6"   # try it: row sums, 6 15
just test                            # every library's baselines, the components, the equivalence checks
just gate                            # the pre-commit gate
just build                           # the xetal-gpu tool (target/release/xetal-gpu)
target/release/xetal-gpu lower libs/Accel/demos/reduce-sum.xtl  # the XIR an example lowers to
target/release/xetal-gpu run libs/Accel/demos/reduce-sum.xtl    # lowered and interpreted (--device opencl:0 for the GPU)
target/release/xetal-gpu kernel libs/Accel/demos/pipeline.xtl   # the OpenCL C it becomes
target/release/xetal-gpu explain libs/Accel/demos/pipeline.xtl  # which X_eTaL became which kernels
target/release/xetal-gpu devices                               # the OpenCL devices here
target/release/xetal-gpu run libs/Accel/demos/pipeline.xir --device opencl:0   # the same program on the GPU
just sizes                           # a million-item reduction: evaluator, interpreter, GPU, timed
just xetal-x && just gpu-ext         # X_eTaL-extensions' xetal-x and the gpu extension beside it
just offload                         # X_eTaL calling the GPU itself: a 512 by 512 product, timed
just ml-bench                        # X_eTaL-ML's benchmarks on every device, docs/ml-results.md
```

X_eTaL-extensions is pinned the same way (`XETAL_EXTENSIONS_COMMIT`,
`just xetal-x`, cloned into `work/xetal-extensions/`): its `xetal-x`
is X_eTaL's CLI plus native extensions, which the GPU extension
needs. It embeds the X_eTaL that X_eTaL-extensions pins (v0.1.0),
so the gate checks that every library program here prints the same
under it.

X_eTaL-ML is pinned too (`XETAL_ML_COMMIT`, cloned into
`work/xetal-ml/` by `scripts/xetal-ml.sh`): its benchmarks and models
run here on the devices, at its ask (`just ml-bench` writes
[`docs/ml-results.md`](docs/ml-results.md)).

X_eTaL is not tracked here: `XETAL_COMMIT` pins a known-good commit
(one line, the full SHA), `just xetal` clones X_eTaL into `work/xetal/`
(gitignored), checks it out, builds the CLI and links it as
`bin/xetal`. The first build takes a few minutes;
`XETAL_SOURCE=../X_eTaL just xetal` clones from a sibling checkout
instead of GitHub. Maintainers move to a newer X_eTaL with
`just xetal-pin [REF]`, in its own commit after `just gate` passes.

## Libraries and examples

```bash
just libs                            # the libraries: name, alias, what
just run-lib Accel                   # a library's test programs
just demo-lib Accel vector-add       # one of its demos (an example program)
just types Accel                     # its exported names and types
just test-lib Accel                  # its reg-rs baselines
```

Each library is its own directory, `libs/<Name>/` (the layout of
X_eTaL-libraries): `src/<Name>.xtl`, `tests/` (reg-rs baselines and
the pinned export types), `demos/` (programs that use it; here, the
example programs the GPU runs) and `docs/README.md` (the reference).
`just path` prints the directories for `XETAL_PATH`.

## Documentation

- [The live site](https://softwarewrighter.github.io/X_eTaL-gpu/) and
  its [`doc/`](https://softwarewrighter.github.io/X_eTaL-gpu/doc/):
  `xetal doc` of the libraries, the example programs and the GPU
  facade, their `##` comments and running examples, every name
  linked (`just pages` builds it into `pages/`, `just publish`
  publishes it, `just check-live` checks it)
- [`docs/plan.md`](docs/plan.md) -- architecture decisions, the
  subset, the roadmap
- [`docs/xetal-asks.md`](docs/xetal-asks.md) -- what this repo needs
  from X_eTaL; the blockers first
- [`docs/ml-results.md`](docs/ml-results.md) -- X_eTaL-ML's benchmarks
  and its world model on every device here, the device's time and
  the bridge's apart
- [`docs/tuple-plan.md`](docs/tuple-plan.md) -- where X_eTaL's
  tuples would help here, what blocks each use, and when to adopt them
- [`docs/macros-and-extensions.md`](docs/macros-and-extensions.md) --
  where macros and native extensions are justified here, and where
  not, with measurements
- `docs/research7.txt` -- the planning conversation (archival, not
  normative)
- [`CHANGES.md`](CHANGES.md) -- every commit
- [`CLAUDE.md`](CLAUDE.md) (also `AGENTS.md`) -- the agent workflow
  (agentrail sagas) and rules

## Development

Development is tracked with agentrail sagas, as in the sibling
repositories: `agentrail status` shows the current step,
`agentrail next` its instructions. Every step ends with the gate
passing, docs updated, a commit to `main` and a push.

## Related Projects

- [X_eTaL](https://github.com/softwarewrighter/X_eTaL) -- the language
  ([try it live](https://softwarewrighter.github.io/X_eTaL/))
- [X_eTaL-fpga](https://github.com/hardwarewrighter/X_eTaL-fpga) --
  the same array programs as hardware
- [X_eTaL-ML](https://github.com/softwarewrighter/X_eTaL-ML) -- the
  machine-learning demos and libraries (the models to run here)
- [X_eTaL-demos](https://github.com/softwarewrighter/X_eTaL-demos) --
  visual array programs
- [X_eTaL-libraries](https://github.com/softwarewrighter/X_eTaL-libraries)
  -- reusable `.xtl` and `.xtlm` libraries

## Links

- Blog: [Software Wrighter Lab](https://software-wrighter-lab.github.io/)
- Discord: [Join the community](https://discord.com/invite/Ctzk5uHggZ)
- YouTube: [Software Wrighter](https://www.youtube.com/@SoftwareWrighter)

## Copyright

Copyright (c) 2026 Michael A Wright

## License

MIT. See [`LICENSE`](LICENSE) and [`COPYRIGHT`](COPYRIGHT).
