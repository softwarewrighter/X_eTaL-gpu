# X_eTaL-gpu

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
realizes an array operation. The target is deliberately old
hardware: Kepler, Maxwell, Pascal, Turing and Ampere cards that the
current CUDA toolkits have retired, reached through OpenCL.

The goal is the concept, one typed array program executing across
generations of parallel hardware with the same answer, not speed.

## What this is

| Part | What it holds |
| ---- | ------------- |
| `libs/Accel/` | the acceleratable subset of X_eTaL as an ordinary, typed, tested library (`"ac:" u_se< "Accel"`), and the example programs, its demos |
| `components/xir/` | the provisional accelerator IR (typed arrays, elementwise maps, reductions), its text form and a reference interpreter |
| `components/opencl/` | the schedule and the OpenCL C 1.2 kernel emitter |
| `components/runtime/` | the OpenCL host runtime (devices, buffers, kernels) |
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

The accelerator IR and the lowering from X_eTaL's Core IR to it belong
to X_eTaL and are not there yet (see
[`docs/xetal-asks.md`](docs/xetal-asks.md), asks G1 to G4), so for now
each example's `.xtl` has a hand-lowered `.xir` twin, and a check
proves the twin's interpretation equals the evaluator's output. The
twins go when X_eTaL's lowering lands.

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
512 by 512 product in 0.13 s against the evaluator's 29 s), and a
tiny two-layer network (dense, ReLU, dense, softmax) whose output on
the GPU matches the evaluator's, and schedules as files per device
are done (saga 2, finished 2026-10-07). Next, saga 3: the GPU as a
native extension X_eTaL programs call themselves (saga 3). The
work moves to an Arch machine with a newer GPU soon; nothing in the
code changes for it (see Build). See [`docs/plan.md`](docs/plan.md) for
the roadmap and the architecture decisions.

## Build

Prerequisites:

- [Rust](https://rustup.rs) (stable) and
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
target/release/xetal-gpu run libs/Accel/demos/reduce-sum.xir   # the IR twin of an example, interpreted
target/release/xetal-gpu kernel libs/Accel/demos/pipeline.xir  # the OpenCL C it becomes
target/release/xetal-gpu explain libs/Accel/demos/pipeline.xir # which values became which kernels
target/release/xetal-gpu devices                               # the OpenCL devices here
target/release/xetal-gpu run libs/Accel/demos/pipeline.xir --device opencl:0   # the same program on the GPU
just sizes                           # a million-item reduction: evaluator, interpreter, GPU, timed
```

X_eTaL-extensions is pinned the same way (`XETAL_EXTENSIONS_COMMIT`,
`just xetal-x`, cloned into `work/xetal-extensions/`): its `xetal-x`
is X_eTaL's CLI plus native extensions, which the GPU extension
needs. It embeds the X_eTaL that X_eTaL-extensions pins (v0.1.0),
so the gate checks that every library program here prints the same
under it.

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

- [`docs/plan.md`](docs/plan.md) -- architecture decisions, the
  subset, the roadmap
- [`docs/xetal-asks.md`](docs/xetal-asks.md) -- what this repo needs
  from X_eTaL; the blockers first
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
