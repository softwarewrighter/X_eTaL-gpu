# Apple OpenCL evidence for the GPU deep dive

Prepared on the Mac (Apple M1 Max), 2026-10-21, for the blog agent's
ask X1: the threshold demo through Apple's OpenCL on the M1 Max,
checked against the evaluator's baseline, and what Apple's
`clGetProgramInfo(CL_PROGRAM_BINARIES)` returns for it. Everything
below came from `scripts/opencl-evidence.sh` (`just opencl-evidence
DIR`), which writes the directory it links.

## Text the post can use

Through OpenCL, the threshold program is one fused kernel of OpenCL
C 1.2 (`k0`, one work-item per element, five outputs), handed to
Apple's driver as source and built by it. On the M1 Max the run gives
all five of X_eTaL's expected output lines, identical to the
evaluator's baseline, in 0.3 ms of launch time for eight elements
(the setup, about 56 ms, is the driver compiling the source).

Asked for the program binary, Apple's OpenCL reports
`CL_PROGRAM_BINARY_TYPE` as `executable` and returns 2,108 bytes.
Those bytes are not GPU code and not Apple's AIR: they are a binary
property list (`bplist00`) with four keys, `clBinaryVersion` (2),
`clPlatformVersion` (`OpenCL 1.2 (Apr 18 2026 19:20:04)`),
`clBinaryDriver` (`1.2 1.0`) and `clBinaryData`, and `clBinaryData`
is a second property list holding `OPTIONS` (the driver's own
defines, `-D__OPENCL_VERSION__=120` and the `cl_khr_*` extensions it
enables, and an include path), `FLAGS` (0) and `SOURCE`, the OpenCL
C as a list of 27 lines, identical to the source the tool handed to
`clCreateProgramWithSource`. No other key exists, so a program
"binary" loaded back with `clCreateProgramWithBinary` on this
platform is compiled again from source. The compiled kernel that
runs on the M1 Max never leaves the driver: there is no public
listing of it, the same gap the Metal handoff names for Mojo's AIR.
What can be said about the implementation comes from its build log:
when a source is refused, Apple's driver logs
`buildComputeProgram: cl2Metal failed`, naming the translator from
OpenCL C to Metal that it runs inside.

## Evidence to link from the draft

All paths are relative to this document. Everything but
`threshold.bin` is text.

| Artifact | What it establishes |
| --- | --- |
| [Lowered XIR](evidence/m1max-opencl-20261021/threshold.xir) | The program this tool runs, from `libs/Accel/demos/threshold.xtl` (X_eTaL-gpu `8d5b943`, X_eTaL `27b3fd0`) |
| [OpenCL C](evidence/m1max-opencl-20261021/threshold.cl) | The generated source, one fused kernel; the `SOURCE` lines in the returned binary are exactly these |
| [The plan](evidence/m1max-opencl-20261021/plan.txt) | Buffers, the kernel, the launch (8 work-items in one group) and which X_eTaL line each came from |
| [The run](evidence/m1max-opencl-20261021/run.out) | The five outputs from the M1 Max, byte for byte the evaluator's `libs/Accel/tests/demo-threshold.out` |
| [Run log](evidence/m1max-opencl-20261021/run.log) | The timing line and the comparison's result |
| [The returned binary](evidence/m1max-opencl-20261021/threshold.bin) | The 2,108 bytes `CL_PROGRAM_BINARIES` returned, as returned |
| [The tool's report](evidence/m1max-opencl-20261021/binary.txt) | `xetal-gpu binary`: the device, the driver, the binary type, the size, the first bytes and every readable string inside |
| [The decoding](evidence/m1max-opencl-20261021/binary-decoded.txt) | Both property lists' keys and values; `SOURCE` checked identical to the OpenCL C; no key where compiled code would be |
| [Runtime tests](evidence/m1max-opencl-20261021/runtime-tests.log) | The runtime's own binary tests on the device, with the `cl2Metal failed` line a refused build logs |
| [Environment](evidence/m1max-opencl-20261021/environment.log) | macOS 26.5, Apple M1 Max, OpenCL 1.2, the commits, rustc |
| [Build commands](evidence/m1max-opencl-20261021/build.log) | Every command as run, with its exit status |

## What this does and does not establish

It establishes that the OpenCL route runs the program correctly on
Apple silicon, and that Apple's OpenCL keeps no compiled code a
program can read back: its program binary is the source and the
options, re-compiled on load. It does not show AIR, Metal shader
source or GPU machine code for the kernel; none is claimed, and
Apple's OpenCL offers no query that returns one. No speed claim is
made from an eight-element program.

## The same query on NVIDIA (ask A7, the Arch lane)

`xetal-gpu binary FILE.xtl --device opencl:N --out FILE` is the same
on every platform, and `scripts/opencl-evidence.sh DIR` captures the
same set of files; NVIDIA's OpenCL returns PTX text, which the script
recognizes and saves beside the binary as `threshold.ptx`.
`docs/arch-rerun.md` item 6 tells the Arch lane what to run on the
RTX 5060 Ti; its capture goes to `docs/evidence/rtx5060-opencl-<date>/`
with its own handoff note, so the post can set the driver's PTX
beside Mojo's and nvcc's.

## Reproduction

```sh
just build
just kernel-binary libs/Accel/demos/threshold.xtl        # the report alone
just opencl-evidence docs/evidence/m1max-opencl-20261021 # the whole directory (must not exist)
```

The subcommand builds the plan's kernels on the device exactly as
`run` does (the same empty option string) and queries the program
back; nothing is uploaded and nothing runs. Without an OpenCL device
it fails by name, and the runtime's device test says it was skipped.
