#!/usr/bin/env bash
# The OpenCL evidence for one demo, the way modular-fork's CUDA and
# Metal captures are laid out (doc/evidence/<machine>-<topic>-<date>/):
# the lowered XIR, the OpenCL C, the plan, the run on the device
# checked against the evaluator's baseline, what the device's
# compiler hands back as the program binary (CL_PROGRAM_BINARIES)
# with its decoding, environment.log and build.log (every command as
# run). Usage:
#   scripts/opencl-evidence.sh DIR [--device opencl:N] [DEMO]
# DEMO is an Accel demo's name (threshold); DIR is created (it must
# not exist, so an old capture is never mixed with a new one).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
dir="${1:?usage: scripts/opencl-evidence.sh DIR [--device opencl:N] [DEMO]}"; shift
device="opencl:0"; demo="threshold"
while [ $# -gt 0 ]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    *) demo="$1"; shift ;;
  esac
done
[ -e "$dir" ] && { echo "opencl-evidence: $dir exists; name a new directory" >&2; exit 1; }
src="libs/Accel/demos/$demo.xtl"
baseline="libs/Accel/tests/demo-$demo.out"
[ -f "$src" ] || { echo "opencl-evidence: no demo $src" >&2; exit 1; }
[ -f "$baseline" ] || { echo "opencl-evidence: no baseline $baseline" >&2; exit 1; }
tool="target/release/xetal-gpu"
[ -x "$tool" ] || (cd components && cargo build -q --release -p xetal-gpu-cli)
export XETAL_PATH="$(scripts/libs.py path)"
mkdir -p "$dir"
log="$dir/build.log"
: > "$log"
# Run a command, recording it and its exit status in build.log.
run() {
  echo "\$ $*" >> "$log"
  "$@"
  echo "(exit $?)" >> "$log"
}
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(uname -s) $(uname -r) $(uname -m)"
  if command -v sw_vers >/dev/null; then sw_vers | tr '\n' ';' | sed 's/;$//'; echo; fi
  if [ -r /etc/os-release ]; then grep PRETTY_NAME /etc/os-release; fi
  echo "X_eTaL-gpu: $(git rev-parse --short HEAD) ($(git status --porcelain | wc -l | tr -d ' ') uncommitted paths)"
  echo "XETAL_COMMIT: $(cat XETAL_COMMIT)"
  echo "xetal-gpu: $("$tool" version)"
  echo "rustc: $(rustc --version)"
  echo "devices:"
  "$tool" devices | sed 's/^/  /'
  echo "device used: $device"
  echo "demo: $src"
  echo "baseline: $baseline"
} > "$dir/environment.log"
run "$tool" lower "$src" > "$dir/$demo.xir"
run "$tool" kernel "$src" > "$dir/$demo.cl"
run "$tool" explain "$src" > "$dir/plan.txt"
run "$tool" run "$src" --device "$device" --time > "$dir/run.out" 2> "$dir/run.log"
if cmp -s "$dir/run.out" "$baseline"; then
  echo "RESULT: the output is identical to $baseline" >> "$dir/run.log"
else
  echo "RESULT: the output differs from $baseline:" >> "$dir/run.log"
  diff "$baseline" "$dir/run.out" >> "$dir/run.log" || true
fi
run "$tool" binary "$src" --device "$device" --out "$dir/$demo.bin" > "$dir/binary.txt"
# Decode what came back: Apple wraps it in property lists (the inner
# one holds the compiler options and the source as lines), NVIDIA
# returns PTX text; anything else is left to the inspection above.
python3 - "$dir" "$demo" "$dir/$demo.cl" > "$dir/binary-decoded.txt" <<'PY'
import plistlib, sys, pathlib
d, demo, cl = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
raw = (d / f"{demo}.bin").read_bytes()
if raw.startswith(b"bplist00"):
    outer = plistlib.loads(raw)
    print("an Apple binary property list, keys and values:")
    for k, v in outer.items():
        print(f"  {k}: {('<' + str(len(v)) + ' bytes>') if isinstance(v, bytes) else v!r}")
    data = outer.get("clBinaryData", b"")
    if data.startswith(b"bplist00"):
        inner = plistlib.loads(data)
        print("clBinaryData is a second property list, keys and values:")
        for k, v in inner.items():
            if isinstance(v, list):
                print(f"  {k}: a list of {len(v)} strings (lines)")
            elif isinstance(v, bytes):
                print(f"  {k}: <{len(v)} bytes>")
            else:
                print(f"  {k}: {v!r}")
        src = inner.get("SOURCE")
        if isinstance(src, list):
            same = "\n".join(src) == cl.read_text()
            print(f"SOURCE joined with newlines is {'identical to' if same else 'NOT identical to'} {cl.name} (the OpenCL C handed to clCreateProgramWithSource)")
        others = [k for k in inner if k not in ("OPTIONS", "FLAGS", "SOURCE")]
        print(f"other keys (where compiled code would be): {others if others else 'none'}")
elif raw.startswith(b"//") and b".version" in raw and b".target" in raw:
    (d / f"{demo}.ptx").write_bytes(raw)
    print(f"PTX text, saved as {demo}.ptx; its first lines:")
    print("\n".join("  " + l for l in raw.decode("utf-8", "replace").splitlines()[:12]))
elif not raw:
    print("no bytes returned")
else:
    print("not decoded here (see binary.txt)")
PY
echo "\$ python3 (decode $demo.bin; see scripts/opencl-evidence.sh)" >> "$log"
echo "(exit 0)" >> "$log"
# The runtime's own binary tests: on a device, they print what it
# returns; a refused build's log can name the implementation (Apple's
# says cl2Metal).
echo "\$ (cd components && cargo test -q -p xetal-gpu-runtime binary_tests -- --nocapture) > runtime-tests.log 2>&1" >> "$log"
(cd components && cargo test -q -p xetal-gpu-runtime binary_tests -- --nocapture) > "$dir/runtime-tests.log" 2>&1 || true
echo "(exit $?)" >> "$log"
echo "opencl-evidence: written to $dir"
ls "$dir" | sed 's/^/  /'
