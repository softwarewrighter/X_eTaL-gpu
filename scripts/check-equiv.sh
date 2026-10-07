#!/usr/bin/env bash
# The equivalence checks: every hand-lowered twin (libs/*/demos/*.xir)
# run by xetal-gpu must give what the pinned X_eTaL evaluator printed
# for its .xtl (the demo's reg-rs baseline, libs/<Name>/tests/demo-<stem>.out):
#   - on the reference interpreter (--device cpu): exactly;
#   - on each OpenCL device found (when the runtime exists and a
#     device is present): Floats within 1e-5, Ints exactly.
# A demo program without a twin fails, unless it says "# no twin".
#   scripts/check-equiv.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
(cd components && cargo build -q --release -p xetal-gpu-cli)
gpu="$root/target/release/xetal-gpu"
fail=0; n=0
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
# The OpenCL devices present (none is not a failure: it is said).
devices=()
while IFS= read -r d; do [ -n "$d" ] && devices+=("$d"); done < <("$gpu" devices | grep -o '^opencl:[0-9]*' || true)
[ ${#devices[@]} -gt 0 ] || echo "check-equiv: no OpenCL device here; the GPU side is skipped"
for xtl in libs/*/demos/*.xtl; do
  name="$(basename "$xtl" .xtl)"; lib="$(basename "$(dirname "$(dirname "$xtl")")")"
  xir="${xtl%.xtl}.xir"; want="libs/$lib/tests/demo-$name.out"
  if [ ! -f "$xir" ]; then
    grep -q '# no twin' "$xtl" && continue
    echo "FAIL: $xtl has no twin $xir (or say '# no twin')"; fail=1; continue
  fi
  [ -f "$want" ] || { echo "FAIL: $xir: no baseline $want (just bless-lib $lib)"; fail=1; continue; }
  n=$((n + 1))
  if ! "$gpu" run "$xir" --device cpu > "$tmp/cpu.out" 2> "$tmp/cpu.err"; then
    echo "FAIL: $xir on cpu:"; cat "$tmp/cpu.err"; fail=1; continue
  fi
  if out="$(scripts/compare-out.py "$want" "$tmp/cpu.out" 0)"; then echo "ok: $lib/$name on cpu, $out"
  else echo "FAIL: $lib/$name on cpu: $out"; fail=1; fi
  # Each device with the default schedule and with products in 8 by 8 tiles.
  for dev in "${devices[@]}"; do for sched in "" "--tile 8"; do
    label="$dev${sched:+ $sched}"
    if ! "$gpu" run "$xir" --device "$dev" $sched > "$tmp/gpu.out" 2> "$tmp/gpu.err"; then
      echo "FAIL: $xir on $label:"; cat "$tmp/gpu.err"; fail=1; continue
    fi
    if out="$(scripts/compare-out.py "$want" "$tmp/gpu.out" 1e-5)"; then echo "ok: $lib/$name on $label, $out"
    else echo "FAIL: $lib/$name on $label: $out"; fail=1; fi
  done; done
done
echo "check-equiv: $n twin$([ $n = 1 ] || echo s) on cpu and ${#devices[@]} OpenCL device$([ ${#devices[@]} = 1 ] || echo s)$([ $fail = 0 ] && echo ', all agree with the evaluator' || echo ', FAILURES')"
exit $fail
