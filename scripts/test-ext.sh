#!/usr/bin/env bash
# Test the gpu extension's facade with reg-rs (extensions/gpu/tests is
# the data directory): types.rgt pins the facade's types; NAME.rgt runs
# each tests/NAME.xtl with scripts/xx
# (xetal-x, the package, the libraries). A program named *-device.xtl
# needs an OpenCL device: without one it is skipped, and the script
# says so. A FAIL line in a baseline's output fails the test.
# XETAL_BLESS=1 creates missing baselines and rebases the rest (review the diff!).
#   scripts/test-ext.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v reg-rs >/dev/null || { echo "test-ext: reg-rs not found on PATH" >&2; exit 127; }
"$root/scripts/gpu-ext.sh" >/dev/null
d="$root/extensions/gpu/tests"
has_device=1
found="$("$root/target/release/xetal-gpu" devices 2>/dev/null || true)"
grep -q '^opencl:' <<<"$found" || has_device=0
xx="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$root/scripts/xx" "$d")"
wanted=("types|$xx type ../lib/Gpu.xtl|")
for p in "$d"/*.xtl; do wanted+=("$(basename "$p" .xtl)|$xx run $(basename "$p")|$p"); done
export REG_RS_DATA_DIR="$d"
cd "$d"
fail=0; skipped=0
for w in "${wanted[@]}"; do
  IFS='|' read -r t cmd prog <<<"$w"
  if [ "$has_device" = 0 ] && [[ "$t" == *-device || "$t" == demo-* ]]; then skipped=$((skipped + 1)); continue; fi
  if [ ! -f "$t.rgt" ]; then
    if [ "${XETAL_BLESS:-}" = 1 ]; then rm -f "$t".tdb*; reg-rs create -t "$t" -c "$cmd" >/dev/null; echo "created: gpu/$t"
    else echo "FAIL: gpu/$t: no baseline $t.rgt (XETAL_BLESS=1 to create)"; fail=1; continue; fi
  fi
  if [ "${XETAL_BLESS:-}" = 1 ]; then reg-rs run -q -p "$t" >/dev/null 2>&1 || true; reg-rs rebase -p "$t" >/dev/null 2>&1 || true
  elif ! reg-rs run -q -p "$t" >/dev/null 2>&1; then echo "FAIL: gpu/$t"; reg-rs run -vv -p "$t" || true; fail=1; fi
  if [ -f "$t.out" ] && grep -q '^FAIL' "$t.out" && { [ -z "$prog" ] || ! grep -q '# shows failures' "$prog"; }; then
    echo "FAIL: gpu/$t: a check failed:"; grep '^FAIL' "$t.out"; fail=1
  fi
done
[ "$skipped" = 0 ] || echo "test-ext: no OpenCL device here; $skipped device test(s) skipped"
# The demos print timings, so they have no baselines: with XETAL_DEMOS=1
# each must run to the end with no FAIL line (the offload demo takes
# about 40 s, most of it the evaluator's 512 by 512 product).
if [ "${XETAL_DEMOS:-}" = 1 ] && [ "$has_device" = 1 ]; then
  for p in "$root"/extensions/gpu/demos/*.xtl; do
    [ -e "$p" ] || continue
    if out="$("$root/scripts/xx" run "$p" 2>&1)" && ! grep -q '^FAIL' <<<"$out"; then echo "ok: demo $(basename "$p")"
    else echo "FAIL: demo $(basename "$p"):"; echo "$out" | tail -5; fail=1; fi
  done
else
  echo "test-ext: demos not run (XETAL_DEMOS=1 runs them)"
fi
echo "test-ext: gpu, ${#wanted[@]} baselines$([ $fail = 0 ] && echo ', all passed' || echo ', FAILURES')"
exit $fail
