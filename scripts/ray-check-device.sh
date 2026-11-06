#!/usr/bin/env bash
# Lower the committed diagnostic from source, require no evaluator fallback,
# and compare every actual renderer output against its evaluator baseline.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
device="${1:-opencl:0}"
libs="$(scripts/xetal-libraries.sh)"
export XETAL_PATH
XETAL_PATH="$(python3 - "$libs" <<'PY'
from pathlib import Path
import sys
print(':'.join(str(p) for p in sorted(Path(sys.argv[1]).glob('libs/*/src')) if p.is_dir()))
PY
)"
(cd components && cargo build -q --release -p xetal-gpu-cli)
gpu="$root/target/release/xetal-gpu"
source="$libs/libs/RayTrace/demos/diagnostic.xtl"
out="$root/work/ray-integration"
mkdir -p "$out"
"$gpu" lower "$source" > "$out/diagnostic-source.xir" 2> "$out/diagnostic-source.err"
if [ -s "$out/diagnostic-source.err" ]; then
    cat "$out/diagnostic-source.err" >&2
    echo 'ray-check-device: diagnostic must lower without evaluator fallback' >&2
    exit 1
fi
[ "$(rg -c '^output ' "$out/diagnostic-source.xir")" = 3 ] || {
    echo 'ray-check-device: expected all three renderer outputs' >&2; exit 1;
}
"$gpu" run "$out/diagnostic-source.xir" --device cpu > "$out/diagnostic-source-cpu.out"
scripts/compare-out.py "$libs/libs/RayTrace/tests/demo-diagnostic.out" "$out/diagnostic-source-cpu.out" 0
"$gpu" run "$out/diagnostic-source.xir" --device "$device" > "$out/diagnostic-source-device.out"
scripts/compare-out.py "$libs/libs/RayTrace/tests/demo-diagnostic.out" "$out/diagnostic-source-device.out" 1e-5
python3 - "$libs/libs/RayTrace/tests/demo-diagnostic.out" "$out/diagnostic-source-device.out" <<'PY_IDS'
from pathlib import Path
import sys
expected, actual = (Path(p).read_text().splitlines() for p in sys.argv[1:])
assert [list(map(int, row.split())) for row in expected[:4]] == [list(map(int, row.split())) for row in actual[:4]], "triangle IDs must agree exactly"
print('ray-check-device: 16 Int triangle IDs agree exactly')
PY_IDS
echo "ray-check-device: actual triangle IDs, linear RGB and sRGB agree on cpu and $device"
