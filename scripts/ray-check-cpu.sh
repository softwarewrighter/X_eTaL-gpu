#!/usr/bin/env bash
# Check committed RayTrace/Vector3 CPU examples against their pinned baselines.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
libs="$(scripts/xetal-libraries.sh)"
xetal="$(scripts/xetal.sh)"
XETAL_PATH="$(python3 - "$libs" <<'PY'
from pathlib import Path
import sys
print(':'.join(str(p) for p in sorted(Path(sys.argv[1]).glob('libs/*/src')) if p.is_dir()))
PY
)"
export XETAL_PATH
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
n=0
for spec in Vector3:tests:basics Vector3:demos:pivot RayTrace:tests:camera RayTrace:tests:intersections RayTrace:tests:lighting RayTrace:tests:pixels RayTrace:demos:diagnostic Extrusion:tests:glyphs Scene:tests:arrays Scene:tests:animation Scene:tests:default; do
    n=$((n + 1))
    IFS=: read -r lib kind name <<<"$spec"
    stem="$name"
    [ "$kind" != demos ] || stem="demo-$name"
    "$xetal" run --seed 1 --ascii "$libs/libs/$lib/$kind/$name.xtl" > "$tmp/got.out"
    result="$(scripts/compare-out.py "$libs/libs/$lib/tests/$stem.out" "$tmp/got.out" 0)"
    echo "ray-check-cpu: $lib/$name: $result"
done
echo "ray-check-cpu: $n committed examples pass with X_eTaL $(cut -c1-7 XETAL_COMMIT) and libraries $(cut -c1-7 XETAL_LIBRARIES_COMMIT)"
