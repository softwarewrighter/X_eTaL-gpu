#!/usr/bin/env bash
# Run every library's ## >> examples (xetal doc --test): this
# repository's libraries with the pinned xetal (scripts/xt), the gpu
# extension's facade with xetal-x (scripts/xx). An example whose
# output differs from the ## lines under it fails.
#   scripts/doc-test.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
fail=0
for f in libs/*/src/*.xtl; do
  out="$(scripts/xt doc --test "$f" 2>&1)" || { echo "$out"; fail=1; }
  echo "$out" | tail -1 | sed "s#^#$f: #"
done
out="$(scripts/xx doc --test extensions/gpu/lib/Gpu.xtl 2>&1)" || { echo "$out"; fail=1; }
echo "$out" | tail -1 | sed "s#^#extensions/gpu/lib/Gpu.xtl: #"
exit $fail
