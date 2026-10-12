#!/usr/bin/env bash
# Lint every X_eTaL file of this repository with the pinned xetal's
# linter (X_eTaL Saga 40): each library's source, tests and demos, and
# the gpu extension's facade, tests and demos. The lints that warn by
# default are denied here, so a finding fails the gate; a file allows
# one with a `# lint: allow NAME` comment saying why.
#   scripts/lint-libs.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
xetal="$(scripts/xetal.sh)"
scripts/xetal-extensions.sh >/dev/null   # Ffi.xtlm and Clock.xtl, which the facade imports
libs="$(scripts/libs.py path)"
export XETAL_PATH="$libs:$root/extensions/gpu/lib:$root/work/xetal-extensions/lib:$root/work/xetal-extensions/extensions/clock/lib"
deny=(-D correctness -D suspicious -D complexity -D style -D deprecated)
fail=0; n=0
for f in libs/*/src/*.xtl libs/*/src/*.xtlm libs/*/tests/*.xtl libs/*/demos/*.xtl \
         extensions/gpu/lib/*.xtl extensions/gpu/tests/*.xtl extensions/gpu/demos/*.xtl; do
  [ -e "$f" ] || continue
  n=$((n + 1))
  allow=()
  # The facade's ffi:b_ind< lines expand (X_eTaL-extensions' Ffi.xtlm, pinned at
  # 7bdd827) to bindings the macro never reads: its finding, not this file's (ask X-3).
  [ "$f" = extensions/gpu/lib/Gpu.xtl ] && allow=(-A unused-binding)
  if ! out="$(cd "$(dirname "$f")" && "$xetal" lint "${deny[@]}" ${allow[@]+"${allow[@]}"} "$(basename "$f")" 2>&1)"; then
    echo "FAIL: $f:"; echo "$out" | sed 's/^/  /'; fail=1
  fi
done
echo "lint: $n file$([ $n = 1 ] || echo s)$([ $fail = 0 ] && echo ', no findings' || echo ', FAILURES')"
exit $fail
