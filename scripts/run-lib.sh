#!/usr/bin/env bash
# Run a library's test programs (or, with --demos, its demos) as the
# tests do: from that directory, every libs/*/src on XETAL_PATH, --seed
# 1; all of them, or the one named. --echo shows each statement before
# its output (a notebook).
#   scripts/run-lib.sh [--echo] [--demos] Name [program]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo=(); dir=tests
while [ $# -gt 0 ]; do case "$1" in
  --echo) echo=(--echo); shift ;; --demos) dir=demos; shift ;; *) break ;; esac; done
name="${1:?usage: run-lib.sh [--echo] [--demos] Name [program]}"
d="$root/libs/$name/$dir"
[ -d "$d" ] || { echo "run: no libs/$name/$dir" >&2; exit 1; }
if [ -n "${2:-}" ]; then progs=("${2%.xtl}.xtl"); else progs=(); for p in "$d"/*.xtl; do progs+=("$(basename "$p")"); done; fi
for p in "${progs[@]}"; do
  [ ${#progs[@]} -gt 1 ] && echo "== $name/$dir/$p"
  (cd "$d" && "$root/scripts/xt" run ${echo[@]+"${echo[@]}"} "$p")
done
