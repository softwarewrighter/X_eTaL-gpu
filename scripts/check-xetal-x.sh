#!/usr/bin/env bash
# Check the pinned xetal-x (X_eTaL-extensions' bridge host): it builds,
# reports the commit in XETAL_EXTENSIONS_COMMIT, runs hello's facade
# through the ext: bridge, and, since it embeds X_eTaL-extensions' own
# pinned X_eTaL rather than this repository's, gives every library
# test and demo here the same output as their baselines (made with
# this repository's xetal), so programs that use extensions mean what
# the baselines say.
#   scripts/check-xetal-x.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
xx="$("$root/scripts/xetal-extensions.sh")"
short="$(cut -c1-7 "$root/XETAL_EXTENSIONS_COMMIT")"
version="$("$xx" --version)"   # whole, not piped to head: pipefail would see its SIGPIPE
case "$version" in *"$short"*) ;; *) echo "check-xetal-x: xetal-x --version does not report $short" >&2; exit 1 ;; esac
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
printf '"hx:" u_se< "Hello"\nhx:a_nswer @\nhx:s_hout "gpu"\n' > "$tmp/hello.xtl"
got="$("$xx" --ext "$root/work/xetal-extensions/extensions/hello" run "$tmp/hello.xtl" | tr '\n' ' ')"
[ "$got" = "42 GPU " ] || { echo "check-xetal-x: hello gave '$got', expected '42 GPU '" >&2; exit 1; }
n=0
for lib in "$root"/libs/*/; do
  for p in "$lib"tests/*.xtl "$lib"demos/*.xtl; do
    [ -e "$p" ] || continue
    dir="$(dirname "$p")"; stem="$(basename "$p" .xtl)"
    case "$dir" in */demos) want="$lib/tests/demo-$stem.out" ;; *) want="$lib/tests/$stem.out" ;; esac
    (cd "$dir" && XETAL_PATH=../src "$xx" run --seed 1 --ascii "$stem.xtl" > "$tmp/out" 2>&1) || true
    cmp -s "$want" "$tmp/out" || { echo "check-xetal-x: ${p#$root/} differs under xetal-x:"; diff "$want" "$tmp/out" | head -10; exit 1; }
    n=$((n + 1))
  done
done
echo "check-xetal-x: ok ($short; hello through the bridge; $n library programs give their baselines under its X_eTaL)"
