#!/usr/bin/env bash
# Check the pinned X_eTaL: the CLI builds, answers, reports the commit
# in XETAL_COMMIT, runs a demo, imports a standard library and dumps
# the Core IR (the form the accelerator lowering will start from).
#   scripts/check-xetal.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
xetal="$("$root/scripts/xetal.sh")"
short="$(cut -c1-7 "$root/XETAL_COMMIT")"
got="$("$xetal" eval -e "'+ r_/_2 2 3 r_eshape r_ange 6")"
[ "$got" = "6 15" ] || { echo "check-xetal: eval gave '$got', expected '6 15'" >&2; exit 1; }
version="$("$xetal" --version)"
grep -q "$short" <<<"$version" || { echo "check-xetal: xetal --version does not report $short" >&2; "$xetal" --version >&2; exit 1; }
"$xetal" run "$root/work/xetal/demos/life.xtl" >/dev/null
got="$("$xetal" eval -e '"s:" u_se< "Stats"
s:m_ean 1 2 3 4')"
[ "$got" = "2.5" ] || { echo "check-xetal: Stats gave '$got', expected '2.5'" >&2; exit 1; }
got="$("$xetal" core -e 'a := 1 2 3 4
2 * a' | tail -1)"
[ "$got" = "(eval (app2 #* 2 a))" ] || { echo "check-xetal: core gave '$got'" >&2; exit 1; }
echo "check-xetal: ok ($short)"
