#!/usr/bin/env bash
# Check the published site (after scripts/publish-pages.sh; GitHub
# Pages takes a minute or so to serve a new gh-pages commit): the
# landing page names this commit, the docs' index and the reference
# pages load, and those pages show the libraries' doc comments.
#   scripts/check-live.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
url="${XETAL_PAGES_URL:-https://softwarewrighter.github.io/X_eTaL-gpu}"
head="$(git -C "$root" rev-parse --short HEAD)"
get() { curl -sfL "$url/$1"; }
fail=0
check() { # check PATH TEXT
  # The page read whole first: grep -q in a pipe would stop early and,
  # under pipefail, report curl's broken pipe as a failure.
  page="$(get "$1" || true)"
  if grep -qF -- "$2" <<<"$page"; then echo "ok: live $1"; else echo "FAIL: live $1 does not show: $2"; fail=1; fi
}
check "" "X_eTaL-gpu $head"
check "doc/index.html" "Accel"
check "doc/libs-Accel-src-Accel.xtl.html" "The sum, item by item"
check "doc/extensions-gpu-lib-Gpu.xtl.html" "Load an XIR program"
check "doc/extensions-gpu-demos-offload.xtl.html" "offload"
[ "$fail" = 0 ] && echo "check-live: ok ($url, main $head)"
exit "$fail"
