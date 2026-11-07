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
check "" "Ray tracing"
check "ray-tracing/" "recorded CPU animation"
check "ray-tracing/" "native CPU still"
check "ray-tracing/" "RTX GPU still"
check "ray-tracing/" "media/logo/logo.webm"
check "doc/index.html" "Accel"
check "doc/libs-Accel-src-Accel.xtl.html" "The sum, item by item"
check "doc/extensions-gpu-lib-Gpu.xtl.html" "Load an XIR program"
check "doc/extensions-gpu-demos-offload.xtl.html" "offload"
check_binary() {
  remote_file="$(mktemp)"
  if curl -sfL "$url/$1" -o "$remote_file" && python3 - "$remote_file" "$root/$2" <<'PY_HASH'
from pathlib import Path
import hashlib
import sys
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).digest() == hashlib.sha256(Path(sys.argv[2]).read_bytes()).digest()
PY_HASH
  then echo "ok: live $1 (verified bytes)"; else echo "FAIL: live $1 does not match built media"; fail=1; fi
  rm -f "$remote_file"
}
check_binary "ray-tracing/media/logo-gpu/still.png" "site-assets/ray/logo-gpu/still.png"
check_binary "ray-tracing/media/logo-gpu/still.tif" "site-assets/ray/logo-gpu/still.tif"
check_binary "ray-tracing/media/logo-gpu/manifest.json" "site-assets/ray/logo-gpu/manifest.json"
check_binary "ray-tracing/media/logo-gpu/masters.zip" "site-assets/ray/logo-gpu/masters.zip"
check_binary "ray-tracing/media/logo-still/still.png" "site-assets/ray/logo-still/still.png"
check_binary "ray-tracing/media/logo-still/still.tif" "site-assets/ray/logo-still/still.tif"
check_binary "ray-tracing/media/logo-still/manifest.json" "site-assets/ray/logo-still/manifest.json"
check_binary "ray-tracing/media/logo-still/masters.zip" "site-assets/ray/logo-still/masters.zip"
check_binary "ray-tracing/media/logo/poster.png" "site-assets/ray/logo/poster.png"
check_binary "ray-tracing/media/logo/logo.webm" "site-assets/ray/logo/logo.webm"
check_binary "ray-tracing/media/logo/masters.zip" "site-assets/ray/logo/masters.zip"
check_binary "ray-tracing/media/logo/manifest.json" "site-assets/ray/logo/manifest.json"
[ "$fail" = 0 ] && echo "check-live: ok ($url, main $head)"
exit "$fail"
