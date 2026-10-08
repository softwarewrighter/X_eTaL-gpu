#!/usr/bin/env bash
# Build the documentation site (xetal doc --out) into pages/doc, as
# X_eTaL's live demo has one: every library here (its ## doc comments,
# ### sections and ## >> examples, its source drawn decorated, every
# name linked to its definition and its uses), the example programs,
# the gpu extension's facade and its demo, and the libraries they
# import (X_eTaL-extensions' Ffi and Clock). It is built with this
# repository's pinned xetal, whose doc tool is the newest (the sidebar
# groups files by directory): documenting only expands the facade's
# binding macro (Ffi.xtlm, on XETAL_PATH) and never calls the bridge,
# so xetal-x is not needed. Paths are relative, so no page name holds
# the checkout's location.
#   scripts/doc-site.sh             # into pages/doc
#   XETAL_DOC_OUT=DIR scripts/doc-site.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
"$root/scripts/xetal-extensions.sh" >/dev/null   # Ffi.xtlm and Clock.xtl, in its clone
out="${XETAL_DOC_OUT:-pages/doc}"
rm -rf "$out"
libs="$(ls -d libs/*/src | paste -sd: -)"
export XETAL_PATH="$libs:extensions/gpu/lib:work/xetal-extensions/lib:work/xetal-extensions/extensions/clock/lib"
files=(extensions/gpu/lib/Gpu.xtl libs/*/src/*.xtl libs/*/demos/*.xtl extensions/gpu/demos/*.xtl)
"$root/scripts/xetal.sh" >/dev/null
bin/xetal doc --out "$out" "${files[@]}" > /dev/null
if grep -rl "$HOME" "$out" >/dev/null 2>&1; then echo "doc-site: a page names $HOME (a path that is not relative)" >&2; exit 1; fi
echo "doc: $(find "$out" -name '*.html' | wc -l | tr -d ' ') pages in $out"
