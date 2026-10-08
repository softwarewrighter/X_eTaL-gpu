#!/usr/bin/env bash
# Build the documentation site (xetal doc --out) into pages/doc, as
# X_eTaL's live demo has one: every library here (its ## doc comments,
# ### sections and ## >> examples, its source drawn decorated, every
# name linked to its definition and its uses), the example programs,
# the gpu extension's facade and its demo, and the libraries they
# import (X_eTaL-extensions' Ffi and Clock). It is built with the
# pinned xetal-x, the one host that can expand the facade's binding
# macro; its X_eTaL is X_eTaL-extensions' pin (v0.1.0), and the gate
# checks every library program prints the same under it. Paths are
# relative, so no page name holds the checkout's location.
#   scripts/doc-site.sh             # into pages/doc
#   XETAL_DOC_OUT=DIR scripts/doc-site.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
"$root/scripts/gpu-ext.sh" >/dev/null
out="${XETAL_DOC_OUT:-pages/doc}"
rm -rf "$out"
libs="$(ls -d libs/*/src | paste -sd: -)"
export XETAL_PATH="$libs:work/xetal-extensions/lib:work/xetal-extensions/extensions/clock/lib"
files=(extensions/gpu/lib/Gpu.xtl libs/*/src/*.xtl libs/*/demos/*.xtl extensions/gpu/demos/*.xtl)
bin/xetal-x --ext extensions/gpu --ext work/xetal-extensions/extensions/clock doc --out "$out" "${files[@]}" > /dev/null
if grep -rl "$HOME" "$out" >/dev/null 2>&1; then echo "doc-site: a page names $HOME (a path that is not relative)" >&2; exit 1; fi
echo "doc: $(find "$out" -name '*.html' | wc -l | tr -d ' ') pages in $out"
