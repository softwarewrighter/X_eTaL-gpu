#!/usr/bin/env bash
# Start a library from templates/Library: libs/Name/ with its README,
# src/Name.xtl, docs/README.md, a demo and a test, and its reg-rs
# baselines created.
#   scripts/new-lib.sh Name alias: "what it is"
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="${XETAL_LIBS_ROOT:-$root}"
usage='usage: new-lib.sh Name alias: "what it is"'
name="${1:?$usage}"; alias="${2:?$usage}"; summary="${3:?$usage}"
[[ "$name" =~ ^[A-Z][A-Za-z0-9]*$ ]] || { echo "new-lib: a library name is UpperCamel" >&2; exit 1; }
[[ "$alias" =~ ^[a-z]+:$ ]] || { echo "new-lib: an alias is lowercase letters and a colon (t:)" >&2; exit 1; }
[ ! -e "$base/libs/$name" ] || { echo "new-lib: libs/$name exists" >&2; exit 1; }
[ ! -e "$root/work/xetal/lib/$name.xtl" ] || { echo "new-lib: $name is a standard library" >&2; exit 1; }
mkdir -p "$base/libs"
cp -R "$root/templates/Library" "$base/libs/$name"
mv "$base/libs/$name/src/__NAME__.xtl" "$base/libs/$name/src/$name.xtl"
esc="$(printf '%s' "$summary" | sed 's/[&|\\]/\\&/g')"
find "$base/libs/$name" -type f | while read -r f; do
  sed -i.bak -e "s|__NAME__|$name|g" -e "s|__ALIAS__|$alias|g" -e "s|__SUMMARY__|$esc|g" "$f" && rm -f "$f.bak"
done
chmod +x "$base/libs/$name"/tests/*.xtl "$base/libs/$name"/demos/*.xtl
XETAL_BLESS=1 "$root/scripts/test-libs.sh" "$name" >/dev/null
echo "new library: libs/$name/ (src, docs, demos, tests; just test-lib $name)"
