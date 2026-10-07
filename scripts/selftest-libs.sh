#!/usr/bin/env bash
# Test the library tooling itself in a scratch root: new-lib makes a
# library that passes; a wrong baseline fails and blessing repairs it;
# an unexpected error fails until blessed; a changed export type fails;
# a new test program without a baseline fails; a baseline whose
# program is gone fails; a FAIL line fails unless the program says it
# shows failures; an undocumented export, a missing demo and a missing
# import line fail; a library named like a standard one is refused.
#   scripts/selftest-libs.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XETAL_LIBS_ROOT="$(mktemp -d)"; export XETAL_LIBS_ROOT
trap 'rm -rf "$XETAL_LIBS_ROOT"' EXIT
t="$root/scripts/test-libs.sh"
d="$XETAL_LIBS_ROOT/libs/Probe"
expect() { # expect pass|fail DESCRIPTION
  if "$t" Probe >/dev/null 2>&1; then got=pass; else got=fail; fi
  [ "$got" = "$1" ] || { echo "selftest: $2: expected $1, got $got" >&2; "$t" Probe || true; exit 1; }
}
bless() { XETAL_BLESS=1 "$t" Probe >/dev/null 2>&1 || true; }
"$root/scripts/new-lib.sh" Probe pr: "A probe & co" >/dev/null
grep -q '^# Probe: A probe & co$' "$d/src/Probe.xtl"
[ -f "$d/tests/basics.rgt" ] && [ -f "$d/tests/types.rgt" ] && [ -f "$d/tests/demo-example.rgt" ]
expect pass "a fresh library"
echo 56 > "$d/tests/basics.out"
expect fail "a wrong baseline"
bless; expect pass "after blessing"
printf '"pr:" u_se< "Probe"\n1 +\n' > "$d/tests/basics.xtl"
expect fail "an unexpected error"
bless; [ -s "$d/tests/basics.err" ]; expect pass "an expected error, blessed"
sed -i.bak 's/{ x -> x }/{ x -> x + 1 }/' "$d/src/Probe.xtl" && rm "$d/src/Probe.xtl.bak"
expect fail "a changed export type"
bless; expect pass "the new type, blessed"
printf '"pr:" u_se< "Probe"\npr:i_dentity 7\n' > "$d/tests/more.xtl"
expect fail "a test program without a baseline"
bless; expect pass "its baseline created"
rm "$d/tests/more.xtl"
expect fail "a baseline whose program is gone"
rm "$d/tests"/more.*; expect pass "the stale baseline removed"
printf '"pr:" u_se< "Probe"\n"FAIL: x"\n' > "$d/tests/basics.xtl"
bless; expect fail "a failed check, even blessed"
printf '"pr:" u_se< "Probe"\n"FAIL: x"   # shows failures\n' > "$d/tests/basics.xtl"
bless; expect pass "a failed check the program says it shows"
printf 'l:t_wice := { x -> x * 2 }\n' >> "$d/src/Probe.xtl"
bless; expect fail "an undocumented export"
printf '\n`pr:t_wice`\n' >> "$d/docs/README.md"
expect pass "once documented"
mv "$d/demos/example.xtl" "$d/example.xtl.away"
expect fail "no demo"
mv "$d/example.xtl.away" "$d/demos/example.xtl"
sed -i.bak '/u_se</d' "$d/src/Probe.xtl" && rm "$d/src/Probe.xtl.bak"
expect fail "no import line in the header"
if "$root/scripts/new-lib.sh" Stats st: "x" >/dev/null 2>&1; then
  echo "selftest: new-lib made a library named like a standard one" >&2; exit 1
fi
echo "selftest-libs: ok"
