#!/usr/bin/env bash
# The pre-commit gate: the pinned X_eTaL (scripts/check-xetal.sh; it
# clones and builds on a fresh checkout), the library tooling's
# self-test, every library's baselines, the Rust components (format,
# clippy, tests), the equivalence checks (CPU evaluator against the
# XIR interpreter and, when a device is present, the GPU), American
# spellings, then ASCII-only markdown for the docs we own.
#   scripts/gate.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
"$root/scripts/check-xetal.sh"
"$root/scripts/selftest-libs.sh"
"$root/scripts/test-libs.sh"
if [ -f "$root/components/Cargo.toml" ]; then
  (cd "$root/components" && cargo fmt --all --check) || { echo "FAIL: cargo fmt (run: cd components && cargo fmt --all)"; exit 1; }
  (cd "$root/components" && cargo clippy -q --all-targets -- -D warnings) || { echo "FAIL: cargo clippy"; exit 1; }
  (cd "$root/components" && cargo test -q >/dev/null 2>&1) || { (cd "$root/components" && cargo test); echo "FAIL: cargo test"; exit 1; }
  echo "ok: components (fmt, clippy, tests)"
fi
[ -x "$root/scripts/check-equiv.sh" ] && "$root/scripts/check-equiv.sh"
# American spellings only (the checker checks itself first).
"$root/scripts/check-spelling.py" --self-test
"$root/scripts/check-spelling.py"
md=(README.md CHANGES.md docs/plan.md docs/xetal-asks.md)
for f in libs/*/README.md libs/*/docs/README.md components/*/README.md; do [ -e "$f" ] && md+=("$f"); done
for f in "${md[@]}"; do sw-markdown-checker -f "$f" >/dev/null || { sw-markdown-checker -f "$f"; exit 1; }; done
echo "gate: ok"
