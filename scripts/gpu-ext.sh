#!/usr/bin/env bash
# Build the gpu extension (extensions/gpu/rust, its own Cargo
# workspace) beside the pinned xetal-x in target/xetal-extensions/release/,
# where xetal-x finds native libraries; with --check, also its format,
# clippy (warnings are errors) and tests, as the gate runs them.
#   scripts/gpu-ext.sh [--check]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$root/scripts/xetal-extensions.sh" >/dev/null       # the SDK it builds on
export CARGO_TARGET_DIR="$root/target/xetal-extensions"
cd "$root/extensions/gpu/rust"
if [ "${1:-}" = --check ]; then
  cargo fmt --check || { echo "FAIL: gpu extension format (cd extensions/gpu/rust && cargo fmt)"; exit 1; }
  cargo clippy -q --all-targets -- -D warnings || { echo "FAIL: gpu extension clippy"; exit 1; }
  cargo test -q >/dev/null 2>&1 || { cargo test; echo "FAIL: gpu extension tests"; exit 1; }
fi
cargo build -q --release
echo "gpu extension: $CARGO_TARGET_DIR/release/libxetal_ext_gpu.$([ "$(uname)" = Darwin ] && echo dylib || echo so)"
