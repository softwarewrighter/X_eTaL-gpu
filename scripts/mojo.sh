#!/usr/bin/env bash
# The Mojo toolchain this repository's Mojo backend (components/mojo) builds
# kernels with: the MAX nightly pinned in MOJO_VERSION (the `max` wheel
# carries the Mojo compiler, its standard library and the `max.gpu` package
# the generated kernels import; the `mojo` wheel alone lacks `max.gpu`),
# installed with uv into the gitignored work/mojo/.venv and linked as
# bin/mojo. The version is the one ../modular-fork/rust-demo proved on the
# M1 Max (Metal) and the RTX 5060 Ti (CUDA). Prints the mojo path on
# success; the gate's Mojo tests skip, saying so, when it is absent.
#
#   scripts/mojo.sh            # install or confirm; prints bin/mojo
#   scripts/mojo.sh --check    # only say whether it is there (exit 1 if not)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
version="$(tr -d '[:space:]' < MOJO_VERSION)"
venv="$root/work/mojo/.venv"
if [ "${1:-}" = --check ]; then
  if [ -x "$venv/bin/mojo" ]; then echo "mojo: $("$venv/bin/mojo" --version 2>/dev/null | grep -v -i crashpad | head -1)"; exit 0; fi
  echo "mojo: not installed (scripts/mojo.sh installs MAX $version into work/mojo/.venv)"; exit 1
fi
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
mkdir -p "$root/work/mojo" "$root/bin"
[ -x "$venv/bin/python" ] || uv venv "$venv" --python "${PYTHON_VERSION:-3.12}" >/dev/null
if ! "$venv/bin/python" -c "import importlib.metadata as m; v=m.version('max'); import sys; sys.exit(0 if v=='$version' else 1)" 2>/dev/null; then
  uv pip install --python "$venv/bin/python" "max==$version" --index https://whl.modular.com/nightly/simple/ --prerelease allow >/dev/null
fi
ln -sfn "$venv/bin/mojo" "$root/bin/mojo"
case "$(uname -s)" in
  Darwin)
    xcrun -sdk macosx metal --version >/dev/null 2>&1 || echo "WARNING: the Metal toolchain is missing; run: xcodebuild -downloadComponent MetalToolchain" >&2 ;;
  Linux)
    command -v nvidia-smi >/dev/null || echo "WARNING: no nvidia-smi; the kernels need a GPU and its driver" >&2 ;;
esac
echo "mojo: $("$venv/bin/mojo" --version 2>/dev/null | grep -v -i crashpad | head -1) ($root/bin/mojo)"
