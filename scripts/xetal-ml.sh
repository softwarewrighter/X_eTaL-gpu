#!/usr/bin/env bash
# Get X_eTaL-ML's programs, weights and expected outputs (its models
# and benchmarks, which this repository runs on GPUs: its ask GA1), as
# scripts/xetal.sh gets X_eTaL: clone X_eTaL-ML into work/xetal-ml
# (gitignored) and check out the known-good commit in XETAL_ML_COMMIT.
# Nothing is built: its .xtl programs run with this repository's
# pinned xetal, its libs/*/src on XETAL_PATH. Prints the clone's path.
#   scripts/xetal-ml.sh
#   scripts/xetal-ml.sh --pin [REF]   # pin a committed ref of ../X_eTaL-ML (default HEAD)
#   XETAL_ML_SOURCE=../X_eTaL-ML scripts/xetal-ml.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
clone="$root/work/xetal-ml"
source="${XETAL_ML_SOURCE:-https://github.com/softwarewrighter/X_eTaL-ML.git}"
if [ "${1:-}" = --pin ]; then
    repo="${XETAL_ML_REPO:-$root/../X_eTaL-ML}"
    sha="$(git -C "$repo" rev-parse --verify "${2:-HEAD}^{commit}")"
    echo "$sha" > "$root/XETAL_ML_COMMIT"
    if [ -d "$clone/.git" ] && ! git -C "$clone" cat-file -e "$sha^{commit}" 2>/dev/null; then
        git -C "$clone" fetch --quiet "$repo" "$sha" 2>/dev/null || true
    fi
    source="${XETAL_ML_SOURCE:-$repo}"
    echo "pinned X_eTaL-ML ${sha:0:7}: $(git -C "$repo" log -1 --format=%s "$sha")" >&2
fi
commit="$(tr -d '[:space:]' < "$root/XETAL_ML_COMMIT")"
if [ ! -d "$clone/.git" ]; then
    mkdir -p "$root/work"
    echo "xetal-ml: cloning $source into work/xetal-ml" >&2
    git clone --quiet "$source" "$clone"
fi
if ! git -C "$clone" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$clone" fetch --quiet origin
fi
if [ "$(git -C "$clone" rev-parse HEAD)" != "$commit" ]; then
    git -C "$clone" checkout --quiet --detach "$commit"
fi
[ -z "$(git -C "$clone" status --porcelain --untracked-files=no)" ] || { echo "xetal-ml: work/xetal-ml has local changes; it must match XETAL_ML_COMMIT" >&2; exit 1; }
echo "$clone"
