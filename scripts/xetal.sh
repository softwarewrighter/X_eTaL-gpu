#!/usr/bin/env bash
# Get xetal (the layout X_eTaL's docs/vendoring.md describes): clone
# X_eTaL into work/xetal (gitignored), check out the known-good commit
# in XETAL_COMMIT, build the release CLI into target/xetal/, symlink
# bin/xetal to it, and print that path. Safe to run at any time: with
# nothing to do it only confirms the build.
#   scripts/xetal.sh                          # quiet; prints bin/xetal's path
#   XETAL_SOURCE=../X_eTaL scripts/xetal.sh   # first clone from a local checkout
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit="$(tr -d '[:space:]' < "$root/XETAL_COMMIT")"
source="${XETAL_SOURCE:-https://github.com/softwarewrighter/X_eTaL.git}"
clone="$root/work/xetal"

if [ ! -d "$clone/.git" ]; then
    mkdir -p "$root/work"
    echo "xetal: cloning $source into work/xetal" >&2
    git clone --quiet "$source" "$clone"
fi
if ! git -C "$clone" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$clone" fetch --quiet origin
fi
if [ "$(git -C "$clone" rev-parse HEAD)" != "$commit" ]; then
    git -C "$clone" checkout --quiet --detach "$commit"
fi
# work/xetal is a pristine checkout of the known-good commit: never edit it.
[ -z "$(git -C "$clone" status --porcelain)" ] || { echo "xetal: work/xetal has local changes; it must match XETAL_COMMIT" >&2; exit 1; }

export CARGO_TARGET_DIR="$root/target/xetal"
(cd "$clone/components/cli" && cargo build -q --release -p xetal-cli >&2)

mkdir -p "$root/bin"
ln -sfn "../target/xetal/release/xetal" "$root/bin/xetal"
echo "$root/bin/xetal"
