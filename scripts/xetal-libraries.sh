#!/usr/bin/env bash
# Fetch the committed general libraries used by ray-tracing integration.
# Never use the sibling working tree as a runtime dependency.
# XETAL_LIBRARIES_SOURCE may name a local Git repo for the initial clone.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
snapshot=""
if [ "$#" != 0 ]; then
    [ "$#" = 2 ] && [ "$1" = --snapshot ] && [[ "$2" =~ ^[0-9a-f]{40}$ ]] || { echo "usage: xetal-libraries.sh [--snapshot FULL_COMMITTED_SHA]" >&2; exit 2; }
    snapshot="$2"
fi
clone="$root/work/xetal-libraries"
source="${XETAL_LIBRARIES_SOURCE:-https://github.com/softwarewrighter/X_eTaL-libraries.git}"
commit="$(tr -d '[:space:]' < "$root/XETAL_LIBRARIES_COMMIT")"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "xetal-libraries: pin must be a full commit ID" >&2; exit 1; }
if [ ! -d "$clone/.git" ]; then
    mkdir -p "$root/work"
    echo "xetal-libraries: cloning $source" >&2
    git clone --quiet --no-hardlinks "$source" "$clone"
fi
[ -z "$(git -C "$clone" status --porcelain)" ] || { echo "xetal-libraries: work/xetal-libraries has local changes; refusing checkout" >&2; exit 1; }
if ! git -C "$clone" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$clone" fetch --quiet origin "$commit"
fi
if [ "$(git -C "$clone" rev-parse HEAD)" != "$commit" ]; then
    git -C "$clone" checkout --quiet --detach "$commit"
fi
if [ -n "$snapshot" ]; then
    if ! git -C "$clone" cat-file -e "$snapshot^{commit}" 2>/dev/null; then
        git -C "$clone" fetch --quiet origin "$snapshot"
    fi
    target="$root/work/xetal-library-snapshots/$snapshot"
    if [ ! -e "$target" ]; then
        mkdir -p "$(dirname "$target")"
        git -C "$clone" worktree add --quiet --detach "$target" "$snapshot"
    fi
    [ "$(git -C "$target" rev-parse HEAD)" = "$snapshot" ] && [ -z "$(git -C "$target" status --porcelain)" ] || { echo "xetal-libraries: snapshot must be clean at the requested commit" >&2; exit 1; }
    printf '%s\n' "$target"
else
    printf '%s\n' "$clone"
fi
