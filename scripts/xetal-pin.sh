#!/usr/bin/env bash
# Pin a newer X_eTaL: write the full SHA of a COMMITTED ref of the
# X_eTaL repository into XETAL_COMMIT, then build it. Commit
# XETAL_COMMIT on its own, after the goldens and timings pass.
#   scripts/xetal-pin.sh            # HEAD of ../X_eTaL
#   scripts/xetal-pin.sh REF        # any ref git rev-parse accepts
#   XETAL_REPO=/path scripts/xetal-pin.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="${XETAL_REPO:-$root/../X_eTaL}"
sha="$(git -C "$repo" rev-parse --verify "${1:-HEAD}^{commit}")"
echo "$sha" > "$root/XETAL_COMMIT"
# A local commit not yet on GitHub: fetch it into the clone from the sibling.
clone="$root/work/xetal"
if [ -d "$clone/.git" ] && ! git -C "$clone" cat-file -e "$sha^{commit}" 2>/dev/null; then
    git -C "$clone" fetch --quiet "$repo" "$sha" 2>/dev/null || true
fi
XETAL_SOURCE="${XETAL_SOURCE:-$repo}" "$root/scripts/xetal.sh" >/dev/null
echo "pinned X_eTaL ${sha:0:7}: $(git -C "$repo" log -1 --format=%s "$sha")"
echo "xetal-pin: now run the goldens (just test) and the timings (just bench-check)" >&2
