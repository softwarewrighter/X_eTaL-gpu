#!/usr/bin/env bash
# Publish the live site: build pages/ fresh from this commit, then make
# it the only commit of the gh-pages branch, which GitHub Pages serves.
# The branch is replaced on every publish (no history), so built files
# never accumulate in git; main never tracks pages/. Refuses when the
# work tree has uncommitted changes, so the site always matches a
# commit of main (its footer names that commit).
#   scripts/publish-pages.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
git diff --quiet HEAD -- || { echo "publish: commit your changes first (the site must match a commit)" >&2; exit 1; }
"$root/scripts/build-pages.sh"
[ -f pages/index.html ] || { echo "publish: no pages/index.html after the build" >&2; exit 1; }
head="$(git rev-parse --short HEAD)"
xetal="$(cut -c1-7 XETAL_COMMIT)"
index="$(mktemp)"; rm -f "$index"
trap 'rm -f "$index"' EXIT
export GIT_INDEX_FILE="$index"
git --work-tree=pages add -A -f .
tree="$(git write-tree)"
unset GIT_INDEX_FILE
commit="$(git commit-tree "$tree" -m "pages: built from main $head with X_eTaL $xetal")"
git push --quiet --force origin "$commit:refs/heads/gh-pages"
echo "published pages/ (main $head, X_eTaL $xetal) to gh-pages: https://softwarewrighter.github.io/X_eTaL-gpu/"
