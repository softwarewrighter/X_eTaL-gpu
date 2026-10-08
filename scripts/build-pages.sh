#!/usr/bin/env bash
# Build the live site into pages/ (not tracked on main; scripts/publish-pages.sh
# publishes it): the documentation (scripts/doc-site.sh, pages/doc) and
# the landing page (scripts/build-pages.py, pages/index.html).
#   scripts/build-pages.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -rf "$root/pages"
"$root/scripts/doc-site.sh"
"$root/scripts/build-pages.py"
touch "$root/pages/.nojekyll"
