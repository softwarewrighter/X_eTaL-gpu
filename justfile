# X_eTaL-gpu tasks. Recipes call scripts/*.sh, which hold the logic and
# work without just too. `just` alone lists the recipes.

set positional-arguments

# List the recipes
default:
    @just --list

# Get and build xetal at the known-good commit in XETAL_COMMIT (clone in work/xetal, binary bin/xetal)
xetal:
    @scripts/xetal.sh

# Pin a newer X_eTaL: a committed ref of ../X_eTaL (default HEAD) into XETAL_COMMIT, built; then test and commit it on its own
xetal-pin ref="HEAD":
    scripts/xetal-pin.sh "$1"

# The pinned X_eTaL: XETAL_COMMIT and the binary's own report
xetal-version:
    @cat XETAL_COMMIT
    @"$(scripts/xetal.sh)" --version

# Evaluate an expression with the pinned xetal, every library on XETAL_PATH: just eval "'+ r_/ 1 2 3"
eval expr:
    @scripts/xt eval -e "$1"

# The Core IR of an expression (what the accelerator lowering starts from): just core "2 * 1 2 3"
core expr:
    @scripts/xt core -e "$1"

# Get and build xetal-x, X_eTaL-extensions' bridge host, at XETAL_EXTENSIONS_COMMIT (clone in work/xetal-extensions, bin/xetal-x)
xetal-x:
    @scripts/xetal-extensions.sh

# Pin a committed ref of ../X_eTaL-extensions (default HEAD) in XETAL_EXTENSIONS_COMMIT and build it; commit it on its own
extensions-pin ref="HEAD":
    scripts/xetal-extensions.sh --pin "$1"

# Build the gpu extension beside xetal-x (extensions/gpu)
gpu-ext:
    @scripts/gpu-ext.sh

# Check the pinned xetal-x: it reports its commit, hello answers through the bridge, every library program gives its baseline under its X_eTaL
check-xetal-x:
    scripts/check-xetal-x.sh

# Check the pinned X_eTaL: CLI builds, answers, reports its commit, dumps Core
check-xetal:
    scripts/check-xetal.sh

# The libraries: name, recommended alias, what it is
libs:
    @scripts/libs.py table | column -t -s "$(printf '\t')"

# The directories to put on XETAL_PATH: export XETAL_PATH="$(just path)"
path:
    @scripts/libs.py path

# Start a library from templates/Library: just new-lib Accel ac: "the acceleratable subset"
new-lib name alias summary:
    scripts/new-lib.sh "$1" "$2" "$3"

# Run a library's test programs (or one): just run-lib Accel basics
run-lib name prog="":
    @scripts/run-lib.sh "$1" ${2:+"$2"}

# Run a library's demos (or one): just demo-lib Accel vector-add
demo-lib name prog="":
    @scripts/run-lib.sh --demos "$1" ${2:+"$2"}

# A library's demo as a notebook, each statement then its output
show-lib name prog="":
    @scripts/run-lib.sh --echo --demos "$1" ${2:+"$2"}

# A library's exported names and their types: just types Accel
types name:
    @for f in libs/$1/src/$1.xtl libs/$1/src/$1.xtlm; do [ -f "$f" ] && scripts/xt type "$f"; done; true

# Test one library with reg-rs: pinned types, test programs, demos
test-lib name:
    scripts/test-libs.sh "$1"

# Create missing baselines and accept new output for one library (review the diff!)
bless-lib name:
    XETAL_BLESS=1 scripts/test-libs.sh "$1"

# Test everything: every library's baselines, the Rust components, the equivalence checks
test:
    scripts/test-libs.sh
    @[ -f components/Cargo.toml ] && (cd components && cargo test -q) || true
    @[ -x scripts/check-equiv.sh ] && scripts/check-equiv.sh || true

# The reductions and the pipeline at large sizes (default 1024 65536 1048576), evaluator against interpreter and GPU, timed: just sizes
sizes *n:
    scripts/check-sizes.py --table "$@"

# Build the GPU tool, xetal-gpu (components/cli), in release mode
build:
    cd components && cargo build -q --release -p xetal-gpu-cli

# The full pre-commit gate: pinned X_eTaL, tooling self-tests, baselines, components, equivalence, spelling, markdown
gate:
    scripts/gate.sh

# Show the agentrail saga state and the current step
status:
    agentrail status

# Open the saga plan
plan:
    @cat docs/plan.md
