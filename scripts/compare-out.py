#!/usr/bin/env python3
"""Compare two outputs as X_eTaL prints them: the same lines, the same
items on each; Ints exactly, Floats within a relative tolerance (0:
exactly, as the reference interpreter must; a device computing in
f32 is allowed 1e-5), -0.0 equal to 0.0.

  scripts/compare-out.py EXPECTED ACTUAL [TOLERANCE]

Exit 0 when they agree; the first difference is printed otherwise.
"""
import sys


def items(path):
    with open(path) as f:
        return [line.split() for line in f.read().splitlines()]


def same(a, b, tol):
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    if ("." not in a and "." not in b) or tol == 0:
        return x == y          # Ints exactly; -0.0 == 0.0
    return abs(x - y) <= tol * max(abs(x), abs(y), 1e-30) or abs(x - y) <= 1e-9


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    want, got = items(sys.argv[1]), items(sys.argv[2])
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    if len(want) != len(got):
        sys.exit(f"{len(want)} lines expected, got {len(got)}")
    for n, (w, g) in enumerate(zip(want, got), 1):
        if len(w) != len(g):
            sys.exit(f"line {n}: {len(w)} items expected, got {len(g)}: {' '.join(g)}")
        for i, (a, b) in enumerate(zip(w, g), 1):
            if not same(a, b, tol):
                sys.exit(f"line {n}, item {i}: expected {a}, got {b}")
    print(f"ok: {len(want)} lines agree" + (f" (tolerance {tol:g})" if tol else " (exactly)"))


if __name__ == "__main__":
    main()
