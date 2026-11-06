#!/usr/bin/env python3
"""Check actual X_eTaL meshes at zero, quarter and full vane rotations."""
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def main():
    libs = Path(subprocess.check_output([str(ROOT / "scripts/xetal-libraries.sh")], text=True).strip())
    xetal = subprocess.check_output([str(ROOT / "scripts/xetal.sh")], text=True).strip()
    env = dict(os.environ, XETAL_PATH=":".join(str(p) for p in sorted(libs.glob("libs/*/src"))))
    prefix = (ROOT / "examples/ray/logo.xtl").read_text().split("pixels :=", 1)[0]
    out = ROOT / "work/ray-live/geometry-check"
    out.mkdir(parents=True, exist_ok=True)
    meshes = []
    for i, angle in enumerate([0.0, math.pi / 2, 2 * math.pi]):
        source = "\n".join(f"angle := {angle}" if line.startswith("angle :=") else line for line in prefix.splitlines())
        path = out / f"frame-{i}.xtl"
        path.write_text(source + "\nr_avel triangles\n")
        run = subprocess.run([xetal, "run", "--seed", "1", "--ascii", str(path)], env=env, capture_output=True, text=True, check=True)
        meshes.append([float(n) for n in run.stdout.split()])
    assert len(meshes[0]) == len(meshes[1]) == len(meshes[2])
    assert len(meshes[0]) % 9 == 0
    count = len(meshes[0]) // 9
    for moved in meshes[1:]:
        assert moved[:44 * 9] == meshes[0][:44 * 9], "stationary X changed"
        assert moved[-14 * 9:] == meshes[0][-14 * 9:], "fixed post or floor changed"
        for j in range(44 * 9, len(moved) - 14 * 9, 3):
            a, b = meshes[0][j:j + 3], moved[j:j + 3]
            ra = a[0] ** 2 + (a[1] - .5) ** 2 + a[2] ** 2
            rb = b[0] ** 2 + (b[1] - .5) ** 2 + b[2] ** 2
            assert abs(ra - rb) < 1e-9, "vane pivot distance changed"
    assert max(abs(a - b) for a, b in zip(meshes[0], meshes[2])) < 1e-10, "full turn is not seamless"
    quarter = meshes[1][44 * 9:-14 * 9]
    original = meshes[0][44 * 9:-14 * 9]
    assert sum(original[::3]) / (len(original) // 3) > 2
    assert abs(sum(quarter[::3]) / (len(quarter) // 3)) < 1e-10
    assert sum(quarter[2::3]) / (len(quarter) // 3) < -2, "arrow did not rotate outward about X"
    print(f"ray scene: {count} triangles; fixed X/post/floor, pivot radii and full-turn geometry pass")


if __name__ == "__main__":
    main()
