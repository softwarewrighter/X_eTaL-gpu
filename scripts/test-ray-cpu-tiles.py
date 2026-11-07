#!/usr/bin/env python3
"""Compare actual full-camera renders with independently evaluated strips."""
from pathlib import Path
import subprocess
import sys
import importlib.util
import os

ROOT = Path(__file__).resolve().parent.parent


def main():
    directory = ROOT / "work/ray-hires/tile-check"
    directory.mkdir(parents=True, exist_ok=True)
    for samples in [1, 2]:
        def render(name, rows=None):
            prefix = directory / f"s{samples}-{name}"
            args = [sys.executable, str(ROOT / "scripts/ray-render-cpu.py"), "--width", "8", "--height", "5", "--angle", "0", "--samples-axis", str(samples), "--output", str(prefix)]
            if rows:
                args += ["--row-start", str(rows[0]), "--rows", str(rows[1])]
            result = subprocess.run(args, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout+result.stderr
            return prefix.with_suffix(".ppm").read_bytes().split(b"\n255\n",1)[1]
        full = render("full")
        strips = render("top", (0,2)) + render("bottom", (2,3))
        assert full == strips, f"full-camera sampling changed across strips at sample axis {samples}"
    spec = importlib.util.spec_from_file_location("sample_source", ROOT / "scripts/ray-render-cpu.py")
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    synthetic = '\n'.join(['"rt:" u_se< "RayTrace"', 'width := 2', 'height := 1', 'angle := 0.0',
        'linear := f_loat 8 3 r_eshape 1 0 0  0 0 0  0 1 0  0 0 0  0 0 0  0 0 0  0 1 0  0 0 0', '1.0 rt:s_rgb linear'])+'\n'
    path = directory / "known-samples.xtl"
    path.write_text(codec.prepare_source(synthetic, 2, 1, 0, 0, 1, 2, "logo"))
    libs = Path(subprocess.check_output([str(ROOT / "scripts/xetal-libraries.sh")], text=True).strip())
    env = dict(os.environ, XETAL_PATH=":".join(str(p) for p in sorted(libs.glob("libs/*/src"))))
    values = list(map(float, subprocess.check_output([str(ROOT / "bin/xetal"), "run", "--ascii", str(path)], env=env, text=True).split()))
    assert all(abs(a-b)<1e-12 for a,b in zip(values, [0.5370987304831942, 0, 0, 0, 0.7353569830524495, 0])) and len(values)==6
    print("ray CPU strips: exact full-frame pixels at 1 and 4 samples/pixel, including a partial last strip")


if __name__ == "__main__":
    main()
