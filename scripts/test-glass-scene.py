#!/usr/bin/env python3
"""Validate the actual composed Scene v2 geometry and future glass volumes."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def main():
    libraries = Path(subprocess.check_output([str(ROOT / "scripts/xetal-libraries.sh")], text=True).strip())
    xetal = subprocess.check_output([str(ROOT / "scripts/xetal.sh")], text=True).strip()
    env = dict(os.environ, XETAL_PATH=":".join(str(p) for p in sorted(libraries.glob("libs/*/src"))))
    source = (ROOT / "examples/ray/glass.xtl").read_text().split("clip :=", 1)[0]
    source += '''
"ch:" u_se< "Check"
c1 := "three analytic spheres" ch:t_est 3 4 ch:i_s s_hape spheres
c2 := "three crystals plus floor and panels" ch:t_est 30 3 3 ch:i_s s_hape triangles
c3 := "separate sphere object identities" ch:t_est 1 2 3 ch:i_s studiosphereobjectids
c4 := "crystal faces share volume identity" ch:t_est ((8 r_eshape 4) c_at (8 r_eshape 5) c_at (8 r_eshape 6) c_at 6 r_eshape 0) ch:i_s studiotriangleobjectids
c5 := "mixed primitive IDs follow declaration order" ch:t_est (3 + r_ange 30) ch:i_s studiotriangleids
c6 := "sphere IDs independent of material" ch:t_est 1 2 3 ch:i_s studiosphereids
c7 := "all crystals are closed outward convex shells" ch:t_est (24 t_ake triangles) ch:i_s (sc:c_losed 8 t_ake triangles) c_at (sc:c_losed 8 t_ake 8 d_rop triangles) c_at sc:c_losed 8 t_ake 16 d_rop triangles
c8 := "sphere bottoms on floor" ch:t_est (3 r_eshape 0.0) ch:n_ear (2 s_elect_2 spheres) - 4 s_elect_2 spheres
c9 := "crystal bottoms on floor" ch:t_est (3 r_eshape 0.0) ch:n_ear ('m_in r_/ 2 s_elect_2 24 3 r_eshape 8 t_ake triangles) c_at ('m_in r_/ 2 s_elect_2 24 3 r_eshape 8 t_ake 8 d_rop triangles) c_at 'm_in r_/ 2 s_elect_2 24 3 r_eshape 8 t_ake 16 d_rop triangles
## Shared glass preflight checks separation even before the opaque preview.
glassshading := (studiomaterials, 1 0 0, sc:o_ptics 3 5 r_eshape 1 1.5 0.01 0.01 0.01  0 1 0 0 0  0 1 0 0 0, emission, studiolights)
(_, _, _, glassids, materialids, bounds) := (geometry gl:p_reflight links)_ glassshading
c10 := "six disjoint closed glass objects" ch:t_est 4 5 6 1 2 3 ch:i_s glassids
c11 := "shared material does not merge volumes" ch:t_est (6 r_eshape 1) ch:i_s materialids
c12 := "one world bound per glass volume" ch:t_est 6 2 3 ch:i_s s_hape bounds
ch:r_eport c1 ch:a_nd c2 ch:a_nd c3 ch:a_nd c4 ch:a_nd c5 ch:a_nd c6 ch:a_nd c7 ch:a_nd c8 ch:a_nd c9 ch:a_nd c10 ch:a_nd c11 ch:a_nd c12
'''
    path = ROOT / "work/ray-glass/composition-check.xtl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    run = subprocess.run([xetal, "run", "--seed", "1", "--ascii", str(path)], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "FAIL" not in run.stdout and "12 checks, all passed" in run.stdout, run.stdout
    print("glass scene: 12 checks pass; closed shells, separate IDs, floor contacts and shared glass preflight")


if __name__ == "__main__":
    main()
