# Asks for X_eTaL libraries

## Ownership and request policy

Reusable algorithms belong in the base/library repositories. Prefer
existing libraries, then ordinary library functions and macro libraries;
new notation should normally be a macro library rather than language
syntax. This file records new or changed general library requests from
X_eTaL-gpu. GPU lowering, scheduling, kernels, device management and
execution integration belong here, building on the existing components.

Reserve [xetal-asks.md](xetal-asks.md) mainly for demonstrated language
bugs. A core feature request needs evidence that library and macro
solutions cannot express the requirement. Existing historical asks are
not automatically new requirements for ray tracing. This repository
records requests; it does not modify sibling repositories.

Each concrete ask records status, intended owner, existing APIs checked,
required behavior, CPU example, tests, why reuse is insufficient, macro
versus function rationale, and any temporary implementation and removal
condition. Record an upstream commit when landed and pin it deliberately.

## Current local library audit

| Code | Finding | Ownership |
| ---- | ------- | --------- |
| `libs/Accel/src/Accel.xtl` | Pure array functions: sums, dot/matrix products, masks, row reductions, transposes, embeddings, windows, convolution, argmax/top-k, dense layers and softmax. No device selection or OpenCL calls; these are general algorithms despite the name. | Existing tested GPU subset/reference fixture here. Reuse general library APIs where available; review consolidation with X_eTaL-libraries and X_eTaL-ML before moving exports. Preserve compatibility and baselines; no migration performed by this audit. |
| `libs/Accel/demos/` and `tests/` | Source programs and evaluator baselines that exercise the lowering and device implementations. | GPU integration coverage belongs here, even when the functions come from a general library. |
| `extensions/gpu/lib/Gpu.xtl` | Device/session facade: load, bind, run, output, schedules, kernel/explain and execution helpers. | GPU-specific; stays here. Reusable math should not accumulate in the facade. |
| `extensions/gpu/lib/Gpu.xtlm` | `gp:a_ccel<` generates calls to this GPU runner; potentially reusable runner pattern, currently coupled to `gp:` and its session interface. | Current macro stays here. Generalize elsewhere only if an actual shared runner contract is established. |
| Proposed RayTrace functions and scene macros | Camera/vector geometry, intersections, materials, lighting and scene notation have CPU as well as GPU meaning. | General library repository; this repo owns the acceleration support and demo harness. |
| Image encoding | TIFF encoding is general host I/O rather than GPU semantics. | Reuse suitable non-GPL host tooling/library; keep only the pixel-output adapter and demo commands here. No rendering in the encoder. |

## Ray-tracing candidates to audit

No confirmed missing API or bug is claimed yet. These are audit targets,
not filed feature requests; the ray-source step must check the owning
repositories before assigning concrete asks.

| Candidate | Proposed owner | Audit and acceptance |
| --------- | -------------- | -------------------- |
| Vector/geometry functions | X_eTaL-libraries, existing math/geometry library if suitable | Find dot, cross, normalization, transforms and intersection helpers; reuse or request only missing functions. CPU analytical examples and pinned export types; lowerable compositions of existing operations. |
| Ray-tracing functions | X_eTaL-libraries | Camera rays, triangle hits, deterministic nearest-hit selection, lighting and bounded reflections as ordinary X_eTaL. Shared CPU/GPU source; keep devices and schedules outside the APIs. |
| Scene/geometry macro library | X_eTaL-libraries | First consider compile-time notation for materials, polygon extrusion and scene tables; expand to ordinary X_eTaL with inspectable output and useful diagnostics. Evaluate against existing macro libraries before proposing syntax. |
| Existing Accel overlap | X_eTaL-libraries and X_eTaL-ML | Map exports to current math and ML APIs, identify duplicates and compatible semantics, then request consolidation only with a migration/pinning plan. Do not relocate working code without that review. |

This ledger and [ray-tracing-plan.md](ray-tracing-plan.md) guide the
capability audit. GPU-specific gaps become local saga steps; actual
general library gaps become concrete entries here after the audit.
