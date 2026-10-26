# Mojo backend results on the Apple M1 Max

The thirteen Accel demos (saga 12, steps 2 and 3), each
lowered from its X_eTaL source, emitted as Mojo, built as a shared
library and run on this Mac's GPU through Metal, 2026-10-23 (lookup
and windows 2026-10-26). Every
output agrees with the evaluator's baseline (`libs/Accel/tests/
demo-<name>.out`): Ints exactly, Floats within 1e-5
(`cargo test -p xetal-gpu-mojo`, which also prints `ok: exact_ints`,
`ok: axes` and `ok: indexed` for the semantics tests against the
interpreter, and `ok: concat`, `topk`, `reshape`, `rotate`,
`lookup`, `products` and `products_t4` for the structural operations
and tiled products). No demo is refused.

Device: `api=metal;name=Apple M1 Max` (Mojo 1.2.0.dev2026101305, MAX
26.7.0.dev2026101305, macOS 26.5; the repository on an SSD). Each
demo was run twice by `xetal-gpu run libs/Accel/demos/<name>.xtl
--device mojo --time` after clearing `work/mojo-cache`: the first
call builds the library, the second loads it. Build is the Mojo
compiler's time; run is the call into the library (the device context,
uploads, kernels and downloads); total is the whole command, the
lowering and the load included.

| Demo | Build, cold (ms) | Run, first call (ms) | Run, cached library (ms) | Total, cached (ms) |
| --- | --- | --- | --- | --- |
| saxpy | 1102 | 5.0 | 2.8 | 51 |
| threshold | 1008 | 3.5 | 3.3 | 52 |
| reduce-sum | 1092 | 5.1 | 3.7 | 52 |
| pipeline | 1019 | 4.0 | 3.4 | 52 |
| vector-add | 1000 | 3.4 | 3.2 | 51 |
| axis-reduce | 1084 | 5.2 | 4.0 | 54 |
| host-data | 1057 | 4.1 | 3.3 | 52 |
| matmul | 1042 | 4.1 | 4.0 | 52 |
| tiny-net | 1171 | 5.2 | 4.3 | 53 |
| topk | 1737 | 10.2 | 7.7 | 59 |
| transpose | 1104 | 4.7 | 4.1 | 49 |
| lookup | 1097 | 48.4 | 4.6 | 49 |
| windows | 1179 | 6.5 | 4.6 | 55 |

The cold build is 1.0 to 1.2 s for every demo but topk (1.7 s: its
lowering is the top-k composition, 18 kernels and 6 outputs), down
from about 2.6 s on 2026-10-16 with the same compiler (the first
build of the day warms it). The run is 3 to 5 ms for every demo but
topk (8 to 10 ms), almost all of it the device context: these are
programs of 8 to 20 items, correctness checks, not a benchmark. With
the library cached the whole command is 50 to 60 ms. The RTX 5060
Ti's numbers through the CUDA driver are in
[`rtx5060-mojo-results.md`](rtx5060-mojo-results.md); whole-process
times compare the hosts, not the cards.

Lookup's first call (48 ms) was the first device call of its process
after a cold build; its second, 4.6 ms, is in line with the others. With
`--tile 16` the products of matmul and windows are tiled
(`tiled_<name>`, shared memory) and give output identical to the
baselines; tiny-net's agrees within 1e-5 (f32 digits).
