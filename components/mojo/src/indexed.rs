//! Kernels whose result items read different positions of their inputs.
//! Axis folds and inner products retain the evaluator's right-fold order.
//! Gathers (lookup, rotate, reshape, concat) copy the item they read as
//! stored; top-k computes its line's threshold in each thread; a tiled
//! inner product stages T by T tiles in shared memory and still folds
//! the products from the last to the first.

use std::collections::BTreeSet;
use std::fmt::Write;

use xetal_gpu_opencl::Schedule;
use xetal_gpu_xir::ir::{permutation, transpose_layout};
use xetal_gpu_xir::{interp, Checked, Op, ReduceOp, Scalar, Shape, TableOp, ValueId};

use crate::{map_expression, mojo_type};

pub(super) fn kernel(checked: &Checked, schedule: &Schedule, id: ValueId) -> String {
    let value = &checked.program.values[id.0];
    let ty = |a| mojo_type(schedule.elem(checked.ty(a).scalar));
    let load = |a: ValueId, index: &str| {
        let name = &checked.program.values[a.0].name;
        let read = format!("b_{name}[unsafe_offset={index}]");
        if checked.ty(a).scalar == Scalar::Bool {
            format!("Bool({read} != 0)")
        } else {
            read
        }
    };
    let mut s = format!("\ndef indexed_{}(\n", value.name);
    for arg in value.op.args().into_iter().collect::<BTreeSet<_>>() {
        let _ = writeln!(
            s,
            "    b_{}: Pointer[{}, ImmutAnyOrigin],",
            checked.program.values[arg.0].name,
            ty(arg)
        );
    }
    let _ = writeln!(s, "    dst: Pointer[{}, MutAnyOrigin],\n):", ty(id));
    let _ = writeln!(s, "    var i = Int(block_idx.x * block_dim.x + thread_idx.x)");
    let n = checked.ty(id).shape.len();
    let _ = writeln!(s, "    if i < {n}:");
    // Empty results still need syntactically valid code, without
    // zero divisors in unreachable indexing expressions.
    if n == 0 {
        s.push_str("        pass\n");
        return s;
    }
    // A raw read: the item as stored (a Bool stays its Int32 0 or 1),
    // for the gathers, whose result has the argument's element type.
    let raw = |a: ValueId, index: &str| format!("b_{}[unsafe_offset={index}]", checked.program.values[a.0].name);
    let lift = |s: &Shape| if s.is_scalar() { Shape(vec![1]) } else { s.clone() };
    match &value.op {
        Op::Lookup { axis, idx, arg } => {
            // Positions from 1; a stray one is refused by the host for a
            // bound input and clamped here for a computed one, as the
            // OpenCL kernel does (a device cannot refuse).
            let (_, len, inner) = interp::axis_split(&checked.ty(*arg).shape, *axis);
            let np = checked.ty(*idx).shape.len();
            let _ = writeln!(s, "        var o = i // {}\n        var r = i % {}", np * inner, np * inner);
            let _ = writeln!(s, "        var p = Int({}) - 1", raw(*idx, &format!("r // {inner}")));
            let _ = writeln!(s, "        p = max(0, min(p, {}))", len - 1);
            let _ = writeln!(
                s,
                "        dst[unsafe_offset=i] = {}",
                raw(*arg, &format!("(o * {len} + p) * {inner} + r % {inner}"))
            );
            return s;
        }
        Op::Rotate { axis, amounts, arg } => {
            // One copy per amount; along the axis item j of a copy rotated
            // by a is the argument's item (j + a) mod len, the shift
            // reduced to 0..len here so the device needs no signed modulo.
            let shape = &checked.ty(*arg).shape;
            let (_, len, inner) = interp::axis_split(shape, *axis);
            let m = shape.len();
            let shifts: Vec<i64> = amounts.iter().map(|a| a.rem_euclid(len as i64)).collect();
            let _ = writeln!(s, "        var c = i // {m}\n        var r = i % {m}");
            let mut pick = shifts.last().unwrap().to_string();
            for (c, sh) in shifts.iter().enumerate().rev().skip(1) {
                pick = format!("{sh} if c == {c} else ({pick})");
            }
            let _ = writeln!(s, "        var shift = {pick}");
            let _ = writeln!(s, "        var o = r // {}\n        var j = (r // {inner}) % {len}", len * inner);
            let _ = writeln!(
                s,
                "        dst[unsafe_offset=i] = {}",
                raw(*arg, &format!("(o * {len} + (j + shift) % {len}) * {inner} + r % {inner}"))
            );
            return s;
        }
        Op::Reshape { arg, .. } => {
            let m = checked.ty(*arg).shape.len().max(1);
            let _ = writeln!(s, "        dst[unsafe_offset=i] = {}", raw(*arg, &format!("i % {m}")));
            return s;
        }
        Op::Concat { axis, a, b } => {
            let (sa, sb) = (lift(&checked.ty(*a).shape), lift(&checked.ty(*b).shape));
            let (_, lx, inner) = interp::axis_split(&sa, *axis);
            let ly = sb.0[axis - 1];
            let _ = writeln!(
                s,
                "        var o = i // {}\n        var r = i % {}",
                (lx + ly) * inner,
                (lx + ly) * inner
            );
            let _ = writeln!(s, "        var j = r // {inner}\n        var k = r % {inner}");
            let _ = writeln!(
                s,
                "        if j < {lx}:\n            dst[unsafe_offset=i] = {}",
                raw(*a, &format!("(o * {lx} + j) * {inner} + k"))
            );
            let _ = writeln!(
                s,
                "        else:\n            dst[unsafe_offset=i] = {}",
                raw(*b, &format!("(o * {ly} + j - {lx}) * {inner} + k"))
            );
            return s;
        }
        Op::TopK { k, axis, arg } => {
            // The line's k-th largest distinct value (its least when it has
            // fewer), found by taking the largest below the last k - 1
            // times; the item is in the mask when it is at least that.
            let shape = &checked.ty(*arg).shape;
            let axis = interp::topk_axis(*axis, shape.rank());
            let (_, len, inner) = interp::axis_split(shape, axis);
            let t = ty(*arg);
            let _ = writeln!(s, "        var base = (i // {}) * {} + i % {inner}", len * inner, len * inner);
            let _ = writeln!(s, "        var t: {t} = {}", raw(*arg, "base"));
            let _ = writeln!(
                s,
                "        for j in range(1, {len}):\n            t = max(t, {})",
                raw(*arg, &format!("base + j * {inner}"))
            );
            let _ = writeln!(s, "        var taken = 1\n        while taken < {k}:");
            let _ = writeln!(s, "            var found = False\n            var next: {t} = t");
            let _ = writeln!(
                s,
                "            for j in range({len}):\n                var x = {}",
                raw(*arg, &format!("base + j * {inner}"))
            );
            let _ = writeln!(
                s,
                "                if x < t and (not found or x > next):\n                    next = x\n                    found = True"
            );
            let _ = writeln!(
                s,
                "            if not found:\n                break\n            t = next\n            taken += 1"
            );
            let out = ty(id);
            let _ = writeln!(
                s,
                "        dst[unsafe_offset=i] = {out}(1) if {} >= t else {out}(0)",
                raw(*arg, "i")
            );
            return s;
        }
        _ => {}
    }
    let expr = match &value.op {
        Op::Reduce { op, arg, axis: Some(axis) } => {
            let (_, len, inner) = interp::axis_split(&checked.ty(*arg).shape, *axis);
            let _ = writeln!(s, "        var base = (i // {inner}) * {len} * {inner} + i % {inner}");
            let _ = writeln!(s, "        var acc = {}", load(*arg, &format!("base + {} * {inner}", len - 1)));
            let _ = writeln!(s, "        var j = Int({len}) - 2\n        while j >= 0:");
            let lhs = load(*arg, &format!("base + j * {inner}"));
            let combine = match op {
                ReduceOp::Add => format!("{lhs} + acc"),
                ReduceOp::Mul => format!("{lhs} * acc"),
                ReduceOp::Min => format!("min({lhs}, acc)"),
                ReduceOp::Max => format!("max({lhs}, acc)"),
            };
            let _ = writeln!(s, "            acc = {combine}\n            j -= 1");
            "acc".into()
        }
        Op::Table { op, a, b } => {
            let bn = checked.ty(*b).shape.len();
            let x = load(*a, &format!("i // {bn}"));
            let y = load(*b, &format!("i % {bn}"));
            match op {
                TableOp::Left => x,
                TableOp::Right => y,
                TableOp::Map(op) => map_expression(*op, &x, &y),
            }
        }
        Op::Take { arg, .. } | Op::Drop { arg, .. } => {
            let slice = interp::slice_of(&value.op, &checked.ty(*arg).shape);
            let (len, inner, start, keep) = (slice.len, slice.inner, slice.start, slice.keep);
            load(
                *arg,
                &format!("((i // {inner}) // {keep} * {len} + (i // {inner}) % {keep} + {start}) * {inner} + i % {inner}"),
            )
        }
        Op::Ravel { arg } => load(*arg, "i"),
        Op::Transpose { perm, arg } => {
            let shape = &checked.ty(*arg).shape;
            let perm = permutation(shape.rank(), perm).expect("checked permutation");
            let (out, strides) = transpose_layout(shape, &perm);
            let mut divisor = 1;
            let mut terms = Vec::new();
            for (&dim, &stride) in out.0.iter().zip(&strides).rev() {
                terms.push(format!("((i // {divisor}) % {dim}) * {stride}"));
                divisor *= dim;
            }
            load(*arg, &if terms.is_empty() { "0".into() } else { terms.join(" + ") })
        }
        Op::Matmul { a, b } => {
            let (_, len, cols) = interp::matmul_split(&checked.ty(*a).shape, &checked.ty(*b).shape);
            let product = |j: &str| {
                format!(
                    "{} * {}",
                    load(*a, &format!("(i // {cols}) * {len} + {j}")),
                    load(*b, &format!("{j} * {cols} + i % {cols}"))
                )
            };
            let _ = writeln!(s, "        var acc = {}", product(&(len - 1).to_string()));
            let _ = writeln!(
                s,
                "        var j = Int({len}) - 2\n        while j >= 0:\n            acc = {} + acc\n            j -= 1",
                product("j")
            );
            "acc".into()
        }
        _ => unreachable!("an indexed operation"),
    };
    let expr = if checked.ty(id).scalar == Scalar::Bool {
        format!("{}(1) if {expr} else {}(0)", ty(id), ty(id))
    } else {
        expr
    };
    let _ = writeln!(s, "        dst[unsafe_offset=i] = {expr}");
    s
}

/// An inner product in T by T tiles (the schedule's `tile`): a block of
/// T * T threads computes one tile of the result, staging a tile of each
/// argument in shared memory per step along the inner axis. The steps
/// run from the last tile to the first and the products within a tile
/// from the last to the first, so the fold is the evaluator's right fold.
pub(super) fn tiled_matmul(checked: &Checked, schedule: &Schedule, id: ValueId, tile: usize) -> String {
    let value = &checked.program.values[id.0];
    let Op::Matmul { a, b } = &value.op else {
        unreachable!("a product")
    };
    let (rows, len, cols) = interp::matmul_split(&checked.ty(*a).shape, &checked.ty(*b).shape);
    let t = mojo_type(schedule.elem(checked.ty(id).scalar));
    let name = |v: ValueId| checked.program.values[v.0].name.clone();
    let (na, nb) = (name(*a), name(*b));
    let tiles_c = cols.max(1).div_ceil(tile);
    let tiles_k = len.max(1).div_ceil(tile);
    let mut params = vec![format!("    b_{na}: Pointer[{t}, ImmutAnyOrigin],")];
    if b != a {
        params.push(format!("    b_{nb}: Pointer[{t}, ImmutAnyOrigin],"));
    }
    format!(
        r#"
def tiled_{v}(
{params}
    dst: Pointer[{t}, MutAnyOrigin],
):
    """%{v}: {rows} by {cols} items in {tile} by {tile} tiles, the inner axis of {len} folded from the right."""
    var sa = unsafe_stack_allocation[{tt}, {t}, address_space=.SHARED]()
    var sb = unsafe_stack_allocation[{tt}, {t}, address_space=.SHARED]()
    var tid = Int(thread_idx.x)
    var ty = tid // {tile}
    var tx = tid % {tile}
    var row = (Int(block_idx.x) // {tiles_c}) * {tile} + ty
    var col = (Int(block_idx.x) % {tiles_c}) * {tile} + tx
    var acc: {t} = {t}(0)
    var started = False
    var step = {last}
    while step >= 0:
        var ka = step * {tile} + tx
        var kb = step * {tile} + ty
        sa[unsafe_offset=tid] = b_{na}[unsafe_offset=row * {len} + ka] if row < {rows} and ka < {len} else {t}(0)
        sb[unsafe_offset=tid] = b_{nb}[unsafe_offset=kb * {cols} + col] if kb < {len} and col < {cols} else {t}(0)
        barrier()
        var j = {tile} - 1
        while j >= 0:
            if step * {tile} + j < {len}:
                var p = sa[unsafe_offset=ty * {tile} + j] * sb[unsafe_offset=j * {tile} + tx]
                if started:
                    acc = p + acc
                else:
                    acc = p
                    started = True
            j -= 1
        barrier()
        step -= 1
    if row < {rows} and col < {cols}:
        dst[unsafe_offset=row * {cols} + col] = acc
"#,
        v = value.name,
        params = params.join("\n"),
        tt = tile * tile,
        last = tiles_k as i64 - 1,
    )
}
