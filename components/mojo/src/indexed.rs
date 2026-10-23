//! Kernels whose result items read different positions of their inputs.
//! Axis folds and inner products retain the evaluator's right-fold order.

use std::collections::BTreeSet;
use std::fmt::Write;

use xetal_gpu_opencl::Schedule;
use xetal_gpu_xir::ir::{permutation, transpose_layout};
use xetal_gpu_xir::{interp, Checked, Op, ReduceOp, Scalar, TableOp, ValueId};

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
