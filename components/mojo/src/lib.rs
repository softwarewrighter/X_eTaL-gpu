//! The Mojo backend: a checked XIR program becomes Mojo source, the way
//! `components/opencl` writes OpenCL C, ported from
//! `modular-fork/rust-demo` (the first proof: Rust writes Mojo, Mojo
//! runs the GPU). Each run of elementwise values of one shape between
//! reductions is one fused kernel (one thread per element, the
//! intermediate values in registers); a whole-array reduction is a tree
//! in shared memory, a pass per 256-fold; the host side of the generated
//! file moves the arrays through Mojo's `DeviceContext` (Metal on Apple
//! silicon, CUDA on NVIDIA, HIP on AMD, the emitter not knowing which)
//! and exports a C ABI that `host.rs` calls through `dlopen`. The
//! generated file is meant to read like a Mojo programmer's.
//!
//! Every operation of the IR is taken. Elementwise chains fuse; whole
//! reductions are trees; everything whose result items read other
//! positions (axis folds, tables, slices, ravel and reshape, inner
//! products, transposes, lookups, rotations, concatenations, top-k) is
//! an indexed kernel, one thread per result item (`indexed.rs`); a
//! product under a schedule with a `tile` is staged in shared memory.

pub mod host;
mod indexed;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write;

use xetal_gpu_opencl::{Elem, Schedule};
use xetal_gpu_xir::{CastOp, Checked, Const, MapOp, Op, ReduceOp, Scalar, ValueId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Threads per block of every kernel here.
pub const TPB: usize = 256;

/// What the generated program expects on its C ABI, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Abi {
    /// The arrays `xir_run` takes, in order: the program's inputs by
    /// name, then the constants the host uploads (`None`).
    pub inputs: Vec<(ValueId, Option<String>)>,
    /// The program's outputs, in order.
    pub outputs: Vec<ValueId>,
}

#[derive(Debug, Clone)]
enum Step {
    /// One fused elementwise kernel over values of one length.
    Kernel { values: Vec<ValueId>, len: usize },
    /// A whole-array reduction: the passes' lengths, from the source's to 1.
    Reduce { value: ValueId, passes: Vec<usize> },
    /// One thread per result, with indexed reads of materialized arguments.
    Indexed { value: ValueId },
}

/// Mojo source for a checked program under a schedule (its element
/// widths), and the ABI the source exports.
pub fn emit(checked: &Checked, schedule: &Schedule, name: &str) -> Result<(String, Abi)> {
    let program = &checked.program;
    // Which values the device keeps in buffers: inputs, uploaded
    // constants, reductions, outputs, and what crosses a stage.
    let mut stage_of: HashMap<ValueId, usize> = HashMap::new();
    let mut steps: Vec<Step> = Vec::new();
    let mut current: Vec<ValueId> = Vec::new();
    let mut current_len: Option<usize> = None;
    let flush = |steps: &mut Vec<Step>, current: &mut Vec<ValueId>, current_len: &mut Option<usize>| {
        if !current.is_empty() {
            steps.push(Step::Kernel {
                values: std::mem::take(current),
                len: current_len.take().unwrap(),
            });
        }
    };
    for (i, v) in program.values.iter().enumerate() {
        let id = ValueId(i);
        let ty = checked.ty(id);
        match &v.op {
            Op::Input | Op::Const(_) => {}
            Op::Map { .. } | Op::Select { .. } | Op::Cast { .. } => {
                let len = ty.shape.len();
                if current_len != Some(len) {
                    flush(&mut steps, &mut current, &mut current_len);
                    current_len = Some(len);
                }
                stage_of.insert(id, steps.len());
                current.push(id);
            }
            Op::Reduce { arg, axis: None, .. } => {
                flush(&mut steps, &mut current, &mut current_len);
                let mut n = checked.ty(*arg).shape.len();
                let mut passes = vec![n];
                while n > 1 {
                    n = n.div_ceil(TPB);
                    passes.push(n);
                }
                steps.push(Step::Reduce { value: id, passes });
            }
            Op::Reduce { axis: Some(_), .. }
            | Op::Table { .. }
            | Op::Take { .. }
            | Op::Drop { .. }
            | Op::Ravel { .. }
            | Op::Matmul { .. }
            | Op::Transpose { .. }
            | Op::Lookup { .. }
            | Op::Rotate { .. }
            | Op::Reshape { .. }
            | Op::Concat { .. }
            | Op::TopK { .. } => {
                flush(&mut steps, &mut current, &mut current_len);
                steps.push(Step::Indexed { value: id });
            }
        }
    }
    flush(&mut steps, &mut current, &mut current_len);

    let outputs: BTreeSet<ValueId> = program.outputs.iter().copied().collect();
    let mut materialized: BTreeSet<ValueId> = outputs.clone();
    for (i, v) in program.values.iter().enumerate() {
        let id = ValueId(i);
        match &v.op {
            Op::Input
            | Op::Reduce { .. }
            | Op::Table { .. }
            | Op::Take { .. }
            | Op::Drop { .. }
            | Op::Ravel { .. }
            | Op::Matmul { .. }
            | Op::Transpose { .. }
            | Op::Lookup { .. }
            | Op::Rotate { .. }
            | Op::Reshape { .. }
            | Op::Concat { .. }
            | Op::TopK { .. } => {
                materialized.insert(id);
            }
            Op::Const(_) if !checked.ty(id).shape.is_scalar() => {
                materialized.insert(id);
            }
            _ => {}
        }
        if steps
            .iter()
            .any(|s| matches!(s, Step::Indexed { value } | Step::Reduce { value, .. } if *value == id))
        {
            materialized.extend(v.op.args());
        }
        for a in v.op.args() {
            let crosses = match &v.op {
                Op::Reduce { .. } => true,
                _ => stage_of.get(&a) != stage_of.get(&id),
            };
            if crosses && stage_of.contains_key(&a) {
                materialized.insert(a);
            }
        }
    }

    // The ABI: inputs, then the constants with more than one written item.
    let mut abi = Abi {
        inputs: Vec::new(),
        outputs: program.outputs.clone(),
    };
    for (i, v) in program.values.iter().enumerate() {
        let id = ValueId(i);
        match &v.op {
            Op::Input => abi.inputs.push((id, Some(v.name.clone()))),
            Op::Const(c) if materialized.contains(&id) && c.len() > 1 => abi.inputs.push((id, None)),
            _ => {}
        }
    }

    let elem = |id: ValueId| schedule.elem(checked.ty(id).scalar);
    let is_bool = |id: ValueId| checked.ty(id).scalar == Scalar::Bool;
    let host_type = |id: ValueId| match checked.ty(id).scalar {
        Scalar::F32 | Scalar::F64 => "Float64",
        Scalar::I32 | Scalar::I64 => "Int64",
        Scalar::Bool => "Int32",
    };
    let literal_of = |id: ValueId| -> Option<String> {
        match &program.values[id.0].op {
            Op::Const(c) if checked.ty(id).shape.is_scalar() && !materialized.contains(&id) => Some(literal(c, elem(id), is_bool(id))),
            _ => None,
        }
    };

    // A product is tiled when the schedule asks for tiles (plan A6) and
    // its items are numbers (a Bool product is not in the subset).
    let tiled = |id: ValueId| schedule.tile > 0 && matches!(program.values[id.0].op, Op::Matmul { .. }) && !is_bool(id);

    let mut s = String::new();
    let kernels = steps.iter().filter(|s| matches!(s, Step::Kernel { .. })).count();
    let reductions = steps.iter().filter(|s| matches!(s, Step::Reduce { .. })).count();
    let indexed_count = steps.iter().filter(|s| matches!(s, Step::Indexed { .. })).count();
    let _ = writeln!(
        s,
        "# {name}.mojo: generated by xetal-gpu (components/mojo) from {name}. Do not edit.\n# {kernels} fused kernel stage(s), {reductions} reduction(s), {} device buffer(s).",
        materialized.len()
    );
    if indexed_count > 0 {
        let _ = writeln!(s, "# {indexed_count} indexed kernel(s), one thread per result.");
    }
    s.push_str(
        r#"
from std.math import exp, log, cos, sin, sqrt, floor, inf, nan
from std.memory import unsafe_stack_allocation
from std.runtime import initialize_runtime
from std.sys import has_accelerator

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
"#,
    );
    let _ = writeln!(s, "\ncomptime TPB = {TPB}");

    // The reduction kernels, one per operation and element type used.
    let mut reduce_kernels: BTreeMap<(String, String), String> = BTreeMap::new();
    for step in &steps {
        if let Step::Reduce { value, .. } = step {
            if let Op::Reduce { op, .. } = &program.values[value.0].op {
                let e = elem(*value);
                reduce_kernels
                    .entry((op.to_string(), dtype(e).to_string()))
                    .or_insert_with(|| reduce_kernel(*op, e));
            }
        }
    }
    for k in reduce_kernels.values() {
        s.push_str(k);
    }

    for step in &steps {
        if let Step::Indexed { value } = step {
            if tiled(*value) {
                s.push_str(&indexed::tiled_matmul(checked, schedule, *value, schedule.tile));
                continue;
            }
            s.push_str(&indexed::kernel(checked, schedule, *value));
        }
    }

    // The stages.
    let mut stage_no = 0;
    for step in &steps {
        if let Step::Kernel { values, len } = step {
            let inside: BTreeSet<ValueId> = values.iter().copied().collect();
            let mut reads: BTreeSet<ValueId> = BTreeSet::new();
            for &id in values {
                for a in program.values[id.0].op.args() {
                    if !inside.contains(&a) && literal_of(a).is_none() {
                        reads.insert(a);
                    }
                }
            }
            let writes: Vec<ValueId> = values.iter().copied().filter(|id| materialized.contains(id)).collect();
            let _ = writeln!(s, "\ndef stage{stage_no}(");
            for &r in &reads {
                let _ = writeln!(
                    s,
                    "    b_{}: Pointer[{}, ImmutAnyOrigin],",
                    program.values[r.0].name,
                    mojo_type(elem(r))
                );
            }
            for &w in &writes {
                let _ = writeln!(
                    s,
                    "    b_{}: Pointer[{}, MutAnyOrigin],",
                    program.values[w.0].name,
                    mojo_type(elem(w))
                );
            }
            let _ = writeln!(s, "    n: Int32,\n):");
            let names: Vec<String> = values.iter().map(|id| format!("%{}", program.values[id.0].name)).collect();
            let _ = writeln!(
                s,
                "    \"\"\"Stage {stage_no}: {} fused, one thread per element.\"\"\"",
                names.join(", ")
            );
            let _ = writeln!(s, "    var i = Int(block_idx.x * block_dim.x + thread_idx.x)");
            let _ = writeln!(s, "    if i < Int(n):");
            for &r in &reads {
                let v = &program.values[r.0];
                let idx = if checked.ty(r).shape.len() == *len { "i" } else { "0" };
                let load = if is_bool(r) {
                    format!("Bool(b_{}[unsafe_offset={idx}] != 0)", v.name)
                } else {
                    format!("b_{}[unsafe_offset={idx}]", v.name)
                };
                let _ = writeln!(s, "        var v_{} = {}", v.name, load);
            }
            for &id in values {
                let _ = writeln!(
                    s,
                    "        var v_{} = {}",
                    program.values[id.0].name,
                    expression(checked, schedule, id, &literal_of)
                );
            }
            for &w in &writes {
                let v = &program.values[w.0];
                let stored = if is_bool(w) {
                    format!("{t}(1) if v_{} else {t}(0)", v.name, t = mojo_type(elem(w)))
                } else {
                    format!("v_{}", v.name)
                };
                let _ = writeln!(s, "        b_{}[unsafe_offset=i] = {}", v.name, stored);
            }
            stage_no += 1;
        }
    }

    // The exports: init, device info, run.
    s.push_str(
        r#"

@export("xir_init")
def xir_init() abi("C") -> Int32:
    """Initializes the Mojo runtime; 1 when this build targets a GPU, else 0."""
    initialize_runtime()
    comptime if has_accelerator():
        return 1
    else:
        return 0


@export("xir_device_info")
def xir_device_info(buf: Pointer[UInt8, MutAnyOrigin], cap: Int64) abi("C") -> Int64:
    """Writes `api=...;name=...` for the device the kernels run on; its length, or -1."""
    initialize_runtime()
    try:
        var ctx = DeviceContext()
        var s = String("api=", ctx.api(), ";name=", ctx.name())
        var bytes = s.as_bytes()
        var n = min(len(bytes), Int(cap))
        for i in range(n):
            buf[unsafe_offset=i] = bytes[i]
        return Int64(n)
    except e:
        return -1


@export("xir_run")
def xir_run(
    inputs: Pointer[Pointer[UInt8, ImmutAnyOrigin], ImmutAnyOrigin],
    outputs: Pointer[Pointer[UInt8, MutAnyOrigin], ImmutAnyOrigin],
) abi("C") -> Int32:
    """Runs the program: inputs[k] as the ABI lists them (the program's
    inputs, then the uploaded constants), outputs[j] in output order, every
    array as Float64, Int64 or Int32 Bool items on the host, converted
    to the scheduled device width here.

    Returns 0 when done, 1 when the device failed.
    """
    initialize_runtime()
    try:
        var ctx = DeviceContext()
"#,
    );
    for (k, (id, _)) in abi.inputs.iter().enumerate() {
        let v = &program.values[id.0];
        let n = checked.ty(*id).shape.len();
        let allocation = n.max(1);
        let _ = writeln!(
            s,
            "        var h_{} = ctx.enqueue_create_host_buffer[DType.{}]({allocation})",
            v.name,
            dtype(elem(*id))
        );
        let _ = writeln!(
            s,
            "        for i in range({n}):\n            h_{}[i] = {}(inputs[unsafe_offset={k}].bitcast[{}]()[unsafe_offset=i])",
            v.name,
            mojo_type(elem(*id)),
            host_type(*id)
        );
    }
    let _ = writeln!(s, "        # device buffers, one per materialized value");
    for &id in &materialized {
        let v = &program.values[id.0];
        let n = checked.ty(id).shape.len().max(1);
        let _ = writeln!(
            s,
            "        var b_{} = ctx.enqueue_create_buffer[DType.{}]({n})",
            v.name,
            dtype(elem(id))
        );
        match &v.op {
            Op::Input => {
                let _ = writeln!(s, "        ctx.enqueue_copy(dst_buf=b_{0}, src_buf=h_{0})", v.name);
            }
            Op::Const(c) => {
                if c.len() > 1 {
                    let _ = writeln!(s, "        ctx.enqueue_copy(dst_buf=b_{0}, src_buf=h_{0})", v.name);
                } else {
                    let _ = writeln!(
                        s,
                        "        b_{}.enqueue_fill({})",
                        v.name,
                        literal(&if c.is_empty() { c.empty_fill() } else { c.clone() }, elem(id), is_bool(id))
                    );
                }
            }
            _ => {}
        }
    }
    let mut stage_no = 0;
    for step in &steps {
        match step {
            Step::Kernel { values, len } => {
                let inside: BTreeSet<ValueId> = values.iter().copied().collect();
                let mut reads: BTreeSet<ValueId> = BTreeSet::new();
                for &id in values {
                    for a in program.values[id.0].op.args() {
                        if !inside.contains(&a) && literal_of(a).is_none() {
                            reads.insert(a);
                        }
                    }
                }
                let writes: Vec<ValueId> = values.iter().copied().filter(|id| materialized.contains(id)).collect();
                let mut args: Vec<String> = reads
                    .iter()
                    .chain(writes.iter())
                    .map(|id| format!("b_{}", program.values[id.0].name))
                    .collect();
                args.push(format!("Int32({len})"));
                let blocks = (*len).max(1).div_ceil(TPB);
                let _ = writeln!(
                    s,
                    "        # stage {stage_no}: {len} element(s), {blocks} block(s) of TPB\n        ctx.enqueue_function[stage{stage_no}]({}, grid_dim={blocks}, block_dim=TPB)",
                    args.join(", ")
                );
                stage_no += 1;
            }
            Step::Indexed { value } if tiled(*value) => {
                let v = &program.values[value.0];
                let Op::Matmul { a, b } = &v.op else { unreachable!() };
                let (rows, _, cols) = xetal_gpu_xir::interp::matmul_split(&checked.ty(*a).shape, &checked.ty(*b).shape);
                let t = schedule.tile;
                let mut args = vec![format!("b_{}", program.values[a.0].name)];
                if b != a {
                    args.push(format!("b_{}", program.values[b.0].name));
                }
                args.push(format!("b_{}", v.name));
                let blocks = rows.max(1).div_ceil(t) * cols.max(1).div_ceil(t);
                let _ = writeln!(
                    s,
                    "        # %{}: {rows} by {cols} in {t} by {t} tiles, a block of {} threads per tile\n        ctx.enqueue_function[tiled_{}]({}, grid_dim={blocks}, block_dim={})",
                    v.name,
                    t * t,
                    v.name,
                    args.join(", "),
                    t * t
                );
            }
            Step::Indexed { value } => {
                let v = &program.values[value.0];
                let args: BTreeSet<_> = v.op.args().into_iter().collect();
                let mut args: Vec<_> = args.iter().map(|a| format!("b_{}", program.values[a.0].name)).collect();
                args.push(format!("b_{}", v.name));
                let blocks = checked.ty(*value).shape.len().max(1).div_ceil(TPB);
                let _ = writeln!(
                    s,
                    "        ctx.enqueue_function[indexed_{}]({}, grid_dim={blocks}, block_dim=TPB)",
                    v.name,
                    args.join(", ")
                );
            }
            Step::Reduce { value, passes } => {
                let (Op::Reduce { op, arg, .. }, v) = (&program.values[value.0].op, &program.values[value.0]) else {
                    unreachable!()
                };
                let e = elem(*value);
                let src0 = format!("b_{}", program.values[arg.0].name);
                let _ = writeln!(
                    s,
                    "        # reduce {op} %{} -> %{}: a partial per block of {TPB}, until one is left ({} pass(es))",
                    program.values[arg.0].name,
                    v.name,
                    passes.len() - 1
                );
                let mut src = src0;
                for (p, pair) in passes.windows(2).enumerate() {
                    let (n, out_n) = (pair[0], pair[1]);
                    let dst = if out_n == 1 {
                        format!("b_{}", v.name)
                    } else {
                        let d = format!("b_{}_p{p}", v.name);
                        let _ = writeln!(s, "        var {d} = ctx.enqueue_create_buffer[DType.{}]({out_n})", dtype(e));
                        d
                    };
                    let _ = writeln!(
                        s,
                        "        ctx.enqueue_function[reduce_{op}_{}]({src}, {dst}, Int32({n}), grid_dim={out_n}, block_dim=TPB)",
                        dtype(e)
                    );
                    src = dst;
                }
                if passes.len() == 1 {
                    // A single item: the reduction is the item itself.
                    let _ = writeln!(s, "        ctx.enqueue_copy(dst_buf=b_{}, src_buf={src})", v.name);
                }
            }
        }
    }
    let _ = writeln!(s, "        # the outputs, back to the host");
    for (j, &o) in program.outputs.iter().enumerate() {
        let v = &program.values[o.0];
        let n = checked.ty(o).shape.len().max(1);
        let _ = writeln!(
            s,
            "        var o_{j} = ctx.enqueue_create_host_buffer[DType.{}]({n})\n        ctx.enqueue_copy(dst_buf=o_{j}, src_buf=b_{})",
            dtype(elem(o)),
            v.name
        );
    }
    let _ = writeln!(s, "        ctx.synchronize()");
    for (j, &o) in program.outputs.iter().enumerate() {
        let n = checked.ty(o).shape.len();
        let _ = writeln!(
            s,
            "        for i in range({n}):\n            outputs[unsafe_offset={j}].bitcast[{host}]()[unsafe_offset=i] = {host}(o_{j}[i])",
            host = host_type(o)
        );
    }
    s.push_str(
        r#"        return 0
    except e:
        print("xir_run failed:", e)
        return 1
"#,
    );
    Ok((s, abi))
}

/// The Mojo scalar type of a device element.
pub fn mojo_type(e: Elem) -> &'static str {
    match e {
        Elem::I32 => "Int32",
        Elem::I64 => "Int64",
        Elem::F32 => "Float32",
        Elem::F64 => "Float64",
    }
}

fn dtype(e: Elem) -> &'static str {
    match e {
        Elem::I32 => "int32",
        Elem::I64 => "int64",
        Elem::F32 => "float32",
        Elem::F64 => "float64",
    }
}

fn literal(c: &Const, e: Elem, bool_value: bool) -> String {
    let t = mojo_type(e);
    match c {
        Const::Int(v) => format!("{t}({})", v[0]),
        Const::Float(v) => float_literal(v[0], e),
        Const::Bool(v) => {
            if bool_value {
                (if v[0] { "True" } else { "False" }).to_string()
            } else {
                format!("{t}({})", if v[0] { 1 } else { 0 })
            }
        }
    }
}

fn float_literal(x: f64, e: Elem) -> String {
    let t = mojo_type(e);
    if x.is_nan() {
        format!("{t}(nan[DType.{}]())", dtype(e))
    } else if x.is_infinite() {
        if x > 0.0 {
            format!("{t}(inf[DType.{}]())", dtype(e))
        } else {
            format!("{t}(-inf[DType.{}]())", dtype(e))
        }
    } else {
        format!("{t}({x:?})")
    }
}

fn expression(checked: &Checked, schedule: &Schedule, id: ValueId, literal_of: &dyn Fn(ValueId) -> Option<String>) -> String {
    let program = &checked.program;
    let v = |a: ValueId| literal_of(a).unwrap_or_else(|| format!("v_{}", program.values[a.0].name));
    let ty = |a: ValueId| mojo_type(schedule.elem(checked.ty(a).scalar));
    match &program.values[id.0].op {
        Op::Map { op, args } => map_expression(*op, &v(args[0]), &args.get(1).map(|&b| v(b)).unwrap_or_default()),
        Op::Select { cond, a, b } => format!("{} if {} else {}", v(*a), v(*cond), v(*b)),
        Op::Cast { op: CastOp { to }, arg } => {
            let from = checked.ty(*arg).scalar;
            let x = v(*arg);
            match (from, to) {
                (Scalar::Bool, Scalar::Bool) => x,
                (Scalar::Bool, _) => format!("{t}(1) if {x} else {t}(0)", t = ty(id)),
                (_, Scalar::Bool) => format!("Bool({x} != 0)"),
                (Scalar::F32 | Scalar::F64, Scalar::I32 | Scalar::I64) => format!("{}(floor({x}))", ty(id)),
                _ => format!("{}({x})", ty(id)),
            }
        }
        _ => unreachable!("only elementwise values have expressions"),
    }
}

fn map_expression(op: MapOp, x: &str, y: &str) -> String {
    use MapOp::*;
    match op {
        Add => format!("{x} + {y}"),
        Sub => format!("{x} - {y}"),
        Mul => format!("{x} * {y}"),
        Div => format!("{x} / {y}"),
        IDiv => format!("{x} // {y}"),
        Mod => format!("{x} % {y}"),
        Min => format!("min({x}, {y})"),
        Max => format!("max({x}, {y})"),
        Neg => format!("-{x}"),
        Abs => format!("abs({x})"),
        Exp => format!("exp({x})"),
        Log => format!("log({x})"),
        Pow => format!("{x} ** {y}"),
        Cos => format!("cos({x})"),
        Sin => format!("sin({x})"),
        Sqrt => format!("sqrt({x})"),
        Eq => format!("Bool({x} == {y})"),
        Ne => format!("Bool({x} != {y})"),
        Lt => format!("Bool({x} < {y})"),
        Le => format!("Bool({x} <= {y})"),
        Gt => format!("Bool({x} > {y})"),
        Ge => format!("Bool({x} >= {y})"),
        And => format!("{x} and {y}"),
        Or => format!("{x} or {y}"),
        Not => format!("not {x}"),
    }
}

fn reduce_kernel(op: ReduceOp, e: Elem) -> String {
    let t = mojo_type(e);
    let (identity, combine) = match op {
        ReduceOp::Add => (format!("{t}(0)"), "lhs + rhs"),
        ReduceOp::Mul => (format!("{t}(1)"), "lhs * rhs"),
        ReduceOp::Max => ("src[unsafe_offset=0]".to_string(), "max(lhs, rhs)"),
        ReduceOp::Min => ("src[unsafe_offset=0]".to_string(), "min(lhs, rhs)"),
    };
    format!(
        r#"
def reduce_{op}_{dt}(
    src: Pointer[{t}, ImmutAnyOrigin],
    dst: Pointer[{t}, MutAnyOrigin],
    n: Int32,
):
    """One block folds TPB items of src into dst[block]: a tree in shared memory."""
    var shared = unsafe_stack_allocation[TPB, {t}, address_space=.SHARED]()
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * TPB + tid
    var v: {t} = {identity}
    if i < Int(n):
        v = src[unsafe_offset=i]
    shared[unsafe_offset=tid] = v
    barrier()
    var active = TPB // 2
    while active > 0:
        if tid < active:
            var lhs = shared[unsafe_offset=tid]
            var rhs = shared[unsafe_offset=tid + active]
            shared[unsafe_offset=tid] = {combine}
        barrier()
        active //= 2
    if tid == 0:
        dst[unsafe_offset=Int(block_idx.x)] = shared[unsafe_offset=0][0]
"#,
        dt = dtype(e),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use xetal_gpu_xir::{check, text};

    #[test]
    fn a_fused_stage_and_a_reduction_are_emitted() {
        let p = text::parse("%x = input f64 [8]\n%two = const f64 [] 2.0\n%y = map mul %x %two\n%s = reduce add %y\noutput %s\n").unwrap();
        let checked = check::check(p).unwrap();
        let (src, abi) = emit(&checked, &Schedule::default(), "t").unwrap();
        assert!(src.contains("var v_y = v_x * Float32(2.0)"));
        assert!(src.contains("def reduce_add_float32("));
        assert!(src.contains("enqueue_function[stage0](b_x, b_y, Int32(8), grid_dim=1, block_dim=TPB)"));
        assert_eq!(abi.inputs, vec![(ValueId(0), Some("x".into()))]);
        assert_eq!(abi.outputs, vec![ValueId(3)]);
    }

    #[test]
    fn a_large_reduction_takes_several_passes() {
        let p = text::parse("%x = input f64 [70000]\n%s = reduce max %x\noutput %s\n").unwrap();
        let checked = check::check(p).unwrap();
        let (src, _) = emit(&checked, &Schedule::default(), "t").unwrap();
        assert!(src.contains("(3 pass(es))"));
        assert!(src.contains("Int32(70000), grid_dim=274"));
        assert!(src.contains("Int32(274), grid_dim=2"));
        assert!(src.contains("Int32(2), grid_dim=1"));
    }

    #[test]
    fn a_tiled_schedule_stages_products_in_shared_memory() {
        let p = text::parse("%a = input f64 [5 7]\n%b = input f64 [7 3]\n%c = matmul %a %b\noutput %c\n").unwrap();
        let checked = check::check(p).unwrap();
        let (untiled, _) = emit(&checked, &Schedule::default(), "t").unwrap();
        assert!(untiled.contains("def indexed_c("), "{untiled}");
        let schedule = Schedule {
            tile: 4,
            ..Schedule::default()
        };
        let (src, _) = emit(&checked, &schedule, "t").unwrap();
        assert!(src.contains("def tiled_c("), "{src}");
        assert!(
            src.contains("unsafe_stack_allocation[16, Float32, address_space=.SHARED]()"),
            "{src}"
        );
        // 5 by 3 in 4 by 4 tiles: 2 by 1 blocks of 16; the 7 inner items in 2 steps, the last first.
        assert!(
            src.contains("enqueue_function[tiled_c](b_a, b_b, b_c, grid_dim=2, block_dim=16)"),
            "{src}"
        );
        assert!(src.contains("var step = 1"), "{src}");
    }
}
