//! From a checked program and a schedule to a plan: grouping the
//! elementwise values into fused kernels, laying out the reductions,
//! and writing the OpenCL C.

use std::collections::{HashMap, HashSet};

use xetal_gpu_xir::interp::Slice;
use xetal_gpu_xir::ir::{is_identity, permutation, transpose_layout};
use xetal_gpu_xir::{Checked, Const, MapOp, Op, ReduceOp, Scalar, Shape, TableOp, Ty, ValueId};

use crate::{Arg, Buffer, BufferId, Elem, Init, KernelInfo, Launch, Plan, Schedule};

/// Where a value lives while the plan is built.
#[derive(Debug, Clone)]
enum Place {
    /// A buffer on the device.
    Buffer(BufferId),
    /// A single-value constant, written as a literal.
    Literal(String),
    /// Computed inside the current kernel as this C variable.
    Register(String),
}

struct Builder<'a> {
    checked: &'a Checked,
    schedule: &'a Schedule,
    buffers: Vec<Buffer>,
    kernels: Vec<KernelInfo>,
    launches: Vec<Launch>,
    source: String,
    places: HashMap<ValueId, Place>,
    /// Which values read each value.
    users: HashMap<ValueId, Vec<ValueId>>,
    /// The elementwise values gathered for the next kernel, and its shape.
    group: Vec<ValueId>,
    group_shape: Option<Shape>,
}

/// Build the plan for a checked program under a schedule.
pub fn plan(checked: &Checked, schedule: &Schedule) -> Plan {
    assert!(
        schedule.work_group.is_power_of_two() && schedule.work_group > 0,
        "the work-group size is a power of two"
    );
    let program = &checked.program;
    let mut users: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    for (i, v) in program.values.iter().enumerate() {
        for a in v.op.args() {
            users.entry(a).or_default().push(ValueId(i));
        }
    }
    let mut b = Builder {
        checked,
        schedule,
        buffers: Vec::new(),
        kernels: Vec::new(),
        launches: Vec::new(),
        source: String::new(),
        places: HashMap::new(),
        users,
        group: Vec::new(),
        group_shape: None,
    };
    let outputs: HashSet<ValueId> = program.outputs.iter().copied().collect();
    for (i, v) in program.values.iter().enumerate() {
        let id = ValueId(i);
        let ty = checked.ty(id);
        match &v.op {
            Op::Input => {
                let buf = b.buffer(Some(id), b.schedule.elem(ty.scalar), ty.shape.len(), Init::Input(v.name.clone()));
                b.places.insert(id, Place::Buffer(buf));
            }
            Op::Const(c) => {
                if ty.shape.is_scalar() {
                    b.places.insert(id, Place::Literal(literal(c, b.schedule.elem(ty.scalar))));
                } else {
                    let buf = b.buffer(Some(id), b.schedule.elem(ty.scalar), ty.shape.len(), Init::Const(c.clone()));
                    b.places.insert(id, Place::Buffer(buf));
                }
            }
            Op::Map { .. } | Op::Select { .. } | Op::Cast { .. } => {
                if b.group_shape.as_ref() != Some(&ty.shape) {
                    b.flush(&outputs);
                    b.group_shape = Some(ty.shape.clone());
                }
                b.group.push(id);
                b.places.insert(id, Place::Register(format!("t_{}", v.name)));
            }
            Op::Take { arg, .. } | Op::Drop { arg, .. } => {
                b.flush(&outputs);
                let slice = xetal_gpu_xir::interp::slice_of(&v.op, &checked.ty(*arg).shape);
                b.slice(id, *arg, slice);
            }
            Op::Rotate { axis, amounts, arg } => {
                b.flush(&outputs);
                b.rotate(id, *arg, *axis, amounts);
            }
            Op::Reshape { shape, arg } => {
                b.flush(&outputs);
                if shape.len() == checked.ty(*arg).shape.len() {
                    // The same items in the same order: the same buffer (or literal), no launch.
                    let place = b.places[arg].clone();
                    b.places.insert(id, place);
                } else {
                    b.cycle(id, *arg);
                }
            }
            Op::Ravel { arg } => {
                // Row order is the ravel: the same buffer, read as a vector.
                b.flush(&outputs);
                let place = b.places[arg].clone();
                b.places.insert(id, place);
            }
            Op::Table { op, a, b: rhs } => {
                b.flush(&outputs);
                b.table(id, *op, *a, *rhs);
            }
            Op::Matmul { a, b: rhs } => {
                b.flush(&outputs);
                b.matmul(id, *a, *rhs);
            }
            Op::Transpose { perm, arg } => {
                b.flush(&outputs);
                let perm = permutation(checked.ty(*arg).shape.rank(), perm).expect("checked");
                if is_identity(&perm) {
                    // A vector, a single value or `1 2 t_ranspose`: the same
                    // items in the same order, so the same buffer, no launch.
                    let place = b.places[arg].clone();
                    b.places.insert(id, place);
                } else {
                    b.transpose(id, *arg, &perm);
                }
            }
            Op::Reduce { op, arg, axis } => {
                b.flush(&outputs);
                match axis {
                    // Along an axis of a vector is the whole vector: the tree.
                    Some(a) if checked.ty(*arg).shape.rank() > 1 => b.reduce_axis(id, *op, *arg, *a),
                    _ => b.reduce(id, *op, *arg),
                }
            }
        }
    }
    b.flush(&outputs);
    let outputs = program
        .outputs
        .iter()
        .map(|o| match &b.places[o] {
            Place::Buffer(buf) => (*o, *buf),
            Place::Literal(_) => {
                // An output that is a single constant (or its ravel or its
                // transpose, which are itself): give it a buffer of its own.
                let ty = checked.ty(*o);
                let mut v = *o;
                let c = loop {
                    match &program.value(v).op {
                        Op::Const(c) => break c.clone(),
                        Op::Ravel { arg } | Op::Transpose { arg, .. } | Op::Reshape { arg, .. } => v = *arg,
                        _ => unreachable!("a literal is a single constant"),
                    }
                };
                let buf = b.buffer(Some(*o), b.schedule.elem(ty.scalar), 1, Init::Const(c));
                (*o, buf)
            }
            Place::Register(_) => unreachable!("an output is written to a buffer when its kernel is flushed"),
        })
        .collect();
    let mut source = String::new();
    if b.buffers.iter().any(|x| x.elem == Elem::F64) {
        source.push_str("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
    }
    source.push_str(HELPERS);
    source.push_str(&b.source);
    Plan {
        schedule: schedule.clone(),
        source,
        buffers: b.buffers,
        kernels: b.kernels,
        launches: b.launches,
        outputs,
    }
}

/// Floored division and remainder, as X_eTaL's `d_iv` and `m_od`.
const HELPERS: &str = "\
// X_eTaL-gpu: generated OpenCL C 1.2. Floored division and remainder, as X_eTaL's d_iv and m_od.
inline long xetal_idiv_long(long a, long b) { long q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline long xetal_imod_long(long a, long b) { return a - xetal_idiv_long(a, b) * b; }
inline int xetal_idiv_int(int a, int b) { int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }
inline int xetal_imod_int(int a, int b) { return a - xetal_idiv_int(a, b) * b; }
";

impl<'a> Builder<'a> {
    fn buffer(&mut self, value: Option<ValueId>, elem: Elem, len: usize, init: Init) -> BufferId {
        let id = BufferId(self.buffers.len());
        let name = match value {
            Some(v) => format!("b_{}", self.checked.program.value(v).name),
            None => format!("p{}", id.0),
        };
        self.buffers.push(Buffer {
            id,
            value,
            name,
            elem,
            len,
            init,
        });
        id
    }

    fn ty(&self, id: ValueId) -> &Ty {
        self.checked.ty(id)
    }

    fn elem_of(&self, id: ValueId) -> Elem {
        self.schedule.elem(self.ty(id).scalar)
    }

    /// Read a value's item i (or its one item) inside a kernel.
    fn read(&self, id: ValueId) -> String {
        match &self.places[&id] {
            Place::Buffer(b) => {
                let name = &self.buffers[b.0].name;
                if self.ty(id).shape.is_scalar() {
                    format!("{name}[0]")
                } else {
                    format!("{name}[i]")
                }
            }
            Place::Literal(s) => s.clone(),
            Place::Register(r) => r.clone(),
        }
    }

    /// The gathered elementwise values become one kernel.
    fn flush(&mut self, outputs: &HashSet<ValueId>) {
        if self.group.is_empty() {
            self.group_shape = None;
            return;
        }
        let group = std::mem::take(&mut self.group);
        let shape = self.group_shape.take().expect("a group has a shape");
        let in_group: HashSet<ValueId> = group.iter().copied().collect();
        let n = shape.len();
        let kname = format!("k{}", self.kernels.iter().filter(|k| k.elements > 0).count());
        // The buffers the kernel reads: every argument that is not in the group and not a literal.
        let mut reads: Vec<BufferId> = Vec::new();
        for id in &group {
            for a in self.checked.program.value(*id).op.args() {
                if let Place::Buffer(b) = &self.places[&a] {
                    if !reads.contains(b) {
                        reads.push(*b);
                    }
                }
            }
        }
        // The values written: outputs, and values a later value outside the group reads.
        let writes: Vec<ValueId> = group
            .iter()
            .copied()
            .filter(|id| outputs.contains(id) || self.users.get(id).is_some_and(|us| us.iter().any(|u| !in_group.contains(u))))
            .collect();
        let mut body = String::new();
        for id in &group {
            let v = self.checked.program.value(*id);
            let elem = self.elem_of(*id);
            let expr = match &v.op {
                Op::Map { op, args } => {
                    let a: Vec<String> = args.iter().map(|x| self.read(*x)).collect();
                    map_expr(*op, &a, self.elem_of(args[0]))
                }
                Op::Select { cond, a, b } => format!("({} ? {} : {})", self.read(*cond), self.read(*a), self.read(*b)),
                Op::Cast { arg, .. } => cast_expr(
                    &self.read(*arg),
                    self.ty(*arg).scalar,
                    self.elem_of(*arg),
                    self.ty(*id).scalar,
                    elem,
                ),
                _ => unreachable!("elementwise only"),
            };
            body.push_str(&format!("    {} t_{} = {};\n", elem, v.name, expr));
        }
        let mut params: Vec<String> = reads
            .iter()
            .map(|b| format!("__global const {}* {}", self.buffers[b.0].elem, self.buffers[b.0].name))
            .collect();
        let mut write_bufs = Vec::new();
        for id in &writes {
            let elem = self.elem_of(*id);
            let buf = self.buffer(Some(*id), elem, n, Init::Device);
            params.push(format!("__global {}* {}", elem, self.buffers[buf.0].name));
            body.push_str(&format!(
                "    {}[i] = t_{};\n",
                self.buffers[buf.0].name,
                self.checked.program.value(*id).name
            ));
            write_bufs.push(buf);
        }
        self.source.push_str(&format!(
            "\n// {}: {} element{}, one work-item each; computes {}\n__kernel void {}({}) {{\n    size_t i = get_global_id(0);\n    if (i >= {}) return;\n{}}}\n",
            kname,
            n,
            if n == 1 { "" } else { "s" },
            group.iter().map(|id| format!("%{}", self.checked.program.value(*id).name)).collect::<Vec<_>>().join(" "),
            kname,
            params.join(", "),
            n,
            body
        ));
        let local = self.schedule.work_group.min(n.max(1).next_power_of_two());
        let global = n.max(1).div_ceil(local) * local;
        let mut args: Vec<Arg> = reads.iter().map(|b| Arg::Buffer(*b)).collect();
        args.extend(write_bufs.iter().map(|b| Arg::Buffer(*b)));
        self.launches.push(Launch {
            kernel: kname.clone(),
            args,
            global,
            local,
            note: format!(
                "{} element{} ({} group{} of {})",
                n,
                if n == 1 { "" } else { "s" },
                global / local,
                if global / local == 1 { "" } else { "s" },
                local
            ),
        });
        self.kernels.push(KernelInfo {
            name: kname,
            computes: group,
            writes: writes.clone(),
            elements: n,
            about: String::new(),
        });
        // The values written now live in buffers; the rest were registers and are gone.
        for (id, buf) in writes.iter().zip(write_bufs) {
            self.places.insert(*id, Place::Buffer(buf));
        }
        for id in in_group {
            if !writes.contains(&id) {
                self.places.remove(&id);
            }
        }
    }

    /// A take or a drop: the `keep` items from `start` along its axis
    /// of every line (`outer` lines, `len` items along the axis,
    /// `inner` after it), one work-item each; along the first axis the
    /// kept items are one contiguous run.
    fn slice(&mut self, id: ValueId, arg: ValueId, s: Slice) {
        let elem = self.elem_of(arg);
        let Place::Buffer(src) = self.places[&arg].clone() else {
            unreachable!("a slice reads a buffer")
        };
        let kname = format!("slice_{elem}");
        if !self.source.contains(&format!("__kernel void {kname}(")) {
            self.source.push_str(&format!(
                "
// {kname}: a take or a drop along an axis: of every line (outer lines
// of len items, inner apart), the keep items from start; one
// work-item per item kept.
__kernel void {kname}(__global const {elem}* in, __global {elem}* out, const long outer, const long len, const long inner, const long start, const long keep) {{
    long k = get_global_id(0);
    if (k >= outer * keep * inner) return;
    long o = k / (keep * inner), r = k % (keep * inner);
    long j = r / inner, i = r % inner;
    out[k] = in[(o * len + start + j) * inner + i];
}}
"
            ));
            self.kernels.push(KernelInfo {
                name: kname.clone(),
                computes: vec![],
                writes: vec![],
                elements: 0,
                about: about(&kname),
            });
        }
        let count = s.outer * s.keep * s.inner;
        let result = self.buffer(Some(id), elem, count, Init::Device);
        let local = self.schedule.work_group.min(count.max(1).next_power_of_two());
        let global = count.max(1).div_ceil(local) * local;
        let name = self.checked.program.value(id).name.clone();
        self.launches.push(Launch {
            kernel: kname.clone(),
            args: vec![
                Arg::Buffer(src),
                Arg::Buffer(result),
                Arg::Long(s.outer as i64),
                Arg::Long(s.len as i64),
                Arg::Long(s.inner as i64),
                Arg::Long(s.start as i64),
                Arg::Long(s.keep as i64),
            ],
            global,
            local,
            note: if s.is_contiguous() {
                format!("%{name}: {count} items from item {}", s.start * s.inner)
            } else {
                format!(
                    "%{name}: {} of {} along the axis, {} line{} of {} each: {count} items",
                    s.keep,
                    s.len,
                    s.outer,
                    if s.outer == 1 { "" } else { "s" },
                    s.inner
                )
            },
        });
        if let Some(k) = self.kernels.iter_mut().find(|k| k.name == kname) {
            k.computes.push(id);
            k.writes.push(id);
        }
        self.places.insert(id, Place::Buffer(result));
    }

    /// A rotate: a gather, one work-item per result item; the amounts
    /// (one copy each along a new first axis when there are several)
    /// are a constant table in the source.
    fn rotate(&mut self, id: ValueId, arg: ValueId, axis: usize, amounts: &[i64]) {
        let elem = self.elem_of(arg);
        let Place::Buffer(src) = self.places[&arg].clone() else {
            unreachable!("a rotate reads a buffer (rank 1 or more)")
        };
        let (outer, len, inner) = xetal_gpu_xir::interp::axis_split(&self.ty(arg).shape, axis);
        let copies = amounts.len();
        let per = outer * len * inner;
        let n = copies * per;
        let kname = format!("rotate{}", self.launches.len());
        let result = self.buffer(Some(id), elem, n, Init::Device);
        let name = self.checked.program.value(id).name.clone();
        let src_name = self.buffers[src.0].name.clone();
        let out_name = self.buffers[result.0].name.clone();
        let amounts_text = amounts.iter().map(|a| format!("{a}L")).collect::<Vec<_>>().join(", ");
        let how = format!("{} o_-_{axis}", amounts.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(" "));
        self.source.push_str(&format!(
            "
// {kname}: %{name} := {how} ({copies} cop{}, {outer} line{} of {len} along the axis, {inner} apart); a gather, one work-item per result item.
__constant long {kname}_amounts[{copies}] = {{ {amounts_text} }};
__kernel void {kname}(__global const {elem}* {src_name}, __global {elem}* {out_name}) {{
    long k = get_global_id(0);
    if (k >= {n}) return;
    long c = k / {per}, r = k % {per};
    long o = r / ({len} * {inner}), q = r % ({len} * {inner});
    long j = q / {inner}, i = q % {inner};
    long s = xetal_imod_long(j + {kname}_amounts[c], {len});
    {out_name}[k] = {src_name}[(o * {len} + s) * {inner} + i];
}}
",
            if copies == 1 { "y" } else { "ies" },
            if outer == 1 { "" } else { "s" },
        ));
        let total = n.max(1);
        let local = self.schedule.work_group.min(total.next_power_of_two());
        self.launches.push(Launch {
            kernel: kname.clone(),
            args: vec![Arg::Buffer(src), Arg::Buffer(result)],
            global: total.div_ceil(local) * local,
            local,
            note: format!(
                "%{name}: {how}, {copies} cop{} of {per} items gathered",
                if copies == 1 { "y" } else { "ies" }
            ),
        });
        self.kernels.push(KernelInfo {
            name: kname,
            computes: vec![id],
            writes: vec![id],
            elements: n,
            about: String::new(),
        });
        self.places.insert(id, Place::Buffer(result));
    }

    /// A reshape to a different count: the argument's items cycled
    /// (`cycle_float`), or a single value written everywhere.
    fn cycle(&mut self, id: ValueId, arg: ValueId) {
        let elem = self.elem_of(arg);
        let n = self.ty(id).shape.len();
        let name = self.checked.program.value(id).name.clone();
        let result = self.buffer(Some(id), elem, n, Init::Device);
        let out_name = self.buffers[result.0].name.clone();
        let total = n.max(1);
        let local = self.schedule.work_group.min(total.next_power_of_two());
        let global = total.div_ceil(local) * local;
        match self.places[&arg].clone() {
            Place::Literal(lit) => {
                let kname = format!("fill{}", self.launches.len());
                self.source.push_str(&format!(
                    "
// {kname}: %{name} := a single value reshaped: written to every item.
__kernel void {kname}(__global {elem}* {out_name}) {{
    long k = get_global_id(0);
    if (k < {n}) {out_name}[k] = {lit};
}}
"
                ));
                self.launches.push(Launch {
                    kernel: kname.clone(),
                    args: vec![Arg::Buffer(result)],
                    global,
                    local,
                    note: format!("%{name}: {n} items, one value written to each"),
                });
                self.kernels.push(KernelInfo {
                    name: kname,
                    computes: vec![id],
                    writes: vec![id],
                    elements: n,
                    about: String::new(),
                });
            }
            Place::Buffer(src) => {
                let m = self.ty(arg).shape.len();
                let kname = format!("cycle_{elem}");
                if !self.source.contains(&format!("__kernel void {kname}(")) {
                    self.source.push_str(&format!(
                        "
// {kname}: a reshape to another count: the m items repeated as needed
// over n (r_eshape cycles its items), one work-item per result item.
__kernel void {kname}(__global const {elem}* in, __global {elem}* out, const long m, const long n) {{
    long k = get_global_id(0);
    if (k < n) out[k] = in[k % m];
}}
"
                    ));
                    self.kernels.push(KernelInfo {
                        name: kname.clone(),
                        computes: vec![],
                        writes: vec![],
                        elements: 0,
                        about: about(&kname),
                    });
                }
                self.launches.push(Launch {
                    kernel: kname.clone(),
                    args: vec![Arg::Buffer(src), Arg::Buffer(result), Arg::Long(m as i64), Arg::Long(n as i64)],
                    global,
                    local,
                    note: format!("%{name}: {m} items cycled over {n}"),
                });
                if let Some(k) = self.kernels.iter_mut().find(|k| k.name == kname) {
                    k.computes.push(id);
                    k.writes.push(id);
                }
            }
            Place::Register(_) => unreachable!("flushed before a reshape"),
        }
        self.places.insert(id, Place::Buffer(result));
    }

    /// A table: one work-item per result item (i, j), the function of
    /// a's item i and b's item j.
    fn table(&mut self, id: ValueId, op: TableOp, a: ValueId, b: ValueId) {
        let (na, nb) = (self.ty(a).shape.len(), self.ty(b).shape.len());
        let elem = self.elem_of(id);
        let read = |s: &Self, v: ValueId, idx: &str| -> String {
            match &s.places[&v] {
                Place::Buffer(buf) => format!("{}[{idx}]", s.buffers[buf.0].name),
                Place::Literal(l) => l.clone(),
                Place::Register(_) => unreachable!("flushed before a table"),
            }
        };
        let (xa, xb) = (read(self, a, "k / nb"), read(self, b, "k % nb"));
        let expr = match op {
            TableOp::Left => xa.clone(),
            TableOp::Right => xb.clone(),
            TableOp::Map(m) => map_expr(m, &[xa.clone(), xb.clone()], self.elem_of(a)),
        };
        let kname = format!("table{}", self.launches.len());
        let mut params = Vec::new();
        let mut args = Vec::new();
        for v in [a, b] {
            if let Place::Buffer(buf) = self.places[&v].clone() {
                let bb = &self.buffers[buf.0];
                let p = format!("__global const {}* {}", bb.elem, bb.name);
                if !params.contains(&p) {
                    params.push(p);
                    args.push(Arg::Buffer(buf));
                }
            }
        }
        let result = self.buffer(Some(id), elem, na * nb, Init::Device);
        let out_name = self.buffers[result.0].name.clone();
        params.push(format!("__global {elem}* {out_name}"));
        args.push(Arg::Buffer(result));
        let name = self.checked.program.value(id).name.clone();
        self.source.push_str(&format!(
            "
// {kname}: %{name} := a '{op} t_able b, {na} by {nb} items, one work-item each.
__kernel void {kname}({}) {{
    long k = get_global_id(0);
    const long nb = {nb};
    if (k >= {total}) return;
    {out_name}[k] = {expr};
}}
",
            params.join(", "),
            total = na * nb
        ));
        let total = (na * nb).max(1);
        let local = self.schedule.work_group.min(total.next_power_of_two());
        self.launches.push(Launch {
            kernel: kname.clone(),
            args,
            global: total.div_ceil(local) * local,
            local,
            note: format!("%{name}: {op} table, {na} by {nb} items"),
        });
        self.kernels.push(KernelInfo {
            name: kname,
            computes: vec![id],
            writes: vec![id],
            elements: na * nb,
            about: String::new(),
        });
        self.places.insert(id, Place::Buffer(result));
    }

    /// A transpose that moves axes: a gather, one work-item per result
    /// item, its position taken apart into the result's coordinates and
    /// put together again with the argument's strides.
    fn transpose(&mut self, id: ValueId, arg: ValueId, perm: &[usize]) {
        let elem = self.elem_of(arg);
        let Place::Buffer(src) = self.places[&arg].clone() else {
            unreachable!("a transpose that moves axes reads a buffer (rank 2 or more)")
        };
        let shape = self.ty(arg).shape.clone();
        let (out, strides) = transpose_layout(&shape, perm);
        let n = out.len();
        let rank = perm.len();
        let swaps_last_two = rank >= 2
            && perm[..rank - 2].iter().enumerate().all(|(i, &a)| a == i + 1)
            && perm[rank - 2] == rank
            && perm[rank - 1] == rank - 1;
        if self.schedule.tile > 0 && swaps_last_two {
            return self.transpose_tiled(id, arg, src, &shape, &out);
        }
        let kname = format!("transpose{}", self.launches.len());
        let mut body = String::new();
        for j in (0..out.rank()).rev() {
            if j == 0 {
                body.push_str(&format!("    src += r * {};\n", strides[0]));
            } else {
                body.push_str(&format!("    src += (r % {}) * {}; r /= {};\n", out.0[j], strides[j], out.0[j]));
            }
        }
        let result = self.buffer(Some(id), elem, n, Init::Device);
        let name = self.checked.program.value(id).name.clone();
        let src_name = self.buffers[src.0].name.clone();
        let out_name = self.buffers[result.0].name.clone();
        let reversed = perm.iter().enumerate().all(|(i, &a)| a == perm.len() - i);
        let how = if reversed {
            "o_\\, the axes reversed".to_string()
        } else {
            format!("{} t_ranspose", perm.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(" "))
        };
        self.source.push_str(&format!(
            "
// {kname}: %{name} := {how}, {shape} to {out}; a gather, one work-item per result item.
__kernel void {kname}(__global const {elem}* {src_name}, __global {elem}* {out_name}) {{
    long k = get_global_id(0);
    if (k >= {n}) return;
    long r = k, src = 0;
{body}    {out_name}[k] = {src_name}[src];
}}
"
        ));
        let total = n.max(1);
        let local = self.schedule.work_group.min(total.next_power_of_two());
        self.launches.push(Launch {
            kernel: kname.clone(),
            args: vec![Arg::Buffer(src), Arg::Buffer(result)],
            global: total.div_ceil(local) * local,
            local,
            note: format!(
                "%{name}: {how}, {shape} to {out}, {n} item{} gathered",
                if n == 1 { "" } else { "s" }
            ),
        });
        self.kernels.push(KernelInfo {
            name: kname,
            computes: vec![id],
            writes: vec![id],
            elements: n,
            about: String::new(),
        });
        self.places.insert(id, Place::Buffer(result));
    }

    /// A transpose of the last two axes (`o_\` of a matrix, `o_\_23`),
    /// in T by T tiles: a work-group stages a block of the argument in
    /// local memory, reading along the rows, and writes it out along the
    /// result's rows, so both sides of the copy are contiguous (the
    /// gather above reads a column: every work-item of a group a row
    /// apart, which the measured 4096 by 4096 case showed nine times
    /// slower than a copy on the Apple M1 Max).
    fn transpose_tiled(&mut self, id: ValueId, arg: ValueId, src: BufferId, shape: &Shape, out: &Shape) {
        let elem = self.elem_of(arg);
        let t = self.schedule.tile;
        let rank = shape.rank();
        let (rows, cols) = (shape.0[rank - 2], shape.0[rank - 1]);
        let planes: usize = shape.0[..rank - 2].iter().product();
        let kname = format!("transpose_tiled{t}_{elem}");
        if !self.source.contains(&format!("__kernel void {kname}(")) {
            self.source.push_str(&transpose_tiled_kernel(&kname, elem, t));
            self.kernels.push(KernelInfo {
                name: kname.clone(),
                computes: vec![],
                writes: vec![],
                elements: 0,
                about: about(&kname),
            });
        }
        let result = self.buffer(Some(id), elem, out.len(), Init::Device);
        let groups = planes * rows.div_ceil(t) * cols.div_ceil(t);
        let name = self.checked.program.value(id).name.clone();
        self.launches.push(Launch {
            kernel: kname.clone(),
            args: vec![
                Arg::Buffer(src),
                Arg::Buffer(result),
                Arg::Long(planes as i64),
                Arg::Long(rows as i64),
                Arg::Long(cols as i64),
            ],
            global: groups.max(1) * t * t,
            local: t * t,
            note: format!(
                "%{name}: the last two axes swapped in {t} by {t} tiles, {shape} to {out}: {groups} work-group{} of {}",
                if groups == 1 { "" } else { "s" },
                t * t
            ),
        });
        if let Some(k) = self.kernels.iter_mut().find(|k| k.name == kname) {
            k.computes.push(id);
            k.writes.push(id);
        }
        self.places.insert(id, Place::Buffer(result));
    }

    /// An inner product: one work-item per result item, its sum folded
    /// from the right (the evaluator's order); with a tile, the operands
    /// staged through local memory a T by T block at a time, the same
    /// order kept.
    fn matmul(&mut self, id: ValueId, a: ValueId, b: ValueId) {
        let elem = self.elem_of(a);
        let (Place::Buffer(ba), Place::Buffer(bb)) = (self.places[&a].clone(), self.places[&b].clone()) else {
            unreachable!("matmul reads buffers")
        };
        let (m, n, p) = xetal_gpu_xir::interp::matmul_split(&self.ty(a).shape, &self.ty(b).shape);
        let t = self.schedule.tile;
        debug_assert!(self.schedule.validate().is_ok(), "a schedule is validated before planning");
        let kname = if t == 0 {
            format!("matmul_{elem}")
        } else {
            format!("matmul_tiled{t}_{elem}")
        };
        if !self.source.contains(&format!("__kernel void {kname}(")) {
            self.source.push_str(&if t == 0 {
                matmul_kernel(&kname, elem)
            } else {
                matmul_tiled_kernel(&kname, elem, t)
            });
            self.kernels.push(KernelInfo {
                name: kname.clone(),
                computes: vec![],
                writes: vec![],
                elements: 0,
                about: about(&kname),
            });
        }
        let result = self.buffer(Some(id), elem, m * p, Init::Device);
        let args = vec![
            Arg::Buffer(ba),
            Arg::Buffer(bb),
            Arg::Buffer(result),
            Arg::Long(m as i64),
            Arg::Long(n as i64),
            Arg::Long(p as i64),
        ];
        let name = self.checked.program.value(id).name.clone();
        let (global, local, note) = if t == 0 {
            let total = (m * p).max(1);
            let local = self.schedule.work_group.min(total.next_power_of_two());
            (
                total.div_ceil(local) * local,
                local,
                format!(
                    "%{name}: {m} by {n} times {n} by {p}, {} sums of {n} products, one work-item each",
                    m * p
                ),
            )
        } else {
            let groups = m.div_ceil(t) * p.div_ceil(t);
            (
                groups * t * t,
                t * t,
                format!(
                    "%{name}: {m} by {n} times {n} by {p} in {t} by {t} tiles: {groups} work-group{} of {}",
                    if groups == 1 { "" } else { "s" },
                    t * t
                ),
            )
        };
        self.launches.push(Launch {
            kernel: kname.clone(),
            args,
            global,
            local,
            note,
        });
        if let Some(k) = self.kernels.iter_mut().find(|k| k.name == kname) {
            k.computes.push(id);
            k.writes.push(id);
        }
        self.places.insert(id, Place::Buffer(result));
    }

    /// A reduction along an axis: one work-item per result item, each
    /// folding its line from the right, as the evaluator does (so the
    /// result is the evaluator's exactly, in the device's width).
    fn reduce_axis(&mut self, id: ValueId, op: ReduceOp, arg: ValueId, axis: usize) {
        let elem = self.elem_of(arg);
        let Place::Buffer(src) = self.places[&arg].clone() else {
            unreachable!("a reduce reads a buffer")
        };
        let (outer, len, inner) = xetal_gpu_xir::interp::axis_split(&self.ty(arg).shape, axis);
        let kname = format!("reduce_axis_{}_{}", op, elem);
        if !self.source.contains(&format!("__kernel void {kname}(")) {
            self.source.push_str(&reduce_axis_kernel(&kname, op, elem));
            self.kernels.push(KernelInfo {
                name: kname.clone(),
                computes: vec![],
                writes: vec![],
                elements: 0,
                about: about(&kname),
            });
        }
        let n = outer * inner;
        let result = self.buffer(Some(id), elem, n, Init::Device);
        let local = self.schedule.work_group.min(n.max(1).next_power_of_two());
        let global = n.max(1).div_ceil(local) * local;
        self.launches.push(Launch {
            kernel: kname.clone(),
            args: vec![
                Arg::Buffer(src),
                Arg::Buffer(result),
                Arg::Long(outer as i64),
                Arg::Long(len as i64),
                Arg::Long(inner as i64),
            ],
            global,
            local,
            note: format!(
                "%{}: {} along axis {}: {} line{} of {}, one work-item each",
                self.checked.program.value(id).name,
                op,
                axis,
                n,
                if n == 1 { "" } else { "s" },
                len
            ),
        });
        if let Some(k) = self.kernels.iter_mut().find(|k| k.name == kname) {
            k.computes.push(id);
            k.writes.push(id);
        }
        self.places.insert(id, Place::Buffer(result));
    }

    /// A reduction: a tree in local memory per work-group, one partial
    /// per group, launched again on the partials until one is left.
    fn reduce(&mut self, id: ValueId, op: ReduceOp, arg: ValueId) {
        let elem = self.elem_of(arg);
        let Place::Buffer(src) = self.places[&arg].clone() else {
            unreachable!("a reduce reads a buffer")
        };
        let n = self.ty(arg).shape.len();
        let kname = format!("reduce_{}_{}", op, elem);
        if !self.source.contains(&format!("__kernel void {kname}(")) {
            self.source.push_str(&reduce_kernel(&kname, op, elem));
            self.kernels.push(KernelInfo {
                name: kname.clone(),
                computes: vec![],
                writes: vec![],
                elements: 0,
                about: about(&kname),
            });
        }
        let local = self.schedule.work_group;
        let result = self.buffer(Some(id), elem, 1, Init::Device);
        let mut input = src;
        let mut count = n;
        let mut pass = 0;
        let mut partials: Vec<BufferId> = Vec::new();
        loop {
            let groups = count.div_ceil(local);
            let out = if groups == 1 {
                result
            } else {
                // Alternate between two partial buffers.
                if partials.len() < 2 {
                    let p = self.buffer(None, elem, groups, Init::Device);
                    partials.push(p);
                }
                partials[pass % 2]
            };
            self.launches.push(Launch {
                kernel: kname.clone(),
                args: vec![
                    Arg::Buffer(input),
                    Arg::Buffer(out),
                    Arg::Long(count as i64),
                    Arg::Local(local * elem.bytes()),
                ],
                global: groups * local,
                local,
                note: format!(
                    "%{}: {} {} item{} to {} partial{} (pass {})",
                    self.checked.program.value(id).name,
                    op,
                    count,
                    if count == 1 { "" } else { "s" },
                    groups,
                    if groups == 1 { "" } else { "s" },
                    pass + 1
                ),
            });
            if groups == 1 {
                break;
            }
            input = out;
            count = groups;
            pass += 1;
        }
        if let Some(k) = self.kernels.iter_mut().find(|k| k.name == kname) {
            k.computes.push(id);
            k.writes.push(id);
        }
        self.places.insert(id, Place::Buffer(result));
    }
}

/// What a shared kernel does, by its name, for `explain`.
fn about(kname: &str) -> String {
    if kname.starts_with("transpose_tiled") {
        "the last two axes swapped in tiles staged in local memory, read and written along the rows".into()
    } else if kname.starts_with("reduce_axis_") {
        "along one axis, one work-item per result item folding its line from the right".into()
    } else if kname.starts_with("reduce_") {
        "a tree reduction in local memory, one partial per work-group".into()
    } else if kname.starts_with("matmul_tiled") {
        "sums of products in tiles staged in local memory, folded from the right".into()
    } else if kname.starts_with("matmul_") {
        "sums of products, one work-item per result item, folded from the right".into()
    } else if kname.starts_with("slice_") {
        "a take or a drop along an axis: the items kept of every line, one work-item per item".into()
    } else if kname.starts_with("cycle_") {
        "a reshape to another count: the items repeated as needed, one work-item per item".into()
    } else {
        String::new()
    }
}

/// The C for a map on already-read operands.
fn map_expr(op: MapOp, a: &[String], arg_elem: Elem) -> String {
    let f = arg_elem.is_float();
    match op {
        MapOp::Add => format!("({} + {})", a[0], a[1]),
        MapOp::Sub => format!("({} - {})", a[0], a[1]),
        MapOp::Mul => format!("({} * {})", a[0], a[1]),
        MapOp::Div => format!("({} / {})", a[0], a[1]),
        MapOp::IDiv => format!("xetal_idiv_{}({}, {})", arg_elem, a[0], a[1]),
        MapOp::Mod => format!("xetal_imod_{}({}, {})", arg_elem, a[0], a[1]),
        MapOp::Min => format!("{}({}, {})", if f { "fmin" } else { "min" }, a[0], a[1]),
        MapOp::Max => format!("{}({}, {})", if f { "fmax" } else { "max" }, a[0], a[1]),
        MapOp::Neg => format!("(-{})", a[0]),
        MapOp::Abs => {
            if f {
                format!("fabs({})", a[0])
            } else {
                format!("({} < 0 ? -{} : {})", a[0], a[0], a[0])
            }
        }
        MapOp::Exp => format!("exp({})", a[0]),
        MapOp::Log => format!("log({})", a[0]),
        MapOp::Eq => format!("({} == {})", a[0], a[1]),
        MapOp::Ne => format!("({} != {})", a[0], a[1]),
        MapOp::Lt => format!("({} < {})", a[0], a[1]),
        MapOp::Le => format!("({} <= {})", a[0], a[1]),
        MapOp::Gt => format!("({} > {})", a[0], a[1]),
        MapOp::Ge => format!("({} >= {})", a[0], a[1]),
        MapOp::And => format!("({} && {})", a[0], a[1]),
        MapOp::Or => format!("({} || {})", a[0], a[1]),
        MapOp::Not => format!("(!{})", a[0]),
    }
}

/// The C for a cast: Float to Int toward negative infinity, a number to Bool when not 0.
fn cast_expr(x: &str, from: Scalar, from_elem: Elem, to: Scalar, to_elem: Elem) -> String {
    match (from, to) {
        (_, Scalar::Bool) if from != Scalar::Bool => format!("({x} != 0)"),
        (f, t) if f.is_float() && t.is_int() => format!("({to_elem})floor({x})"),
        _ if from_elem == to_elem => x.to_string(),
        _ => format!("({to_elem}){x}"),
    }
}

/// A single-value constant as a C literal.
fn literal(c: &Const, elem: Elem) -> String {
    match c {
        Const::Int(v) => match elem {
            Elem::I64 => format!("{}L", v[0]),
            _ => format!("{}", v[0]),
        },
        Const::Float(v) => float_literal(v[0], elem),
        Const::Bool(v) => if v[0] { "1" } else { "0" }.to_string(),
    }
}

fn float_literal(x: f64, elem: Elem) -> String {
    if x.is_nan() {
        return "NAN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "INFINITY".into() } else { "(-INFINITY)".into() };
    }
    let mut s = format!("{x}");
    if !s.contains('.') && !s.contains('e') {
        s.push_str(".0");
    }
    if elem == Elem::F32 {
        s.push('f');
    }
    if x < 0.0 {
        format!("({s})")
    } else {
        s
    }
}

fn combine(op: ReduceOp, elem: Elem, x: &str, acc: &str) -> String {
    let f = elem.is_float();
    match op {
        ReduceOp::Add => format!("{x} + {acc}"),
        ReduceOp::Mul => format!("{x} * {acc}"),
        ReduceOp::Min => format!("{}({x}, {acc})", if f { "fmin" } else { "min" }),
        ReduceOp::Max => format!("{}({x}, {acc})", if f { "fmax" } else { "max" }),
    }
}

fn matmul_kernel(name: &str, elem: Elem) -> String {
    format!(
        "
// {name}: a by b, a's last axis (n) with b's first; one work-item per
// result item (i, j), its n products summed from the last to the first,
// as X_eTaL's '+ '* i_nner sums them.
__kernel void {name}(__global const {elem}* a, __global const {elem}* b, __global {elem}* c, const long m, const long n, const long p) {{
    long ij = get_global_id(0);
    if (ij >= m * p) return;
    long i = ij / p, j = ij % p;
    {elem} acc = a[i * n + n - 1] * b[(n - 1) * p + j];
    for (long k = n - 2; k >= 0; k--) acc = a[i * n + k] * b[k * p + j] + acc;
    c[ij] = acc;
}}
"
    )
}

fn matmul_tiled_kernel(name: &str, elem: Elem, t: usize) -> String {
    format!(
        "
// {name}: a by b in {t} by {t} tiles: each work-group computes a {t} by {t}
// block of the result, staging a block of a and of b in local memory per
// step; the steps go from the last block of the axis to the first and
// each block's products from the last to the first, so every sum is
// folded from the right, as X_eTaL's '+ '* i_nner folds it.
__kernel void {name}(__global const {elem}* a, __global const {elem}* b, __global {elem}* c, const long m, const long n, const long p) {{
    __local {elem} sa[{t}][{t}];
    __local {elem} sb[{t}][{t}];
    long lid = get_local_id(0);
    long ty = lid / {t}, tx = lid % {t};
    long blocksp = (p + {t} - 1) / {t};
    long g = get_group_id(0);
    long i = (g / blocksp) * {t} + ty, j = (g % blocksp) * {t} + tx;
    {elem} acc = 0;
    int started = 0;
    for (long s = (n + {t} - 1) / {t} - 1; s >= 0; s--) {{
        long ka = s * {t} + tx, kb = s * {t} + ty;
        sa[ty][tx] = (i < m && ka < n) ? a[i * n + ka] : 0;
        sb[ty][tx] = (kb < n && j < p) ? b[kb * p + j] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (long kk = {t} - 1; kk >= 0; kk--) {{
            if (s * {t} + kk < n) {{
                {elem} prod = sa[ty][kk] * sb[kk][tx];
                acc = started ? prod + acc : prod;
                started = 1;
            }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    if (i < m && j < p) c[i * p + j] = acc;
}}
"
    )
}

fn transpose_tiled_kernel(name: &str, elem: Elem, t: usize) -> String {
    format!(
        "
// {name}: the last two axes of a (planes of rows by cols) swapped in
// {t} by {t} tiles: each work-group reads a block along the rows into
// local memory and writes it along the result's rows, so both the read
// and the write are contiguous. The padding column keeps the local
// reads off one bank.
__kernel void {name}(__global const {elem}* in, __global {elem}* out, const long planes, const long rows, const long cols) {{
    __local {elem} tile[{t}][{t} + 1];
    // Block arithmetic in int: a 64-bit division is emulated on a GPU.
    int lid = get_local_id(0);
    int ty = lid / {t}, tx = lid % {t};
    int nr = (int)rows, nc = (int)cols;
    int brows = (nr + {t} - 1) / {t}, bcols = (nc + {t} - 1) / {t};
    int g = get_group_id(0);
    int plane = g / (brows * bcols), rem = g % (brows * bcols);
    int br = rem / bcols, bc = rem % bcols;
    int r = br * {t} + ty, c = bc * {t} + tx;
    long base = (long)plane * rows * cols;
    if (plane < planes && r < nr && c < nc) tile[ty][tx] = in[base + (long)r * cols + c];
    barrier(CLK_LOCAL_MEM_FENCE);
    int orow = bc * {t} + ty, ocol = br * {t} + tx;
    if (plane < planes && orow < nc && ocol < nr) out[base + (long)orow * rows + ocol] = tile[tx][ty];
}}
"
    )
}

fn reduce_axis_kernel(name: &str, op: ReduceOp, elem: Elem) -> String {
    let step = combine(op, elem, "in[base + j * inner]", "acc");
    format!(
        "
// {name}: along one axis, one work-item per result item; its line of
// len items (stride inner) folded from the right, as X_eTaL's r_/ does.
__kernel void {name}(__global const {elem}* in, __global {elem}* out, const long outer, const long len, const long inner) {{
    long k = get_global_id(0);
    if (k >= outer * inner) return;
    long o = k / inner, i = k % inner;
    long base = o * len * inner + i;
    {elem} acc = in[base + (len - 1) * inner];
    for (long j = len - 2; j >= 0; j--) acc = {step};
    out[k] = acc;
}}
"
    )
}

fn reduce_kernel(name: &str, op: ReduceOp, elem: Elem) -> String {
    let f = elem.is_float();
    let identity = match (op, elem) {
        (ReduceOp::Add, _) => "0".to_string(),
        (ReduceOp::Mul, _) => "1".to_string(),
        (ReduceOp::Min, Elem::F32 | Elem::F64) => "INFINITY".into(),
        (ReduceOp::Max, Elem::F32 | Elem::F64) => "(-INFINITY)".into(),
        (ReduceOp::Min, Elem::I32) => "INT_MAX".into(),
        (ReduceOp::Max, Elem::I32) => "INT_MIN".into(),
        (ReduceOp::Min, Elem::I64) => "LONG_MAX".into(),
        (ReduceOp::Max, Elem::I64) => "LONG_MIN".into(),
    };
    let combine = match op {
        ReduceOp::Add => "s[lid] + s[lid + h]".to_string(),
        ReduceOp::Mul => "s[lid] * s[lid + h]".to_string(),
        ReduceOp::Min => format!("{}(s[lid], s[lid + h])", if f { "fmin" } else { "min" }),
        ReduceOp::Max => format!("{}(s[lid], s[lid + h])", if f { "fmax" } else { "max" }),
    };
    format!(
        "
// {name}: a tree reduction in local memory, one partial per work-group;
// launched again on the partials until one value is left.
__kernel void {name}(__global const {elem}* in, __global {elem}* out, const long n, __local {elem}* s) {{
    size_t gid = get_global_id(0), lid = get_local_id(0), ls = get_local_size(0);
    s[lid] = gid < (size_t)n ? in[gid] : {identity};
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t h = ls / 2; h > 0; h >>= 1) {{
        if (lid < h) s[lid] = {combine};
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    if (lid == 0) out[get_group_id(0)] = s[0];
}}
"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Width;
    use xetal_gpu_xir::{check, text};

    fn plan_of(src: &str, schedule: &Schedule) -> Plan {
        plan(&check(text::parse(src).unwrap()).unwrap(), schedule)
    }

    #[test]
    fn fuses_a_chain_into_one_kernel_and_writes_only_what_is_needed() {
        let p = plan_of(
            "%a = const i64 [4] 1 2 3 4\n%k = const i64 [] 10\n%b = map mul %k %a\n%c = map add %a %b\n%n = map neg %c\noutput %n\n",
            &Schedule::default(),
        );
        assert_eq!(p.kernels.len(), 1);
        assert_eq!(p.kernels[0].computes.len(), 3);
        assert_eq!(p.kernels[0].writes, vec![ValueId(4)]);
        assert_eq!(p.buffers.len(), 2); // %a (const) and %n (written); %k is a literal, %b and %c registers
        assert!(p.source.contains("long t_b = (10L * b_a[i]);"));
        assert!(p.source.contains("b_n[i] = t_n;"));
        assert_eq!(p.launches.len(), 1);
        assert_eq!((p.launches[0].global, p.launches[0].local), (4, 4));
    }

    #[test]
    fn a_value_read_by_a_later_kernel_is_written() {
        let p = plan_of(
            "%a = const f64 [8] 1 2 3 4 5 6 7 8\n%two = const f64 [] 2.0\n%b = map mul %two %a\n%r = reduce add %b\n%c = map add %b %r\noutput %c\n",
            &Schedule::default(),
        );
        // k0 computes %b (written: the reduce and k1 read it), reduce, k1 computes %c reading b_b[i] and b_r[0].
        assert_eq!(
            p.kernels.iter().map(|k| k.name.as_str()).collect::<Vec<_>>(),
            ["k0", "reduce_add_float", "k1"]
        );
        assert!(p.source.contains("float t_b = (2.0f * b_a[i]);"));
        assert!(p.source.contains("float t_c = (b_b[i] + b_r[0]);"));
        assert_eq!(p.launches.len(), 3);
    }

    #[test]
    fn a_large_reduce_takes_several_passes() {
        let src = format!("%a = const i64 [{}] 1\n%r = reduce add %a\noutput %r\n", 70_000);
        let p = plan_of(&src, &Schedule::default());
        let passes: Vec<(usize, usize)> = p.launches.iter().map(|l| (l.global, l.local)).collect();
        // 70000 -> 274 partials -> 2 partials -> 1
        assert_eq!(passes, [(274 * 256, 256), (2 * 256, 256), (256, 256)]);
        assert_eq!(p.buffers.iter().filter(|b| b.value.is_none()).count(), 2);
        assert!(p.source.contains("__kernel void reduce_add_long("));
    }

    #[test]
    fn an_axis_reduce_is_one_work_item_per_result() {
        let p = plan_of(
            "%m = const f64 [3 5] 1.0\n%r = reduce add axis=2 %m\n%c = reduce max axis=1 %m\noutput %r\noutput %c\n",
            &Schedule::default(),
        );
        assert_eq!(p.launches.len(), 2);
        assert_eq!(p.launches[0].args[2..], [Arg::Long(3), Arg::Long(5), Arg::Long(1)]);
        assert_eq!(p.launches[1].args[2..], [Arg::Long(1), Arg::Long(3), Arg::Long(5)]);
        assert_eq!((p.launches[0].global, p.launches[0].local), (4, 4));
        assert!(p
            .source
            .contains("for (long j = len - 2; j >= 0; j--) acc = in[base + j * inner] + acc;"));
        assert!(p.source.contains("acc = fmax(in[base + j * inner], acc);"));
        // Along the only axis of a vector: the tree.
        let p = plan_of(
            "%v = const i64 [4] 1 2 3 4\n%r = reduce add axis=1 %v\noutput %r\n",
            &Schedule::default(),
        );
        assert!(p.source.contains("__kernel void reduce_add_long("));
    }

    #[test]
    fn inner_products_untiled_and_tiled() {
        let src = "%a = const f64 [5 3] 1.0\n%b = const f64 [3 7] 1.0\n%c = matmul %a %b\noutput %c\n";
        let p = plan_of(src, &Schedule::default());
        assert_eq!(p.launches[0].args[3..], [Arg::Long(5), Arg::Long(3), Arg::Long(7)]);
        assert_eq!((p.launches[0].global, p.launches[0].local), (64, 64));
        assert!(p
            .source
            .contains("for (long k = n - 2; k >= 0; k--) acc = a[i * n + k] * b[k * p + j] + acc;"));
        let p = plan_of(
            src,
            &Schedule {
                tile: 4,
                ..Schedule::default()
            },
        );
        // 5 by 7 in 4 by 4 tiles: 2 by 2 blocks, 16 work-items each.
        assert_eq!((p.launches[0].global, p.launches[0].local), (64, 16));
        assert!(p.source.contains("__kernel void matmul_tiled4_float("));
        assert!(p.source.contains("__local float sa[4][4];"));
    }

    #[test]
    fn structure_kernels() {
        let p = plan_of(
            "%wb = const f64 [4 2] 1.0\n%w = drop -1 %wb\n%b = take -1 %wb\n%br = ravel %b\n%o = const i64 [3] 0 1 2\n%bias = table right %o %br\n%sc = const f64 [] 2.0\n%t = table mul %br %sc\noutput %w\noutput %bias\noutput %t\n",
            &Schedule::default(),
        );
        // drop -1 of [4 2]: one line of 4 along the axis, 2 apart, 3 kept from 0; take -1: 1 kept from 3.
        assert_eq!(
            p.launches[0].args[2..],
            [Arg::Long(1), Arg::Long(4), Arg::Long(2), Arg::Long(0), Arg::Long(3)]
        );
        assert_eq!(
            p.launches[1].args[2..],
            [Arg::Long(1), Arg::Long(4), Arg::Long(2), Arg::Long(3), Arg::Long(1)]
        );
        assert_eq!(p.launches[0].note, "%w: 6 items from item 0");
        assert!(p.source.contains("b_bias[k] = b_b[k % nb];"));
        assert!(p.source.contains("b_t[k] = (b_b[k / nb] * 2.0f);"));
        // ravel is the same buffer: no launch of its own.
        assert_eq!(p.launches.len(), 4);
    }

    #[test]
    fn a_transpose_is_a_gather_and_an_identity_is_nothing() {
        let p = plan_of(
            "%t = const i64 [2 3 4] 1\n%r = transpose %t\n%p = transpose 3 1 2 %t\n%v = const f64 [3] 1.0\n%vt = transpose %v\n%m = const i64 [2 3] 1\n%i = transpose 1 2 %m\n%k = const f64 [] 2.0\n%kt = transpose %k\noutput %r\noutput %p\noutput %vt\noutput %i\noutput %kt\n",
            &Schedule::default(),
        );
        // o_\ t: [2 3 4] to [4 3 2], the argument's strides 12 4 1 reversed.
        assert!(p
            .source
            .contains("    src += (r % 2) * 12; r /= 2;\n    src += (r % 3) * 4; r /= 3;\n    src += r * 1;\n    b_r[k] = b_t[src];"));
        // 3 1 2 t_ranspose t: [3 4 2]; axis 1 (stride 12) to the end, 2 (4) first, 3 (1) second.
        assert!(p
            .source
            .contains("    src += (r % 2) * 12; r /= 2;\n    src += (r % 4) * 1; r /= 4;\n    src += r * 4;\n    b_p[k] = b_t[src];"));
        // The vector, the identity permutation and the single value: no launch, the argument's buffer.
        assert_eq!(p.launches.len(), 2);
        assert_eq!((p.launches[0].global, p.launches[0].local), (32, 32));
        assert_eq!(
            p.launches[0].note,
            "%r: o_\\, the axes reversed, [2 3 4] to [4 3 2], 24 items gathered"
        );
        assert_eq!(p.launches[1].note, "%p: 3 1 2 t_ranspose, [2 3 4] to [3 4 2], 24 items gathered");
        let holder = |v: usize| p.buffer(p.outputs.iter().find(|(o, _)| o.0 == v).unwrap().1).name.clone();
        assert_eq!((holder(4), holder(6), holder(8)), ("b_v".into(), "b_m".into(), "b_kt".into()));
    }

    #[test]
    fn a_transpose_of_the_last_two_axes_is_tiled_when_the_schedule_tiles() {
        let sched = Schedule {
            tile: 4,
            ..Schedule::default()
        };
        let p = plan_of(
            "%m = const f64 [5 7] 1.0\n%mt = transpose %m\n%t = const i64 [2 3 4] 1\n%ts = transpose 1 3 2 %t\n%tr = transpose %t\noutput %mt\noutput %ts\noutput %tr\n",
            &sched,
        );
        // [5 7]: 2 by 2 blocks of 16 work-items; [2 3 4] with axes 2 and 3 swapped: 2 planes of 1 by 1 blocks.
        assert_eq!(p.launches[0].args[2..], [Arg::Long(1), Arg::Long(5), Arg::Long(7)]);
        assert_eq!((p.launches[0].global, p.launches[0].local), (64, 16));
        assert_eq!(p.launches[1].args[2..], [Arg::Long(2), Arg::Long(3), Arg::Long(4)]);
        assert_eq!((p.launches[1].global, p.launches[1].local), (32, 16));
        assert!(p.source.contains("__kernel void transpose_tiled4_float("));
        assert!(p.source.contains("__kernel void transpose_tiled4_long("));
        // The axes reversed on rank 3 is not a swap of the last two: the gather.
        assert!(p.launches[2].kernel.starts_with("transpose") && !p.launches[2].kernel.contains("tiled"));
        assert!(p.source.contains("__local float tile[4][4 + 1];"));
        assert!(p.launches[0]
            .note
            .starts_with("%mt: the last two axes swapped in 4 by 4 tiles, [5 7] to [7 5]: 4 work-groups of 16"));
    }

    #[test]
    fn rotate_slice_along_an_axis_and_reshape() {
        let p = plan_of(
            "%x = const i64 [1 3 4] 1\n%r = rotate axis=3 -1 0 1 %x\n%d = drop -1 axis=4 %r\n%s = reshape [3 4] %x\n%c = reshape [2 12] %x\n%k = const f64 [] 2.0\n%f = reshape [5] %k\noutput %d\noutput %s\noutput %c\noutput %f\n",
            &Schedule::default(),
        );
        assert!(p.source.contains("__constant long rotate0_amounts[3] = { -1L, 0L, 1L };"));
        assert!(p.source.contains("long s = xetal_imod_long(j + rotate0_amounts[c], 4);"));
        assert_eq!(p.launches[0].note, "%r: -1 0 1 o_-_3, 3 copies of 12 items gathered");
        // drop -1 along axis 4 of [3 1 3 4]: 9 lines of 4, 1 apart, 3 kept from 0.
        assert_eq!(
            p.launches[1].args[2..],
            [Arg::Long(9), Arg::Long(4), Arg::Long(1), Arg::Long(0), Arg::Long(3)]
        );
        assert_eq!(p.launches[1].note, "%d: 3 of 4 along the axis, 9 lines of 1 each: 27 items");
        // The same count: no launch; another count: the cycle kernel; a single value: a fill.
        assert_eq!(p.launches.len(), 4);
        assert_eq!(p.launches[2].kernel, "cycle_long");
        assert_eq!(p.launches[2].args[2..], [Arg::Long(12), Arg::Long(24)]);
        assert!(p.source.contains("if (k < 5) b_f[k] = 2.0f;"));
        let holder = |v: usize| p.buffer(p.outputs.iter().find(|(o, _)| o.0 == v).unwrap().1).name.clone();
        assert_eq!(holder(3), "b_x");
    }

    #[test]
    fn widths_follow_the_schedule() {
        let src = "%a = const f64 [2] 1.5 2.5\n%i = const i64 [2] 1 2\n%k = const f64 [] 0.5\n%b = map mul %k %a\n%j = map mod %i %i\n%f = cast f64 %i\noutput %b\noutput %j\noutput %f\n";
        let p = plan_of(src, &Schedule::default());
        assert!(p.source.contains("float t_b = (0.5f * b_a[i]);"));
        assert!(p.source.contains("long t_j = xetal_imod_long(b_i[i], b_i[i]);"));
        assert!(p.source.contains("float t_f = (float)b_i[i];"));
        assert!(!p.source.contains("cl_khr_fp64"));
        let p = plan_of(
            src,
            &Schedule {
                float: Width::W64,
                int: Width::W32,
                work_group: 64,
                tile: 0,
            },
        );
        assert!(p.source.starts_with("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"));
        assert!(p.source.contains("double t_b = (0.5 * b_a[i]);"));
        assert!(p.source.contains("int t_j = xetal_imod_int(b_i[i], b_i[i]);"));
        assert!(p.source.contains("double t_f = (double)b_i[i];"));
    }

    #[test]
    fn expressions() {
        let e = |src: &str| plan_of(src, &Schedule::default()).source;
        let s = e("%x = const f64 [2] -0.75 1.0\n%t = const f64 [] 0.5\n%m = map gt %x %t\n%mf = cast f64 %m\n%z = const f64 [] 0.0\n%s = select %m %x %z\n%ab = map abs %x\n%ex = map exp %x\n%fl = cast i64 %x\noutput %mf\noutput %s\noutput %ab\noutput %ex\noutput %fl\n");
        assert!(s.contains("int t_m = (b_x[i] > 0.5f);"));
        assert!(s.contains("float t_mf = (float)t_m;"));
        assert!(s.contains("float t_s = (t_m ? b_x[i] : 0.0f);"));
        assert!(s.contains("float t_ab = fabs(b_x[i]);"));
        assert!(s.contains("float t_ex = exp(b_x[i]);"));
        assert!(s.contains("long t_fl = (long)floor(b_x[i]);"));
        let s = e(
            "%a = const i64 [2] -3 7\n%ab = map abs %a\n%b = map max %a %ab\n%bb = cast bool %a\n%n = map not %bb\noutput %b\noutput %n\n",
        );
        assert!(s.contains("long t_ab = (b_a[i] < 0 ? -b_a[i] : b_a[i]);"));
        assert!(s.contains("long t_b = max(b_a[i], t_ab);"));
        assert!(s.contains("int t_bb = (b_a[i] != 0);"));
        assert!(s.contains("int t_n = (!t_bb);"));
    }

    #[test]
    fn literals() {
        assert_eq!(float_literal(2.0, Elem::F32), "2.0f");
        assert_eq!(float_literal(-0.5, Elem::F64), "(-0.5)");
        assert_eq!(float_literal(1e21, Elem::F64), "1000000000000000000000.0");
        assert_eq!(float_literal(f64::INFINITY, Elem::F32), "INFINITY");
        assert_eq!(literal(&Const::Int(vec![-3]), Elem::I64), "-3L");
        assert_eq!(literal(&Const::Int(vec![-3]), Elem::I32), "-3");
    }

    #[test]
    fn a_constant_output_gets_a_buffer() {
        let p = plan_of("%k = const i64 [] 7\noutput %k\n", &Schedule::default());
        assert_eq!(p.buffers.len(), 1);
        assert_eq!(p.outputs, vec![(ValueId(0), BufferId(0))]);
        assert!(p.launches.is_empty());
    }
}
