//! From a checked program and a schedule to a plan: grouping the
//! elementwise values into fused kernels, laying out the reductions,
//! and writing the OpenCL C.

use std::collections::{HashMap, HashSet};

use xetal_gpu_xir::{Checked, Const, MapOp, Op, ReduceOp, Scalar, Shape, Ty, ValueId};

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
            Op::Reduce { op, arg } => {
                b.flush(&outputs);
                b.reduce(id, *op, *arg);
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
                // An output that is a single constant: give it a buffer of its own.
                let ty = checked.ty(*o);
                let Op::Const(c) = &program.value(*o).op else { unreachable!() };
                let buf = b.buffer(Some(*o), b.schedule.elem(ty.scalar), 1, Init::Const(c.clone()));
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
