//! Structured regions: one work-item owns the local state and executes
//! its loops without intermediate dispatches or host guard readbacks.
use std::collections::HashMap;
use std::fmt::Write;

use super::{cast_expr, literal, map_expr, Builder, Place};
use crate::{Arg, Elem, GuardCheck, Init, KernelInfo, Launch};
use xetal_gpu_xir::{check, Checked, Const, MapOp, Op, Program, RegionKind, ValueId};

#[derive(Clone)]
struct Read {
    base: String,
    scalar: bool,
    literal: bool,
}
impl Read {
    fn at(&self, index: &str) -> String {
        if self.literal {
            self.base.clone()
        } else {
            format!("{}[{}]", self.base, if self.scalar { "0" } else { index })
        }
    }
}

struct Local<'a> {
    schedule: &'a crate::Schedule,
    serial: usize,
    codes: Vec<String>,
    source: String,
}
impl Local<'_> {
    fn fresh(&mut self) -> String {
        self.serial += 1;
        format!("r{}", self.serial)
    }
    fn fail(&mut self, condition: &str, message: String) {
        self.codes.push(message);
        let _ = writeln!(self.source, "if ({condition}) {{ status[i] = {}; return; }}", self.codes.len());
    }
    fn region(&mut self, kind: RegionKind, body: &Program, args: &[Read]) -> Read {
        let checked = check(body.clone()).expect("checked region");
        if kind == RegionKind::Each {
            return self.body(&checked, args);
        }
        let first_ty = checked.ty(body.inputs()[0]);
        let state = self.fresh();
        let elem = self.schedule.elem(first_ty.scalar);
        let n = first_ty.shape.len();
        let _ = writeln!(self.source, "{elem} {state}[{}];", n.max(1));
        let _ = writeln!(self.source, "for (size_t j=0; j<{n}; ++j) {state}[j] = {};", args[0].at("j"));
        let mut refs = args.to_vec();
        refs[0] = Read {
            base: state.clone(),
            scalar: first_ty.shape.is_scalar(),
            literal: false,
        };
        if let RegionKind::Repeat(count) = kind {
            let iter = self.fresh();
            let _ = writeln!(self.source, "for (size_t {iter}=0; {iter}<{count}; ++{iter}) {{");
        } else {
            self.source.push_str("{\n");
        }
        let output = self.body(&checked, &refs);
        let _ = writeln!(self.source, "for (size_t j=0; j<{n}; ++j) {state}[j] = {};", output.at("j"));
        self.source.push_str("}\n");
        refs[0].clone()
    }
    fn body(&mut self, checked: &Checked, inputs: &[Read]) -> Read {
        let p = &checked.program;
        let mut refs = HashMap::new();
        let mut input = 0;
        for (offset, value) in p.values.iter().enumerate() {
            let id = ValueId(offset);
            let ty = checked.ty(id);
            if matches!(value.op, Op::Input) {
                refs.insert(id, inputs[input].clone());
                input += 1;
                continue;
            }
            if let Op::Region { kind, args, body } = &value.op {
                let args: Vec<Read> = args.iter().map(|a| refs[a].clone()).collect();
                // Nested Each maps a vector locally. Outer Each supplies GPU
                // parallelism; nested maps are deliberately sequential here.
                let result = if *kind == RegionKind::Each {
                    let name = self.fresh();
                    let elem = self.schedule.elem(ty.scalar);
                    let _ = writeln!(self.source, "{elem} {name}[{}];", ty.shape.len().max(1));
                    let j = self.fresh();
                    let _ = writeln!(self.source, "for (size_t {j}=0; {j}<{}; ++{j}) {{", ty.shape.len());
                    let item = self.fresh();
                    let _ = writeln!(
                        self.source,
                        "{} {item}[1] = {{ {} }};",
                        self.schedule.elem(checked.ty(value.op.args()[0]).scalar),
                        args[0].at(&j)
                    );
                    let mut local_args = args.clone();
                    local_args[0] = Read {
                        base: item,
                        scalar: true,
                        literal: false,
                    };
                    let result = self.region(RegionKind::Each, body, &local_args);
                    let _ = writeln!(self.source, "{name}[{j}] = {};\n}}", result.at("0"));
                    Read {
                        base: name,
                        scalar: false,
                        literal: false,
                    }
                } else {
                    self.region(*kind, body, &args)
                };
                refs.insert(id, result);
                continue;
            }
            let name = self.fresh();
            let n = ty.shape.len();
            let elem = self.schedule.elem(ty.scalar);
            let read = |arg: ValueId, at: &str| refs[&arg].at(at);
            if let Op::Guard { cond, message } = &value.op {
                self.fail(&read(*cond, "0"), message.clone());
                refs.insert(
                    id,
                    Read {
                        base: "0".into(),
                        scalar: true,
                        literal: true,
                    },
                );
                continue;
            }
            if let Op::Const(c) = &value.op {
                let values: Vec<String> = (0..n.max(1))
                    .map(|j| {
                        let filled;
                        let c = if matches!(c, Const::Int(v) if v.is_empty())
                            || matches!(c, Const::Float(v) if v.is_empty())
                            || matches!(c, Const::Bool(v) if v.is_empty())
                        {
                            filled = c.empty_fill();
                            &filled
                        } else {
                            c
                        };
                        let c = match c {
                            Const::Int(v) => Const::Int(vec![*v.get(j).unwrap_or(&v[0])]),
                            Const::Float(v) => Const::Float(vec![*v.get(j).unwrap_or(&v[0])]),
                            Const::Bool(v) => Const::Bool(vec![*v.get(j).unwrap_or(&v[0])]),
                        };
                        literal(&c, elem)
                    })
                    .collect();
                let _ = writeln!(self.source, "{elem} {name}[{}] = {{ {} }};", n.max(1), values.join(", "));
                refs.insert(
                    id,
                    Read {
                        base: name,
                        scalar: ty.shape.is_scalar(),
                        literal: false,
                    },
                );
                continue;
            }
            let _ = writeln!(self.source, "{elem} {name}[{}];", n.max(1));
            let _ = writeln!(self.source, "for (size_t j=0; j<{n}; ++j) {{");
            let expression = match &value.op {
                Op::Map { op, args } => {
                    let reads: Vec<String> = args.iter().map(|arg| read(*arg, "j")).collect();
                    if matches!(op, MapOp::IDiv | MapOp::Mod) {
                        self.fail(
                            &format!("{} == 0", reads[1]),
                            format!("%{}: map {op}: division by zero", value.name),
                        );
                    }
                    map_expr(*op, &reads, self.schedule.elem(checked.ty(args[0]).scalar))
                }
                Op::Select { cond, a, b } => format!("({} ? {} : {})", read(*cond, "j"), read(*a, "j"), read(*b, "j")),
                Op::Cast { op, arg } => cast_expr(
                    &read(*arg, "j"),
                    checked.ty(*arg).scalar,
                    self.schedule.elem(checked.ty(*arg).scalar),
                    op.to,
                    elem,
                ),
                Op::Ravel { arg } => read(*arg, "j"),
                Op::Reshape { arg, .. } => read(*arg, &format!("j % {}", checked.ty(*arg).shape.len().max(1))),
                Op::Take { arg, .. } | Op::Drop { arg, .. } => {
                    let s = xetal_gpu_xir::interp::slice_of(&value.op, &checked.ty(*arg).shape);
                    read(
                        *arg,
                        &format!(
                            "(j / {} * {} + {} + j / {} % {}) * {} + j % {}",
                            s.keep * s.inner,
                            s.len,
                            s.start,
                            s.inner,
                            s.keep.max(1),
                            s.inner,
                            s.inner
                        ),
                    )
                }
                Op::Lookup { axis, idx, arg } => {
                    let shape = &checked.ty(*arg).shape.0;
                    let len = shape[axis - 1];
                    let inner: usize = shape[*axis..].iter().product();
                    let count = checked.ty(*idx).shape.len();
                    let position = read(*idx, &format!("j / {inner} % {}", count.max(1)));
                    self.fail(
                        &format!("({position}) < 1 || ({position}) > {len}"),
                        format!("%{}: region lookup index out of range 1..={len}", value.name),
                    );
                    read(
                        *arg,
                        &format!(
                            "(j / {} * {len} + (size_t)({position}) - 1) * {inner} + j % {inner}",
                            (count * inner).max(1)
                        ),
                    )
                }
                Op::Concat { axis, a, b } => {
                    let ta = checked.ty(*a);
                    let tb = checked.ty(*b);
                    let da = if ta.shape.is_scalar() { 1 } else { ta.shape.0[axis - 1] };
                    let db = if tb.shape.is_scalar() { 1 } else { tb.shape.0[axis - 1] };
                    let inner: usize = if ta.shape.is_scalar() {
                        1
                    } else {
                        ta.shape.0[*axis..].iter().product()
                    };
                    let width = (da + db) * inner;
                    let k = format!("j % {width}");
                    format!(
                        "({k} < {} ? {} : {})",
                        da * inner,
                        read(*a, &format!("j / {width} * {} + {k}", da * inner)),
                        read(*b, &format!("j / {width} * {} + {k} - {}", db * inner, da * inner))
                    )
                }
                Op::Reduce { op, arg, axis } => {
                    let t = checked.ty(*arg);
                    let (len, inner) = match axis {
                        Some(a) => (t.shape.0[a - 1], t.shape.0[*a..].iter().product()),
                        None => (t.shape.len(), 1),
                    };
                    let position = |k: &str| format!("(j / {inner} * {len} + {k}) * {inner} + j % {inner}");
                    let acc = self.fresh();
                    let _ = writeln!(self.source, "{elem} {acc} = {};", read(*arg, &position(&(len - 1).to_string())));
                    let _ = writeln!(self.source, "for (size_t k={}; k>0; --k) {{", len - 1);
                    let lhs = read(*arg, &position("(k-1)"));
                    let expression = map_expr(
                        match op {
                            xetal_gpu_xir::ReduceOp::Add => MapOp::Add,
                            xetal_gpu_xir::ReduceOp::Mul => MapOp::Mul,
                            xetal_gpu_xir::ReduceOp::Min => MapOp::Min,
                            xetal_gpu_xir::ReduceOp::Max => MapOp::Max,
                        },
                        &[lhs, acc.clone()],
                        elem,
                    );
                    let _ = writeln!(self.source, "{acc} = {expression};\n}}");
                    acc
                }
                Op::Table { op, a, b } => {
                    let nb = checked.ty(*b).shape.len().max(1);
                    let (a, b) = (read(*a, &format!("j / {nb}")), read(*b, &format!("j % {nb}")));
                    match op {
                        xetal_gpu_xir::TableOp::Left => a,
                        xetal_gpu_xir::TableOp::Right => b,
                        xetal_gpu_xir::TableOp::Map(op) => {
                            if matches!(op, MapOp::IDiv | MapOp::Mod) {
                                self.fail(&format!("{b} == 0"), format!("%{}: map {op}: division by zero", value.name));
                            }
                            map_expr(*op, &[a, b], elem)
                        }
                    }
                }
                Op::Rotate { axis, amounts, arg } => {
                    let t = checked.ty(*arg);
                    let len = t.shape.0[axis - 1];
                    let inner: usize = t.shape.0[*axis..].iter().product();
                    let count = t.shape.len().max(1);
                    let mut amount = "0".to_string();
                    for (k, a) in amounts.iter().enumerate().rev() {
                        let a = a.rem_euclid(len.max(1) as i64);
                        amount = format!("(j / {count} == {k} ? {a} : {amount})");
                    }
                    let q = format!("(j % {count})");
                    read(
                        *arg,
                        &format!(
                            "({q} / {} * {len} + ({q} / {inner} + {amount}) % {}) * {inner} + {q} % {inner}",
                            (len * inner).max(1),
                            len.max(1)
                        ),
                    )
                }
                Op::Transpose { arg, perm } => {
                    let t = checked.ty(*arg);
                    let perm = xetal_gpu_xir::ir::permutation(t.shape.rank(), perm).unwrap();
                    let (shape, strides) = xetal_gpu_xir::ir::transpose_layout(&t.shape, &perm);
                    let mut div = shape.len();
                    let terms: Vec<String> = shape
                        .0
                        .iter()
                        .zip(strides)
                        .map(|(&d, stride)| {
                            div /= d.max(1);
                            format!("(j / {} % {}) * {stride}", div.max(1), d.max(1))
                        })
                        .collect();
                    read(*arg, &if terms.is_empty() { "0".into() } else { terms.join(" + ") })
                }
                // Operations not yet used by ray-local functions are refused by
                // the region checker, rather than emitting an incorrect shader.
                _ => unreachable!("unsupported local operation passed region capability check"),
            };
            let _ = writeln!(self.source, "{name}[j] = {expression};\n}}");
            refs.insert(
                id,
                Read {
                    base: name,
                    scalar: ty.shape.is_scalar(),
                    literal: false,
                },
            );
        }
        refs[&p.outputs[0]].clone()
    }
}

impl Builder<'_> {
    pub(super) fn region(&mut self, id: ValueId, kind: RegionKind, args: &[ValueId], body: &Program) {
        let lanes = if kind == RegionKind::Each {
            self.ty(args[0]).shape.len()
        } else {
            1
        };
        let mut params = Vec::new();
        let mut launch_args = Vec::new();
        let mut refs = Vec::new();
        for arg in args {
            let read = match &self.places[arg] {
                Place::Literal(s) => Read {
                    base: s.clone(),
                    scalar: true,
                    literal: true,
                },
                Place::Buffer(b) => {
                    let buf = &self.buffers[b.0];
                    if !launch_args.contains(&Arg::Buffer(*b)) {
                        params.push(format!("__global const {} *{}", buf.elem, buf.name));
                        launch_args.push(Arg::Buffer(*b));
                    }
                    Read {
                        base: buf.name.clone(),
                        scalar: self.ty(*arg).shape.is_scalar(),
                        literal: false,
                    }
                }
                Place::Register(_) => unreachable!("region boundary flushes"),
            };
            refs.push(read);
        }
        let mut local = Local {
            schedule: self.schedule,
            serial: 0,
            codes: vec![],
            source: String::new(),
        };
        if kind == RegionKind::Each {
            let elem = self.elem_of(args[0]);
            let _ = writeln!(local.source, "{elem} item[1] = {{ {} }};", refs[0].at("i"));
            refs[0] = Read {
                base: "item".into(),
                scalar: true,
                literal: false,
            };
        }
        let result = local.region(kind, body, &refs);
        let elem = self.elem_of(id);
        let output = self.buffer(Some(id), elem, self.ty(id).shape.len(), Init::Device);
        params.push(format!("__global {elem} *output"));
        launch_args.push(Arg::Buffer(output));
        let status = self.buffer(None, Elem::I32, lanes, Init::Device);
        params.push("__global int *status".into());
        launch_args.push(Arg::Buffer(status));
        let tail = if kind == RegionKind::Each {
            format!("output[i] = {};", result.at("0"))
        } else {
            format!(
                "for (size_t j=0; j<{}; ++j) output[j] = {};",
                self.ty(id).shape.len(),
                result.at("j")
            )
        };
        let name = format!("region_{}", self.checked.program.value(id).name);
        let _ = writeln!(
            self.source,
            "\n__kernel void {name}({}) {{\nsize_t i=get_global_id(0);\nif (i>={lanes}) return;\nstatus[i]=0;\n{}\n{tail}\n}}",
            params.join(", "),
            local.source
        );
        self.kernels.push(KernelInfo {
            name: name.clone(),
            computes: vec![id],
            writes: vec![id],
            elements: lanes,
            about: "structured local loops; one work-item per mapped item".into(),
        });
        let wg = self.schedule.work_group;
        self.launches.push(Launch {
            kernel: name,
            args: launch_args,
            global: lanes.max(1).div_ceil(wg) * wg,
            local: wg,
            note: "structured region".into(),
        });
        if !local.codes.is_empty() && lanes > 0 {
            self.guard_checks.push(GuardCheck {
                before_launch: self.launches.len(),
                buffer: status,
                message: "region validation".into(),
                codes: local.codes,
            });
        }
        self.places.insert(id, Place::Buffer(output));
    }
}
