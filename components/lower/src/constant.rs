//! Bounded partial evaluation by the pinned evaluator, only for closed,
//! pure expressions. Bound device inputs are never treated as constants.
use super::*;
use xetal_base::{NodeId, Span};

fn expr(kind: Kind) -> Expr {
    Expr {
        id: NodeId(0),
        span: Span::default(),
        kind,
    }
}
fn pure(e: &Expr, locals: &HashSet<String>, free: &mut HashSet<String>, globals: &mut HashSet<String>) -> bool {
    match &e.kind {
        Kind::Var(n) => {
            if !locals.contains(n) {
                free.insert(n.clone());
            }
            true
        }
        Kind::Global(n) => {
            globals.insert(n.clone());
            true
        }
        Kind::Prim(n) => !n.ends_with('!') && (!n.starts_with("[]") || n == "[]P_ANIC"),
        Kind::Lit(_) | Kind::Str(_) | Kind::Unit | Kind::NoMatch => true,
        Kind::Array(v) | Kind::Tuple(v) => v.iter().all(|e| pure(e, locals, free, globals)),
        Kind::Proj { tuple, .. } => pure(tuple, locals, free, globals),
        Kind::Axes { f, .. } => pure(f, locals, free, globals),
        Kind::Lam { param, body, .. } => {
            let mut scope = locals.clone();
            if let Param::Name(n) = param {
                scope.insert(n.clone());
            }
            pure(body, &scope, free, globals)
        }
        Kind::Let { name, value, body, rec } => {
            let mut scope = locals.clone();
            scope.insert(name.clone());
            pure(value, if *rec { &scope } else { locals }, free, globals) && pure(body, &scope, free, globals)
        }
        Kind::App(a, b) => pure(a, locals, free, globals) && pure(b, locals, free, globals),
        Kind::App2 { f, left, right } => {
            pure(f, locals, free, globals) && pure(left, locals, free, globals) && pure(right, locals, free, globals)
        }
        Kind::If { cond, then, other } => {
            pure(cond, locals, free, globals) && pure(then, locals, free, globals) && pure(other, locals, free, globals)
        }
        Kind::Set { .. } => false,
    }
}

impl Translator {
    pub(super) fn known(&self, v: &V) -> Option<Array> {
        match v {
            V::Static(a, _) => Some(a.clone()),
            V::Node(root) => {
                let mut reachable = HashSet::new();
                let mut pending = vec![*root];
                while let Some(id) = pending.pop() {
                    if !reachable.insert(id) {
                        continue;
                    }
                    let original = &self.program.value(id).op;
                    let mut op = original;
                    while let Op::Predicated { op: inner, .. } = op {
                        op = inner;
                    }
                    if matches!(op, Op::Input | Op::Guard { .. } | Op::Region { .. }) {
                        return None;
                    }
                    pending.extend(original.args());
                }
                let mut ordered: Vec<_> = reachable.into_iter().collect();
                ordered.sort_by_key(|id| id.0);
                let mut mapping = HashMap::new();
                let mut p = Program::default();
                for old in ordered {
                    let value = self.program.value(old);
                    let op = value.op.remap(|a| mapping[&a]);
                    let id = ValueId(p.values.len());
                    mapping.insert(old, id);
                    p.values.push(Value {
                        name: format!("v{}", id.0),
                        ty: value.ty.clone(),
                        op,
                    });
                }
                p.outputs.push(mapping[root]);
                interp::run(&check(p).ok()?, &HashMap::new()).ok()?.pop()
            }
            _ => None,
        }
    }
    fn snapshot(&self, v: &V) -> Option<Expr> {
        if let Some(a) = self.known(v) {
            let text = match &a.data {
                Data::Int(xs) => xs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "),
                Data::Float(xs) => {
                    if xs.iter().any(|x| !x.is_finite()) {
                        return None;
                    }
                    xs.iter().map(|x| xetal_gpu_xir::format::float(*x)).collect::<Vec<_>>().join(" ")
                }
                Data::Bool(xs) => format!(
                    "({}) = 1",
                    xs.iter().map(|x| if *x { "1" } else { "0" }).collect::<Vec<_>>().join(" ")
                ),
            };
            let fill = if a.scalar.is_float() {
                "0.0"
            } else if a.scalar == Scalar::Bool {
                "(0 = 1)"
            } else {
                "0"
            };
            let text = if a.shape.is_scalar() {
                text
            } else {
                format!(
                    "({}) r_eshape ({})",
                    a.shape.0.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(" "),
                    if a.shape.is_empty() { fill } else { &text }
                )
            };
            let p = xetal_core::lower(&text).ok()?;
            return p.items.into_iter().find_map(|i| if let Item::Eval(e) = i { Some(e) } else { None });
        }
        match v {
            V::Tuple(parts) => Some(expr(Kind::Tuple(parts.iter().map(|v| self.snapshot(v)).collect::<Option<_>>()?))),
            V::Str(s) => Some(expr(Kind::Str(s.clone()))),
            V::Unit => Some(expr(Kind::Unit)),
            V::Prim(p) if p.funs.is_empty() && p.axes.is_empty() => Some(expr(Kind::Prim(p.name.clone()))),
            _ => None,
        }
    }
    /// Evaluate a closed body using exactly the pinned evaluator. A fixed
    /// transition budget bounds compile-time work; effects and unknown
    /// device operands make this optimization ineligible, never guessed.
    pub(super) fn constant_body(&self, body: &Expr, env: &Env) -> Option<V> {
        self.constant_body_hint(body, env, None)
    }
    pub(super) fn constant_body_hint(&self, body: &Expr, env: &Env, hint: Option<Scalar>) -> Option<V> {
        let mut free = HashSet::new();
        let mut pending = HashSet::new();
        if !pure(body, &HashSet::new(), &mut free, &mut pending) {
            return None;
        }
        let mut e = body.clone();
        for name in free {
            let value = self.snapshot(env.get(&name)?)?;
            e = expr(Kind::Let {
                name,
                rec: false,
                value: Box::new(value),
                body: Box::new(e),
            });
        }
        let mut seen = HashSet::new();
        let mut defs = Vec::new();
        while let Some(name) = pending.iter().next().cloned() {
            pending.remove(&name);
            if !seen.insert(name.clone()) {
                continue;
            }
            let value = self.globals.get(&name)?;
            let mut free = HashSet::new();
            let mut dependencies = HashSet::new();
            if !pure(value, &HashSet::new(), &mut free, &mut dependencies) || !free.is_empty() {
                return None;
            }
            pending.extend(dependencies.into_iter().filter(|n| !seen.contains(n)));
            defs.push(Item::Def {
                name,
                value: value.clone(),
            });
        }
        defs.push(Item::Eval(e));
        let p = xetal_ir::Program {
            items: defs,
            notes: vec![],
            names: vec![],
        };
        let mut output = Vec::new();
        let mut result = None;
        let mut keep = |value: &xetal_value::Value<'_>| result = owned(value, hint);
        let status = xetal_step::Machine::new(&p, &mut output, xetal_arith::Rng::seeded(1))
            .hooks(Some(&mut keep), None)
            .run(1_000_000)
            .ok()?;
        if status == xetal_step::Status::Done {
            result
        } else {
            None
        }
    }
}
fn owned(v: &xetal_value::Value<'_>, hint: Option<Scalar>) -> Option<V> {
    use xetal_value::Value as E;
    if let E::Tuple(xs) = v {
        return Some(V::Tuple(xs.iter().map(|v| owned(v, hint)).collect::<Option<_>>()?));
    }
    if let E::Unit = v {
        return Some(V::Unit);
    }
    let a = xetal_value::as_array(v);
    let shape = Shape(a.shape().to_vec());
    // Empty numeric values do not carry a runtime scalar type. Retain the
    // ordinary typed lowering unless the primitive supplies its output type.
    if a.data().is_empty() {
        let scalar = hint?;
        let data = match scalar {
            Scalar::I32 | Scalar::I64 => Data::Int(vec![]),
            Scalar::F32 | Scalar::F64 => Data::Float(vec![]),
            Scalar::Bool => Data::Bool(vec![]),
        };
        return Some(V::Static(
            Array::new(scalar, shape, data).ok()?,
            scalar.is_int() || scalar == Scalar::Bool,
        ));
    }
    let first = a.data().first()?;
    let array = match first {
        E::Int(_) => Array::ints(
            shape,
            a.data()
                .iter()
                .map(|v| if let E::Int(x) = v { Some(*x) } else { None })
                .collect::<Option<_>>()?,
        ),
        E::Float(_) => Array::floats(
            shape,
            a.data()
                .iter()
                .map(|v| if let E::Float(x) = v { Some(*x) } else { None })
                .collect::<Option<_>>()?,
        ),
        E::Bool(_) => Array::new(
            Scalar::Bool,
            shape,
            Data::Bool(
                a.data()
                    .iter()
                    .map(|v| if let E::Bool(x) = v { Some(*x) } else { None })
                    .collect::<Option<_>>()?,
            ),
        )
        .ok()?,
        _ => return None,
    };
    let shapey = array.scalar.is_int() || array.scalar == Scalar::Bool;
    Some(V::Static(array, shapey))
}
