//! Type and shape checking: every value gets a type, or the program
//! is refused with a message naming the value.
//!
//! Rules (X_eTaL's, for the subset):
//! - a map's arguments have one scalar type, as its kind allows
//!   (`Num`: any number; `Float`: `/`, `exp`, `log` are Float only;
//!   `Int`: `idiv`, `mod`; comparisons give Bool; logic is Bool only);
//! - shapes agree exactly, except that a rank-0 argument extends
//!   to every item of the others (scalar extension);
//! - `select` takes a Bool condition and two values of one type;
//! - `reduce` folds a numeric array of rank 1 or more to rank 0;
//! - `cast` keeps the shape and changes the scalar type.

use crate::ir::*;
use crate::{err, Result};

/// A checked program: the program and the type of every value.
#[derive(Debug, Clone, PartialEq)]
pub struct Checked {
    pub program: Program,
    pub types: Vec<Ty>,
}

impl Checked {
    pub fn ty(&self, id: ValueId) -> &Ty {
        &self.types[id.0]
    }
}

/// Check a program; on success every value has its type.
pub fn check(program: Program) -> Result<Checked> {
    let mut types: Vec<Ty> = Vec::with_capacity(program.values.len());
    for (i, v) in program.values.iter().enumerate() {
        let name = &v.name;
        let ty_of = |id: ValueId| -> &Ty { &types[id.0] };
        let ty = match &v.op {
            Op::Input => v.ty.clone().expect("an input declares its type"),
            Op::Const(c) => {
                let ty = v.ty.clone().expect("a const declares its type");
                let ok = match c {
                    Const::Int(_) => ty.scalar.is_int(),
                    Const::Float(_) => ty.scalar.is_float(),
                    Const::Bool(_) => ty.scalar == Scalar::Bool,
                };
                if !ok {
                    return err(format!("%{name}: the items do not fit {}", ty.scalar));
                }
                ty
            }
            Op::Map { op, args } => {
                let tys: Vec<&Ty> = args.iter().map(|a| ty_of(*a)).collect();
                let sc = tys[0].scalar;
                if let Some(t) = tys.iter().find(|t| t.scalar != sc) {
                    return err(format!(
                        "%{name}: map {op} needs arguments of one scalar type, got {} and {}",
                        sc, t.scalar
                    ));
                }
                let result = match op.kind() {
                    MapKind::Num => {
                        require(name, op, sc.is_num(), sc, "a number")?;
                        sc
                    }
                    MapKind::Float => {
                        require(name, op, sc.is_float(), sc, "a Float (f32 or f64; cast an Int first)")?;
                        sc
                    }
                    MapKind::Int => {
                        require(name, op, sc.is_int(), sc, "an Int (i32 or i64)")?;
                        sc
                    }
                    MapKind::EqCompare => Scalar::Bool,
                    MapKind::OrdCompare => {
                        require(name, op, sc.is_num(), sc, "a number")?;
                        Scalar::Bool
                    }
                    MapKind::Logic => {
                        require(name, op, sc == Scalar::Bool, sc, "a Bool")?;
                        Scalar::Bool
                    }
                };
                let shape = extend(name, &format!("map {op}"), &tys.iter().map(|t| &t.shape).collect::<Vec<_>>())?;
                Ty { scalar: result, shape }
            }
            Op::Select { cond, a, b } => {
                let (tc, ta, tb) = (ty_of(*cond), ty_of(*a), ty_of(*b));
                if tc.scalar != Scalar::Bool {
                    return err(format!("%{name}: select needs a Bool condition, got {}", tc.scalar));
                }
                if ta.scalar != tb.scalar {
                    return err(format!(
                        "%{name}: select needs both values of one scalar type, got {} and {}",
                        ta.scalar, tb.scalar
                    ));
                }
                let shape = extend(name, "select", &[&tc.shape, &ta.shape, &tb.shape])?;
                Ty { scalar: ta.scalar, shape }
            }
            Op::Reduce { op, arg, axis } => {
                let t = ty_of(*arg);
                if !t.scalar.is_num() {
                    return err(format!("%{name}: reduce {op} needs numbers, got {}", t.scalar));
                }
                if t.shape.is_scalar() {
                    return err(format!("%{name}: reduce {op} needs an array of rank 1 or more, got a single value"));
                }
                let shape = match axis {
                    None if t.shape.is_empty() => return err(format!("%{name}: reduce {op} of an empty array has no value")),
                    None => Shape::scalar(),
                    Some(a) if *a > t.shape.rank() => {
                        return err(format!("%{name}: reduce {op} axis={a}: the array has rank {}", t.shape.rank()));
                    }
                    Some(a) => {
                        let mut dims = t.shape.0.clone();
                        if dims[a - 1] == 0 {
                            return err(format!("%{name}: reduce {op} axis={a}: that axis is empty, so there is no value"));
                        }
                        dims.remove(a - 1);
                        Shape(dims)
                    }
                };
                Ty { scalar: t.scalar, shape }
            }
            Op::Cast { op, arg } => Ty {
                scalar: op.to,
                shape: ty_of(*arg).shape.clone(),
            },
        };
        if let Some(declared) = &v.ty {
            if *declared != ty {
                return err(format!("%{name}: declared {declared}, computed {ty}"));
            }
        }
        debug_assert_eq!(types.len(), i);
        types.push(ty);
    }
    if program.outputs.is_empty() {
        return err("the program has no output");
    }
    Ok(Checked { program, types })
}

fn require(name: &str, op: &MapOp, ok: bool, got: Scalar, wanted: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        err(format!("%{name}: map {op} needs {wanted}, got {got}"))
    }
}

/// Scalar extension: the one shape the arguments share, a rank-0
/// argument extending to it.
fn extend(name: &str, what: &str, shapes: &[&Shape]) -> Result<Shape> {
    let mut result: Option<&Shape> = None;
    for s in shapes {
        if s.is_scalar() {
            continue;
        }
        match result {
            None => result = Some(s),
            Some(r) if r == *s => {}
            Some(r) => return err(format!("%{name}: {what}: shapes differ, {r} and {s} (only a single value extends)")),
        }
    }
    Ok(result.cloned().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    fn types_of(src: &str) -> Vec<String> {
        check(parse(src).unwrap()).unwrap().types.iter().map(|t| t.to_string()).collect()
    }

    fn error_of(src: &str) -> String {
        check(parse(src).unwrap()).unwrap_err().0
    }

    #[test]
    fn infers_types_and_shapes() {
        let t = types_of(
            "%a = const i64 [4] 1 2 3 4\n%k = const i64 [] 2\n%b = map mul %k %a\n%m = map gt %b %a\n%s = select %m %a %k\n%r = reduce add %s\n%f = cast f64 %r\n%e = map exp %f\noutput %e\n",
        );
        assert_eq!(
            t,
            ["i64 [4]", "i64 []", "i64 [4]", "bool [4]", "i64 [4]", "i64 []", "f64 []", "f64 []"]
        );
    }

    #[test]
    fn refuses_mismatches() {
        assert_eq!(
            error_of("%a = const i64 [2] 1 2\n%b = const f64 [2] 1.0 2.0\n%c = map add %a %b\noutput %c\n"),
            "%c: map add needs arguments of one scalar type, got i64 and f64"
        );
        assert_eq!(
            error_of("%a = const i64 [2] 1 2\n%c = map div %a %a\noutput %c\n"),
            "%c: map div needs a Float (f32 or f64; cast an Int first), got i64"
        );
        assert_eq!(
            error_of("%a = const f64 [2] 1.0 2.0\n%c = map mod %a %a\noutput %c\n"),
            "%c: map mod needs an Int (i32 or i64), got f64"
        );
        assert_eq!(
            error_of("%a = const i64 [2] 1 2\n%c = map and %a %a\noutput %c\n"),
            "%c: map and needs a Bool, got i64"
        );
        assert_eq!(
            error_of("%a = const i64 [2] 1 2\n%b = const i64 [3] 1 2 3\n%c = map add %a %b\noutput %c\n"),
            "%c: map add: shapes differ, [2] and [3] (only a single value extends)"
        );
        assert_eq!(
            error_of("%a = const i64 [2] 1 2\n%c = select %a %a %a\noutput %c\n"),
            "%c: select needs a Bool condition, got i64"
        );
        assert_eq!(
            error_of("%a = const i64 [] 1\n%c = reduce add %a\noutput %c\n"),
            "%c: reduce add needs an array of rank 1 or more, got a single value"
        );
        assert_eq!(
            error_of("%a = const bool [2] 1 0\n%c = reduce add %a\noutput %c\n"),
            "%c: reduce add needs numbers, got bool"
        );
        assert_eq!(error_of("%a = const i64 [2] 1 2\n"), "the program has no output");
        assert_eq!(
            error_of("%a = const i64 [0]\n%c = reduce add %a\noutput %c\n"),
            "%c: reduce add of an empty array has no value"
        );
    }

    #[test]
    fn reduces_along_an_axis() {
        let t = types_of("%m = const i64 [2 3] 1\n%r = reduce add axis=2 %m\n%c = reduce add axis=1 %m\n%w = reduce add %m\n%v = reduce max axis=1 %r\noutput %v\n");
        assert_eq!(t, ["i64 [2 3]", "i64 [2]", "i64 [3]", "i64 []", "i64 []"]);
        assert_eq!(
            error_of("%m = const i64 [2 3] 1\n%r = reduce add axis=3 %m\noutput %r\n"),
            "%r: reduce add axis=3: the array has rank 2"
        );
        assert_eq!(
            error_of("%m = const i64 [2 0]\n%r = reduce add axis=2 %m\noutput %r\n"),
            "%r: reduce add axis=2: that axis is empty, so there is no value"
        );
    }

    #[test]
    fn compares_anything_for_equality() {
        assert_eq!(
            types_of("%a = const bool [2] 1 0\n%c = map eq %a %a\noutput %c\n"),
            ["bool [2]", "bool [2]"]
        );
    }
}
