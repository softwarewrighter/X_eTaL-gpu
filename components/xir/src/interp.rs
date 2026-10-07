//! The reference interpreter: a checked program run on the host,
//! computing as X_eTaL's evaluator does (Int as i64, Float as f64,
//! floored `d_iv` and `m_od`, a reduce folded from the right). At
//! the narrower scalar types a device would use (i32, f32) every
//! result is rounded to that type, so the interpreter at f32 models
//! what a GPU computes in single precision.

use std::collections::HashMap;

use crate::check::Checked;
use crate::ir::*;
use crate::{err, Result};

/// An array's items.
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Bool(Vec<bool>),
}

impl Data {
    pub fn len(&self) -> usize {
        match self {
            Data::Int(v) => v.len(),
            Data::Float(v) => v.len(),
            Data::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A typed array: the scalar type, the shape and the items in row
/// order.
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    pub scalar: Scalar,
    pub shape: Shape,
    pub data: Data,
}

impl Array {
    /// An array of the type and items given, checking they fit.
    pub fn new(scalar: Scalar, shape: Shape, data: Data) -> Result<Array> {
        let fits = match (&data, scalar) {
            (Data::Int(_), s) => s.is_int(),
            (Data::Float(_), s) => s.is_float(),
            (Data::Bool(_), Scalar::Bool) => true,
            _ => false,
        };
        if !fits {
            return err(format!("the items are not {scalar}"));
        }
        if data.len() != shape.len() {
            return err(format!("shape {shape} holds {} items, got {}", shape.len(), data.len()));
        }
        let mut a = Array { scalar, shape, data };
        a.narrow();
        Ok(a)
    }

    pub fn ints(shape: Shape, items: Vec<i64>) -> Array {
        Array::new(Scalar::I64, shape, Data::Int(items)).expect("Int items")
    }

    pub fn floats(shape: Shape, items: Vec<f64>) -> Array {
        Array::new(Scalar::F64, shape, Data::Float(items)).expect("Float items")
    }

    pub fn ty(&self) -> Ty {
        Ty {
            scalar: self.scalar,
            shape: self.shape.clone(),
        }
    }

    /// Round every item to the array's scalar type (i32 wraps, f32
    /// rounds); nothing to do for i64, f64 and Bool.
    fn narrow(&mut self) {
        match (self.scalar, &mut self.data) {
            (Scalar::I32, Data::Int(v)) => v.iter_mut().for_each(|x| *x = *x as i32 as i64),
            (Scalar::F32, Data::Float(v)) => v.iter_mut().for_each(|x| *x = *x as f32 as f64),
            _ => {}
        }
    }

    /// Item i, or item 0 of a single value extended to every position.
    fn at(&self, i: usize) -> Item {
        let j = if self.shape.is_scalar() { 0 } else { i };
        match &self.data {
            Data::Int(v) => Item::Int(v[j]),
            Data::Float(v) => Item::Float(v[j]),
            Data::Bool(v) => Item::Bool(v[j]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Item {
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// Run a checked program with its inputs bound by name; the outputs
/// in order.
pub fn run(checked: &Checked, inputs: &HashMap<String, Array>) -> Result<Vec<Array>> {
    let program = &checked.program;
    let mut values: Vec<Array> = Vec::with_capacity(program.values.len());
    for (i, v) in program.values.iter().enumerate() {
        let ty = checked.ty(ValueId(i));
        let array = match &v.op {
            Op::Input => {
                let Some(given) = inputs.get(&v.name) else {
                    return err(format!("input %{} ({}) is not bound", v.name, ty));
                };
                if given.ty() != *ty {
                    return err(format!("input %{} is {}, bound to {}", v.name, ty, given.ty()));
                }
                given.clone()
            }
            Op::Const(c) => {
                let n = ty.shape.len();
                let data = match c {
                    Const::Int(items) => Data::Int(fill(items, n)),
                    Const::Float(items) => Data::Float(fill(items, n)),
                    Const::Bool(items) => Data::Bool(fill(items, n)),
                };
                Array::new(ty.scalar, ty.shape.clone(), data)?
            }
            Op::Map { op, args } => {
                let args: Vec<&Array> = args.iter().map(|a| &values[a.0]).collect();
                map(&v.name, *op, &args, ty)?
            }
            Op::Select { cond, a, b } => {
                let (c, a, b) = (&values[cond.0], &values[a.0], &values[b.0]);
                let n = ty.shape.len();
                let data = match &a.data {
                    Data::Int(_) => Data::Int((0..n).map(|i| pick(c.at(i), a.at(i), b.at(i)).int()).collect()),
                    Data::Float(_) => Data::Float((0..n).map(|i| pick(c.at(i), a.at(i), b.at(i)).float()).collect()),
                    Data::Bool(_) => Data::Bool((0..n).map(|i| pick(c.at(i), a.at(i), b.at(i)).boolean()).collect()),
                };
                Array::new(ty.scalar, ty.shape.clone(), data)?
            }
            Op::Reduce { op, arg, axis } => reduce(*op, &values[arg.0], *axis, ty)?,
            Op::Cast { op, arg } => cast(&values[arg.0], op.to)?,
        };
        values.push(array);
    }
    Ok(program.outputs.iter().map(|o| values[o.0].clone()).collect())
}

fn fill<T: Clone>(items: &[T], n: usize) -> Vec<T> {
    if items.len() == n {
        items.to_vec()
    } else {
        vec![items[0].clone(); n]
    }
}

fn pick(c: Item, a: Item, b: Item) -> Item {
    if c.boolean() {
        a
    } else {
        b
    }
}

impl Item {
    fn int(self) -> i64 {
        match self {
            Item::Int(x) => x,
            _ => unreachable!("checked: an Int"),
        }
    }

    fn float(self) -> f64 {
        match self {
            Item::Float(x) => x,
            _ => unreachable!("checked: a Float"),
        }
    }

    fn boolean(self) -> bool {
        match self {
            Item::Bool(x) => x,
            _ => unreachable!("checked: a Bool"),
        }
    }
}

/// Floored division, as X_eTaL's `d_iv` (`-7 d_iv 2` is -4, `7 d_iv -2` is -4).
pub fn idiv(a: i64, b: i64) -> i64 {
    let q = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// The remainder of floored division, as X_eTaL's `m_od` (`-3 m_od 2` is 1, `3 m_od -2` is -1).
pub fn imod(a: i64, b: i64) -> i64 {
    a - idiv(a, b) * b
}

fn map(name: &str, op: MapOp, args: &[&Array], ty: &Ty) -> Result<Array> {
    let n = ty.shape.len();
    let data = match (op.kind(), op) {
        (MapKind::Num | MapKind::Int, _) if args[0].scalar.is_int() => {
            let f = |i: usize| -> Result<i64> {
                let x = args[0].at(i).int();
                Ok(match op {
                    MapOp::Neg => x.wrapping_neg(),
                    MapOp::Abs => x.wrapping_abs(),
                    _ => {
                        let y = args[1].at(i).int();
                        match op {
                            MapOp::Add => x.wrapping_add(y),
                            MapOp::Sub => x.wrapping_sub(y),
                            MapOp::Mul => x.wrapping_mul(y),
                            MapOp::Min => x.min(y),
                            MapOp::Max => x.max(y),
                            MapOp::IDiv | MapOp::Mod if y == 0 => return err(format!("%{name}: map {op}: division by zero")),
                            MapOp::IDiv => idiv(x, y),
                            MapOp::Mod => imod(x, y),
                            _ => unreachable!(),
                        }
                    }
                })
            };
            Data::Int((0..n).map(f).collect::<Result<_>>()?)
        }
        (MapKind::Num | MapKind::Float, _) => {
            let f = |i: usize| -> f64 {
                let x = args[0].at(i).float();
                match op {
                    MapOp::Neg => -x,
                    MapOp::Abs => x.abs(),
                    MapOp::Exp => x.exp(),
                    MapOp::Log => x.ln(),
                    _ => {
                        let y = args[1].at(i).float();
                        match op {
                            MapOp::Add => x + y,
                            MapOp::Sub => x - y,
                            MapOp::Mul => x * y,
                            MapOp::Div => x / y,
                            MapOp::Min => x.min(y),
                            MapOp::Max => x.max(y),
                            _ => unreachable!(),
                        }
                    }
                }
            };
            Data::Float((0..n).map(f).collect())
        }
        (MapKind::EqCompare | MapKind::OrdCompare, _) => {
            let f = |i: usize| -> bool {
                let (x, y) = (args[0].at(i), args[1].at(i));
                let ord = match (x, y) {
                    (Item::Int(x), Item::Int(y)) => x.partial_cmp(&y),
                    (Item::Float(x), Item::Float(y)) => x.partial_cmp(&y),
                    (Item::Bool(x), Item::Bool(y)) => x.partial_cmp(&y),
                    _ => unreachable!("checked: one scalar type"),
                };
                use std::cmp::Ordering::*;
                match op {
                    MapOp::Eq => ord == Some(Equal),
                    MapOp::Ne => ord != Some(Equal),
                    MapOp::Lt => ord == Some(Less),
                    MapOp::Le => matches!(ord, Some(Less | Equal)),
                    MapOp::Gt => ord == Some(Greater),
                    MapOp::Ge => matches!(ord, Some(Greater | Equal)),
                    _ => unreachable!(),
                }
            };
            Data::Bool((0..n).map(f).collect())
        }
        (MapKind::Logic, _) => {
            let f = |i: usize| -> bool {
                let x = args[0].at(i).boolean();
                match op {
                    MapOp::Not => !x,
                    MapOp::And => x && args[1].at(i).boolean(),
                    MapOp::Or => x || args[1].at(i).boolean(),
                    _ => unreachable!(),
                }
            };
            Data::Bool((0..n).map(f).collect())
        }
        _ => unreachable!("checked: {op} on {}", args[0].scalar),
    };
    Array::new(ty.scalar, ty.shape.clone(), data)
}

/// The lengths before an axis, along it and after it (row order), so
/// item (o, j, i) is at `(o * len + j) * inner + i`.
pub fn axis_split(shape: &Shape, axis: usize) -> (usize, usize, usize) {
    let d = &shape.0;
    (d[..axis - 1].iter().product(), d[axis - 1], d[axis..].iter().product())
}

/// A reduce folds from the right, as X_eTaL's `r_/` does: the last
/// item first, then each earlier item combined with the result so far;
/// along an axis, each line of the array along that axis is folded so.
fn reduce(op: ReduceOp, a: &Array, axis: Option<usize>, ty: &Ty) -> Result<Array> {
    if let Some(ax) = axis {
        let (outer, len, inner) = axis_split(&a.shape, ax);
        let line = |o: usize, i: usize| -> Array {
            let data = |idx: usize| (o * len + idx) * inner + i;
            let d = match &a.data {
                Data::Int(v) => Data::Int((0..len).map(|j| v[data(j)]).collect()),
                Data::Float(v) => Data::Float((0..len).map(|j| v[data(j)]).collect()),
                Data::Bool(_) => unreachable!("checked: numbers"),
            };
            Array {
                scalar: a.scalar,
                shape: Shape(vec![len]),
                data: d,
            }
        };
        let scalar_ty = Ty {
            scalar: ty.scalar,
            shape: Shape::scalar(),
        };
        let mut ints = Vec::new();
        let mut floats = Vec::new();
        for o in 0..outer {
            for i in 0..inner {
                match reduce(op, &line(o, i), None, &scalar_ty)?.data {
                    Data::Int(v) => ints.push(v[0]),
                    Data::Float(v) => floats.push(v[0]),
                    Data::Bool(_) => unreachable!(),
                }
            }
        }
        let data = if a.scalar.is_int() { Data::Int(ints) } else { Data::Float(floats) };
        return Array::new(ty.scalar, ty.shape.clone(), data);
    }
    let data = match &a.data {
        Data::Int(v) => {
            let f = |x: i64, acc: i64| match op {
                ReduceOp::Add => x.wrapping_add(acc),
                ReduceOp::Mul => x.wrapping_mul(acc),
                ReduceOp::Min => x.min(acc),
                ReduceOp::Max => x.max(acc),
            };
            let (last, rest) = v.split_last().expect("checked: not empty");
            Data::Int(vec![rest.iter().rev().fold(*last, |acc, x| f(*x, acc))])
        }
        Data::Float(v) => {
            let f = |x: f64, acc: f64| {
                let r = match op {
                    ReduceOp::Add => x + acc,
                    ReduceOp::Mul => x * acc,
                    ReduceOp::Min => x.min(acc),
                    ReduceOp::Max => x.max(acc),
                };
                if a.scalar == Scalar::F32 {
                    r as f32 as f64
                } else {
                    r
                }
            };
            let (last, rest) = v.split_last().expect("checked: not empty");
            Data::Float(vec![rest.iter().rev().fold(*last, |acc, x| f(*x, acc))])
        }
        Data::Bool(_) => unreachable!("checked: numbers"),
    };
    Array::new(ty.scalar, Shape::scalar(), data)
}

/// A cast: between Int and Float as X_eTaL's `f_loat` and `f_loor`
/// go (toward negative infinity), a Bool to 0 or 1, a number to Bool
/// when it is not 0.
fn cast(a: &Array, to: Scalar) -> Result<Array> {
    let data = match (&a.data, to) {
        (Data::Int(v), t) if t.is_int() => Data::Int(v.clone()),
        (Data::Int(v), t) if t.is_float() => Data::Float(v.iter().map(|x| *x as f64).collect()),
        (Data::Int(v), Scalar::Bool) => Data::Bool(v.iter().map(|x| *x != 0).collect()),
        (Data::Float(v), t) if t.is_float() => Data::Float(v.clone()),
        (Data::Float(v), t) if t.is_int() => Data::Int(v.iter().map(|x| x.floor() as i64).collect()),
        (Data::Float(v), Scalar::Bool) => Data::Bool(v.iter().map(|x| *x != 0.0).collect()),
        (Data::Bool(v), t) if t.is_int() => Data::Int(v.iter().map(|x| i64::from(*x)).collect()),
        (Data::Bool(v), t) if t.is_float() => Data::Float(v.iter().map(|x| f64::from(u8::from(*x))).collect()),
        (Data::Bool(v), Scalar::Bool) => Data::Bool(v.clone()),
        _ => unreachable!("every cast is covered"),
    };
    Array::new(to, a.shape.clone(), data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::check;
    use crate::format::show;
    use crate::text::parse;

    fn eval(src: &str) -> Vec<String> {
        let checked = check(parse(src).unwrap()).unwrap();
        run(&checked, &HashMap::new()).unwrap().iter().map(show).collect()
    }

    #[test]
    fn arithmetic_with_scalar_extension() {
        let out = eval("%a = const i64 [4] 1 2 3 4\n%k = const i64 [] 10\n%b = map mul %k %a\n%c = map add %a %b\n%n = map neg %c\noutput %c\noutput %n\n");
        assert_eq!(out, ["11 22 33 44", "-11 -22 -33 -44"]);
    }

    #[test]
    fn floored_division_and_modulo() {
        assert_eq!(idiv(-7, 2), -4);
        assert_eq!(idiv(7, -2), -4);
        assert_eq!(idiv(7, 2), 3);
        assert_eq!(imod(-3, 2), 1);
        assert_eq!(imod(3, -2), -1);
        assert_eq!(imod(7, 3), 1);
        let out =
            eval("%a = const i64 [3] -3 3 7\n%b = const i64 [3] 2 -2 3\n%m = map mod %a %b\n%d = map idiv %a %b\noutput %m\noutput %d\n");
        assert_eq!(out, ["1 -1 1", "-2 -2 2"]);
        let e = run(
            &check(parse("%a = const i64 [1] 1\n%z = const i64 [] 0\n%m = map mod %a %z\noutput %m\n").unwrap()).unwrap(),
            &HashMap::new(),
        )
        .unwrap_err();
        assert_eq!(e.0, "%m: map mod: division by zero");
    }

    #[test]
    fn floats_compare_select_reduce() {
        let out = eval("%x = const f64 [4] -0.75 0.0 0.5 1.0\n%t = const f64 [] 0.25\n%m = map gt %x %t\n%z = const f64 [] 0.0\n%s = select %m %x %z\n%r = reduce add %s\n%mx = reduce max %x\n%q = map div %x %t\noutput %m\noutput %s\noutput %r\noutput %mx\noutput %q\n");
        assert_eq!(out, ["0 0 1 1", "0.0 0.0 0.5 1.0", "1.5", "1.0", "-3.0 0.0 2.0 4.0"]);
    }

    #[test]
    fn reduce_folds_from_the_right() {
        // 1 - (2 - 3) would be 2 for sub; for the associative ops only
        // the Float rounding order shows: 0.1 + (0.2 + 0.3) differs from (0.1 + 0.2) + 0.3.
        let out = eval("%x = const f64 [3] 0.1 0.2 0.3\n%r = reduce add %x\noutput %r\n");
        assert_eq!(out, ["0.6"]);
        assert_eq!(format!("{}", (0.1f64 + 0.2) + 0.3), "0.6000000000000001");
    }

    #[test]
    fn reduces_along_an_axis_as_xetal_does() {
        // M := 2 3 r_eshape r_ange 6: '+ r_/ M is 5 7 9, '+ r_/_2 M is 6 15, 'm_ax r_/_2 M is 3 6;
        // T := 2 2 3 r_eshape r_ange 12: '+ r_/_2 T and '+ r_/_3 T as xetal prints them.
        let out = eval("%m = const i64 [2 3] 1 2 3 4 5 6\n%c = reduce add axis=1 %m\n%r = reduce add axis=2 %m\n%x = reduce max axis=2 %m\n%t = const i64 [2 2 3] 1 2 3 4 5 6 7 8 9 10 11 12\n%t2 = reduce add axis=2 %t\n%t3 = reduce add axis=3 %t\n%t1 = reduce add axis=1 %t\noutput %c\noutput %r\noutput %x\noutput %t2\noutput %t3\noutput %t1\n");
        assert_eq!(
            out,
            ["5 7 9", "6 15", "3 6", " 5  7  9\n17 19 21", " 6 15\n24 33", " 8 10 12\n14 16 18"]
        );
    }

    #[test]
    fn casts() {
        let out = eval("%a = const i64 [3] -1 0 2\n%f = cast f64 %a\n%b = cast bool %a\n%i = cast i64 %b\n%g = const f64 [2] -1.5 2.5\n%fl = cast i64 %g\n%bf = cast f64 %b\noutput %f\noutput %b\noutput %i\noutput %fl\noutput %bf\n");
        assert_eq!(out, ["-1.0 0.0 2.0", "1 0 1", "1 0 1", "-2 2", "1.0 0.0 1.0"]);
    }

    #[test]
    fn narrow_types_round() {
        let out = eval("%a = const f32 [2] 0.1 0.2\n%s = map add %a %a\n%r = reduce add %a\noutput %s\noutput %r\n");
        assert_eq!(out, ["0.20000000298023224 0.4000000059604645", "0.30000001192092896"]);
        let out = eval("%a = const i32 [1] 2147483647\n%one = const i32 [] 1\n%s = map add %a %one\noutput %s\n");
        assert_eq!(out, ["-2147483648"]);
    }

    #[test]
    fn inputs_are_bound_by_name() {
        let checked = check(parse("%a = input i64 [3]\n%k = const i64 [] 2\n%b = map mul %k %a\noutput %b\n").unwrap()).unwrap();
        let mut inputs = HashMap::new();
        assert_eq!(run(&checked, &inputs).unwrap_err().0, "input %a (i64 [3]) is not bound");
        inputs.insert("a".to_string(), Array::ints(Shape(vec![2]), vec![1, 2]));
        assert_eq!(run(&checked, &inputs).unwrap_err().0, "input %a is i64 [3], bound to i64 [2]");
        inputs.insert("a".to_string(), Array::ints(Shape(vec![3]), vec![1, 2, 3]));
        assert_eq!(run(&checked, &inputs).unwrap().iter().map(show).collect::<Vec<_>>(), ["2 4 6"]);
    }
}
