//! The text form of a program: one value per line.
//!
//! ```text
//! # a comment runs to the end of the line
//! %a = input i64 [8]
//! %b = const i64 [8] 1 2 3 4 5 6 7 8
//! %k = const f64 [] 2.0
//! %c = map add %a %b
//! %m = map gt %c %b
//! %s = select %m %c %b
//! %r = reduce add %s
//! %f = cast f64 %r
//! output %c
//! output %f
//! ```
//!
//! A shape is `[]` (one value), `[8]` (a vector) or `[2 3]` (a matrix,
//! rows first). A constant lists every item in row order, or one
//! item for every position. Names are letters, digits and
//! underscores; a value is defined before it is used, once.

use crate::ir::*;
use crate::{err, Result};

/// Parse a program from its text.
pub fn parse(src: &str) -> Result<Program> {
    let mut program = Program::default();
    for (n, raw) in src.lines().enumerate() {
        let line_no = n + 1;
        let line = match raw.find('#') {
            Some(i) => &raw[..i],
            None => raw,
        };
        let spaced = line.replace('[', " [ ").replace(']', " ] ").replace('=', " = ");
        let toks: Vec<&str> = spaced.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        let at = |msg: String| -> Result<()> { err(format!("line {line_no}: {msg}")) };
        if toks[0] == "output" {
            if toks.len() != 2 {
                at("output takes one value: output %name".into())?;
            }
            let id = reference(&program, toks[1]).map_err(|e| prefix(line_no, e))?;
            program.outputs.push(id);
            continue;
        }
        let name = match toks[0].strip_prefix('%') {
            Some(name) if is_name(name) => name,
            _ => {
                return err(format!(
                    "line {line_no}: a line is `%name = op ...` or `output %name`, got `{}`",
                    toks[0]
                ))
            }
        };
        if program.find(name).is_some() {
            at(format!("%{name} is defined twice"))?;
        }
        if toks.get(1) != Some(&"=") {
            at(format!("expected `=` after %{name}"))?;
        }
        let rest = &toks[2..];
        let Some(&kind) = rest.first() else {
            return err(format!("line {line_no}: %{name} has no operation"));
        };
        let (ty, op) = operation(&program, kind, &rest[1..]).map_err(|e| prefix(line_no, e))?;
        program.values.push(Value {
            name: name.to_string(),
            ty,
            op,
        });
    }
    Ok(program)
}

fn prefix(line_no: usize, e: crate::Error) -> crate::Error {
    crate::Error(format!("line {line_no}: {}", e.0))
}

fn is_name(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_') && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn reference(program: &Program, tok: &str) -> Result<ValueId> {
    let Some(name) = tok.strip_prefix('%') else {
        return err(format!("expected a value reference `%name`, got `{tok}`"));
    };
    match program.find(name) {
        Some(id) => Ok(id),
        None => err(format!("%{name} is not defined (a value is defined before it is used)")),
    }
}

fn scalar(tok: &str) -> Result<Scalar> {
    Scalar::parse(tok).ok_or_else(|| crate::Error(format!("expected a scalar type (i32, i64, f32, f64, bool), got `{tok}`")))
}

/// A shape `[ n n ... ]`; returns it and the tokens after it.
fn shape<'a>(toks: &'a [&'a str]) -> Result<(Shape, &'a [&'a str])> {
    if toks.first() != Some(&"[") {
        return err(format!("expected a shape `[...]`, got `{}`", toks.first().unwrap_or(&"")));
    }
    let mut dims = Vec::new();
    let mut i = 1;
    loop {
        let Some(&t) = toks.get(i) else { return err("unclosed shape `[`") };
        if t == "]" {
            break;
        }
        let n: usize = t
            .parse()
            .map_err(|_| crate::Error(format!("a shape holds whole numbers, got `{t}`")))?;
        dims.push(n);
        i += 1;
    }
    Ok((Shape(dims), &toks[i + 1..]))
}

fn operation(program: &Program, kind: &str, rest: &[&str]) -> Result<(Option<Ty>, Op)> {
    match kind {
        "input" => {
            let (sc, rest) = typed(rest)?;
            if !rest.is_empty() {
                return err(format!("input takes a scalar type and a shape, got extra `{}`", rest.join(" ")));
            }
            Ok((Some(sc), Op::Input))
        }
        "const" => {
            let (ty, items) = typed(rest)?;
            let n = ty.shape.len();
            if items.len() != n && items.len() != 1 {
                return err(format!(
                    "const {} needs {n} item{} (or one for every position), got {}",
                    ty,
                    if n == 1 { "" } else { "s" },
                    items.len()
                ));
            }
            let c = match ty.scalar {
                Scalar::I32 | Scalar::I64 => Const::Int(items.iter().map(|t| int(t)).collect::<Result<_>>()?),
                Scalar::F32 | Scalar::F64 => Const::Float(items.iter().map(|t| float(t)).collect::<Result<_>>()?),
                Scalar::Bool => Const::Bool(items.iter().map(|t| boolean(t)).collect::<Result<_>>()?),
            };
            Ok((Some(ty), Op::Const(c)))
        }
        "map" => {
            let Some(&name) = rest.first() else {
                return err("map needs an operation: map add %a %b");
            };
            let op = MapOp::parse(name).ok_or_else(|| {
                crate::Error(format!(
                    "unknown map operation `{name}` (one of {})",
                    MapOp::ALL.iter().map(|o| o.name()).collect::<Vec<_>>().join(", ")
                ))
            })?;
            let args: Vec<ValueId> = rest[1..].iter().map(|t| reference(program, t)).collect::<Result<_>>()?;
            if args.len() != op.arity() {
                return err(format!(
                    "map {op} takes {} argument{}, got {}",
                    op.arity(),
                    if op.arity() == 1 { "" } else { "s" },
                    args.len()
                ));
            }
            Ok((None, Op::Map { op, args }))
        }
        "select" => {
            if rest.len() != 3 {
                return err("select takes three values: select %cond %a %b");
            }
            Ok((
                None,
                Op::Select {
                    cond: reference(program, rest[0])?,
                    a: reference(program, rest[1])?,
                    b: reference(program, rest[2])?,
                },
            ))
        }
        "reduce" => {
            if rest.len() != 2 {
                return err("reduce takes an operation and a value: reduce add %v");
            }
            let op = ReduceOp::parse(rest[0]).ok_or_else(|| {
                crate::Error(format!(
                    "unknown reduce operation `{}` (one of {})",
                    rest[0],
                    ReduceOp::ALL.iter().map(|o| o.name()).collect::<Vec<_>>().join(", ")
                ))
            })?;
            Ok((
                None,
                Op::Reduce {
                    op,
                    arg: reference(program, rest[1])?,
                },
            ))
        }
        "cast" => {
            if rest.len() != 2 {
                return err("cast takes a scalar type and a value: cast f64 %a");
            }
            let to = scalar(rest[0])?;
            Ok((
                None,
                Op::Cast {
                    op: CastOp { to },
                    arg: reference(program, rest[1])?,
                },
            ))
        }
        other => err(format!("unknown operation `{other}` (input, const, map, select, reduce, cast)")),
    }
}

/// `scalar shape ...`: the type and the tokens after it.
fn typed<'a>(toks: &'a [&'a str]) -> Result<(Ty, &'a [&'a str])> {
    let Some(&s) = toks.first() else {
        return err("expected a scalar type and a shape");
    };
    let sc = scalar(s)?;
    let (sh, rest) = shape(&toks[1..])?;
    Ok((Ty { scalar: sc, shape: sh }, rest))
}

fn int(t: &str) -> Result<i64> {
    t.parse().map_err(|_| crate::Error(format!("not an Int: `{t}`")))
}

fn float(t: &str) -> Result<f64> {
    match t {
        "inf" => Ok(f64::INFINITY),
        "-inf" => Ok(f64::NEG_INFINITY),
        _ => t.parse().map_err(|_| crate::Error(format!("not a Float: `{t}`"))),
    }
}

fn boolean(t: &str) -> Result<bool> {
    match t {
        "0" | "false" => Ok(false),
        "1" | "true" => Ok(true),
        _ => err(format!("not a Bool (0, 1, false, true): `{t}`")),
    }
}

/// Print a program in the text form; `parse(&print(p)) == p`.
pub fn print(program: &Program) -> String {
    let mut out = String::new();
    let name = |id: ValueId| format!("%{}", program.value(id).name);
    for v in &program.values {
        out.push_str(&format!("%{} = ", v.name));
        match &v.op {
            Op::Input => out.push_str(&format!("input {}", v.ty.as_ref().expect("an input declares its type"))),
            Op::Const(c) => {
                out.push_str(&format!("const {}", v.ty.as_ref().expect("a const declares its type")));
                match c {
                    Const::Int(items) => items.iter().for_each(|x| out.push_str(&format!(" {x}"))),
                    Const::Float(items) => items.iter().for_each(|x| out.push_str(&format!(" {}", crate::format::float(*x)))),
                    Const::Bool(items) => items.iter().for_each(|x| out.push_str(if *x { " 1" } else { " 0" })),
                }
            }
            Op::Map { op, args } => {
                out.push_str(&format!("map {op}"));
                args.iter().for_each(|a| out.push_str(&format!(" {}", name(*a))));
            }
            Op::Select { cond, a, b } => out.push_str(&format!("select {} {} {}", name(*cond), name(*a), name(*b))),
            Op::Reduce { op, arg } => out.push_str(&format!("reduce {op} {}", name(*arg))),
            Op::Cast { op, arg } => out.push_str(&format!("cast {} {}", op.to, name(*arg))),
        }
        out.push('\n');
    }
    for o in &program.outputs {
        out.push_str(&format!("output {}\n", name(*o)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
%a = input i64 [8]
%b = const i64 [8] 1 2 3 4 5 6 7 8
%k = const f64 [] 2.0
%c = map add %a %b
%m = map gt %c %b
%s = select %m %c %b
%r = reduce add %s
%f = cast f64 %r
%t = const bool [2] 1 0
output %c
output %f
";

    #[test]
    fn round_trips() {
        let p = parse(SAMPLE).unwrap();
        assert_eq!(print(&p), SAMPLE);
        assert_eq!(parse(&print(&p)).unwrap(), p);
    }

    #[test]
    fn comments_and_spacing() {
        let p = parse("# hello\n  %a=const i64 [ 2 ] 1 2 # two\n\noutput %a\n").unwrap();
        assert_eq!(p.values.len(), 1);
        assert_eq!(p.outputs, vec![ValueId(0)]);
        assert_eq!(p.value(ValueId(0)).ty.as_ref().unwrap().shape, Shape(vec![2]));
    }

    #[test]
    fn one_item_fills() {
        let p = parse("%z = const f64 [4] 0.0\n").unwrap();
        assert_eq!(p.value(ValueId(0)).op, Op::Const(Const::Float(vec![0.0])));
    }

    #[test]
    fn errors_name_the_line() {
        let e = |s: &str| parse(s).unwrap_err().0;
        assert_eq!(
            e("%a = const i64 [2] 1 2 3\n"),
            "line 1: const i64 [2] needs 2 items (or one for every position), got 3"
        );
        assert_eq!(
            e("%a = const i64 [1] 1\n%b = map add %a %c\n"),
            "line 2: %c is not defined (a value is defined before it is used)"
        );
        assert_eq!(e("%a = const i64 [1] 1\n%a = const i64 [1] 2\n"), "line 2: %a is defined twice");
        assert_eq!(
            e("%a = const i64 [1] 1\n%b = map neg %a %a\n"),
            "line 2: map neg takes 1 argument, got 2"
        );
        assert_eq!(e("%a = const i64 [1] 1\n%b = map plus %a\n"), "line 2: unknown map operation `plus` (one of add, sub, mul, div, idiv, mod, min, max, neg, abs, exp, log, eq, ne, lt, le, gt, ge, and, or, not)");
        assert_eq!(
            e("%a = input i6 [1]\n"),
            "line 1: expected a scalar type (i32, i64, f32, f64, bool), got `i6`"
        );
        assert_eq!(e("%a = input i64 [1\n"), "line 1: unclosed shape `[`");
        assert_eq!(
            e("a = input i64 [1]\n"),
            "line 1: a line is `%name = op ...` or `output %name`, got `a`"
        );
        assert_eq!(
            e("%a = frob\n"),
            "line 1: unknown operation `frob` (input, const, map, select, reduce, cast)"
        );
        assert_eq!(e("%a = const bool [1] 2\n"), "line 1: not a Bool (0, 1, false, true): `2`");
    }
}
