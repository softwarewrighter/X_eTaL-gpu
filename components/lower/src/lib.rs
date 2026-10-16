//! The lowering: X_eTaL source to an XIR program, through the pinned
//! X_eTaL's own crates (`xetal_program::load` assembles the program
//! with its libraries and lowers it to Core; nothing is parsed here).
//!
//! The translator is a small evaluator over Core. A value is either
//! **static** (known now: a literal array, a shape, a count, a file
//! read with `[]N_GET`, Int arithmetic on those) or a **node** of the
//! XIR program being built (an array the device computes). A primitive
//! applied to static arguments folds when it builds data or computes
//! shapes (`r_eshape` of a literal, `o_ffsets`, `r_ange`, `s_hape`,
//! `t_ally`, `c_at`, Int arithmetic); any other primitive emits an XIR
//! operation, its static arguments written in as constants. Library
//! functions are closures inlined at their calls; `p_ower` with a
//! static count unrolls. What is outside the subset (recursion,
//! dynamic shapes, `c_at` of device arrays, `^`, `s_\`, ...) is refused
//! with a reason naming the X_eTaL construct.
//!
//! Semantics are X_eTaL's: the evaluator is the specification (plan A2),
//! and the folding of static values runs the XIR interpreter on them,
//! which reproduces the evaluator (its tests say so).

pub mod cache;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use xetal_gpu_xir::ir::{Const, MapOp, Op, Program, ReduceOp, Scalar, Shape, TableOp, Ty, Value, ValueId};
use xetal_gpu_xir::{check, interp, Array, Data, Error, Result};

fn err<T>(message: impl Into<String>) -> Result<T> {
    Err(Error(message.into()))
}
use xetal_ir::{Expr, Item, Kind, Param};
use xetal_lex::Number;

/// What a program prints, in order: a value the device computes, or
/// text the evaluator printed for a statement outside the subset.
#[derive(Debug, Clone, PartialEq)]
pub enum Printed {
    Device(ValueId),
    Host(String),
}

/// A lowered program: the XIR, the inputs the host evaluated (a
/// top-level binding outside the subset, run by the pinned `xetal`
/// up to that statement and bound by name), what it prints in order,
/// and the X_eTaL line each value came from (for `explain`).
#[derive(Debug, Clone)]
pub struct Lowered {
    pub program: Program,
    pub inputs: HashMap<String, Array>,
    pub printed: Vec<Printed>,
    pub lines: HashMap<ValueId, String>,
    /// Why each host-evaluated input (and printed statement) was not lowered.
    pub reasons: Vec<(String, String)>,
}

/// Lower the program in `path` (its libraries found beside it and on
/// `XETAL_PATH`; `[]N_GET` reads relative to its directory); the
/// program only, for programs the device can take whole.
pub fn lower_file(path: &Path) -> Result<Program> {
    lower_file_full(path).map(|l| l.program)
}

/// Lower the program in `path`, with whatever the device cannot take
/// evaluated by the pinned `xetal` and bound (see [`Lowered`]).
pub fn lower_file_full(path: &Path) -> Result<Lowered> {
    let src = std::fs::read_to_string(path).map_err(|e| Error(format!("cannot read {}: {e}", path.display())))?;
    let dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    lower_source(&path.to_string_lossy(), &src, &dir)
}

/// Lower `src`, reported as `name`; files are read relative to `dir`.
/// A lowering is cached by everything it read (see [`cache`]).
pub fn lower_source(name: &str, src: &str, dir: &Path) -> Result<Lowered> {
    let loaded = xetal_program::load(name, src).map_err(|d| Error(format!("{}: {}", d.code, d.message)))?;
    let cache_dir = cache::dir();
    // The files []N_GET reads are part of the key; they are known only after
    // translating, so the key is checked against a manifest of them.
    let combined = loaded.sources.combined().to_string();
    if let Some(cd) = &cache_dir {
        let manifest = cd.join(format!("{}.files", cache::key(&combined, &[])));
        if let Ok(list) = std::fs::read_to_string(&manifest) {
            let files: Vec<(PathBuf, String)> = list
                .lines()
                .filter(|l| !l.is_empty())
                .map(|l| cache::file_entry(Path::new(l)))
                .collect();
            if let Some(hit) = cache::read(cd, &cache::key(&combined, &files)) {
                return Ok(hit);
            }
        }
    }
    let mut t = Translator {
        program: Program::default(),
        types: Vec::new(),
        globals: HashMap::new(),
        dir: dir.to_path_buf(),
        names: HashMap::new(),
        inputs: HashMap::new(),
        printed: Vec::new(),
        reasons: Vec::new(),
        lines: HashMap::new(),
        line: String::new(),
        top: Env::default(),
        files_read: Vec::new(),
    };
    let host = Host::new(dir, src);
    let mut env = Env::default();
    for item in &loaded.program.items {
        match item {
            Item::Def { name, value } => {
                t.globals.insert(name.clone(), value.clone());
            }
            Item::Let { name, value, .. } => {
                let place = loaded.sources.locate(value.span.start);
                t.line = source_line(loaded.sources.written_text(place.index), place.offset, place.line);
                let mark = t.mark();
                let v = match t.eval(value, &env) {
                    Ok(v) => v,
                    Err(reason) => {
                        // Outside the subset: the evaluator makes the value, the
                        // device takes it as an input. Whatever the attempt pushed goes.
                        t.rollback(mark);
                        if name.starts_with('%') {
                            // A tuple pattern's hidden value: its parts, bound next
                            // by name, fall back to the evaluator one by one.
                            continue;
                        }
                        let end = loaded.sources.locate(value.span.end);
                        let end = statement_end(loaded.sources.written_text(end.index), end.offset);
                        t.reasons.push((name.clone(), reason.0.clone()));
                        match host.value(name, end, &reason.0)? {
                            HostValue::Array(a) => {
                                let ty = a.ty();
                                let id = t.push(name, Some(ty), Op::Input)?;
                                t.inputs.insert(t.program.value(id).name.clone(), a);
                                V::Node(id)
                            }
                            HostValue::Text(s) => V::Str(s),
                        }
                    }
                };
                // Data bound at the top level (weights, a batch) is written in
                // once, under its own name; shapes and counts stay static.
                let v = match v {
                    V::Static(a, false) => V::Node(t.constant(a, Some(name))),
                    v => v,
                };
                env = env.bind(name, v);
                t.top = env.clone();
            }
            Item::Set { name, .. } => return err(format!("{name}!: a variable that changes is outside the subset")),
            Item::Eval(e) => {
                let place = loaded.sources.locate(e.span.start);
                t.line = source_line(loaded.sources.written_text(place.index), place.offset, place.line);
                let mark = t.mark();
                let v = match t.eval(e, &env) {
                    Ok(v) => v,
                    Err(reason) => {
                        t.rollback(mark);
                        let (start, end) = (loaded.sources.locate(e.span.start), loaded.sources.locate(e.span.end));
                        let end_offset = statement_end(loaded.sources.written_text(end.index), end.offset);
                        let text = host.printed(start.offset, end_offset, &reason.0)?;
                        t.reasons.push((t.line.clone(), reason.0.clone()));
                        t.printed.push(Printed::Host(text));
                        continue;
                    }
                };
                match v {
                    V::Node(id) => {
                        t.program.outputs.push(id);
                        t.printed.push(Printed::Device(id));
                    }
                    V::Static(a, _) => {
                        let id = t.constant(a, None);
                        t.program.outputs.push(id);
                        t.printed.push(Printed::Device(id));
                    }
                    V::Unit => {}
                    V::Str(text) => t.printed.push(Printed::Host(text)),
                    other => return err(format!("a {} cannot be printed by the device", other.what())),
                }
            }
        }
    }
    if t.program.outputs.is_empty() {
        return err("the program prints nothing the device could compute");
    }
    let lowered = Lowered {
        program: t.program,
        inputs: t.inputs,
        printed: t.printed,
        lines: t.lines,
        reasons: t.reasons,
    };
    if let Some(cd) = &cache_dir {
        let files: Vec<(PathBuf, String)> = t.files_read.iter().map(|p| cache::file_entry(p)).collect();
        let _ = std::fs::create_dir_all(cd);
        let manifest: String = t.files_read.iter().map(|p| p.to_string_lossy().into_owned() + "\n").collect();
        let _ = std::fs::write(cd.join(format!("{}.files", cache::key(&combined, &[]))), manifest);
        cache::write(cd, &cache::key(&combined, &files), &lowered);
    }
    Ok(lowered)
}

/// Lower one function, given as its source text (`{ x w -> ... }`), for
/// arguments of the given types: the function is defined as `u:a_ccel`
/// in a program of its own (so its libraries resolve from `dir` and
/// `XETAL_PATH`), applied to one input per parameter, and its result
/// is the program's output. This is what the gpu extension runs for
/// `gp:a_ccel<`: the arguments' shapes are known when they are bound,
/// so the lowering is specialized to them (and cached by the caller).
pub fn lower_function(function: &str, params: &[(String, Ty)], dir: &Path, uses: &str) -> Result<Lowered> {
    let src = format!("{uses}\nu:a_ccel := {function}\n");
    let loaded = xetal_program::load("gp:a_ccel<", &src).map_err(|d| Error(format!("{}: {}", d.code, d.message)))?;
    let mut t = Translator {
        program: Program::default(),
        types: Vec::new(),
        globals: HashMap::new(),
        dir: dir.to_path_buf(),
        names: HashMap::new(),
        inputs: HashMap::new(),
        printed: Vec::new(),
        reasons: Vec::new(),
        lines: HashMap::new(),
        line: "the accelerated function".into(),
        top: Env::default(),
        files_read: Vec::new(),
    };
    for item in &loaded.program.items {
        match item {
            Item::Def { name, value } => {
                t.globals.insert(name.clone(), value.clone());
            }
            Item::Eval(_) | Item::Let { .. } | Item::Set { .. } => {}
        }
    }
    let Some(def) = t.globals.get("u:a_ccel").cloned() else {
        return err("the function did not define itself (is it `{ x -> ... }`?)");
    };
    let mut f = t.eval(&def, &Env::default())?;
    for (name, ty) in params {
        let id = t.push(name, Some(ty.clone()), Op::Input)?;
        f = t.apply(f, V::Node(id))?;
    }
    match f {
        V::Node(id) => t.program.outputs.push(id),
        V::Static(a, _) => {
            let id = t.constant(a, None);
            t.program.outputs.push(id);
        }
        other => return err(format!("the function gives a {}, which the device cannot return", other.what())),
    }
    let printed = t.program.outputs.iter().map(|o| Printed::Device(*o)).collect();
    Ok(Lowered {
        program: t.program,
        inputs: HashMap::new(),
        printed,
        lines: t.lines,
        reasons: Vec::new(),
    })
}

/// The end of the statement containing `offset`: the end of its line,
/// or of the last line of a brace block it opens (a lambda over lines).
fn statement_end(text: &str, offset: usize) -> usize {
    let offset = offset.min(text.len());
    let start = text[..offset].rfind('\n').map_or(0, |n| n + 1);
    let mut depth = 0i32;
    let mut end = start;
    for (i, c) in text[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            '\n' if depth <= 0 && start + i >= offset => {
                end = start + i;
                return end;
            }
            _ => {}
        }
        end = start + i + c.len_utf8();
    }
    end
}

/// What the evaluator gave for a binding: an array, or text.
enum HostValue {
    Array(Array),
    Text(String),
}

/// The source line at `offset` of `text`, prefixed with its number.
fn source_line(text: &str, offset: usize, line: usize) -> String {
    let offset = offset.min(text.len());
    let start = text[..offset].rfind('\n').map_or(0, |n| n + 1);
    let end = text[offset..].find('\n').map_or(text.len(), |n| offset + n);
    format!("{line}: {}", text[start..end].trim())
}

/// The pinned `xetal`, run on a prefix of the program for what the
/// device cannot compute.
struct Host {
    xetal: Option<PathBuf>,
    dir: PathBuf,
    src: String,
}

impl Host {
    fn new(dir: &Path, src: &str) -> Host {
        let xetal = std::env::var_os("XETAL_BIN").map(PathBuf::from).or_else(|| {
            let mut roots: Vec<PathBuf> = Vec::new();
            if let Ok(cwd) = std::env::current_dir() {
                roots.push(cwd);
            }
            roots.push(std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf()));
            for root in roots {
                let mut d: Option<&Path> = Some(&root);
                while let Some(p) = d {
                    let candidate = p.join("bin/xetal");
                    if candidate.exists() {
                        return Some(candidate);
                    }
                    d = p.parent();
                }
            }
            None
        });
        Host {
            xetal,
            dir: dir.to_path_buf(),
            src: src.to_string(),
        }
    }

    /// The program up to `end`, run; its printed lines.
    fn run(&self, text: &str, reason: &str) -> Result<Vec<String>> {
        let Some(xetal) = &self.xetal else {
            return err(format!(
                "{reason}; the evaluator would compute it, but no xetal was found (just xetal, or XETAL_BIN)"
            ));
        };
        // xetal run takes a file: a temporary one beside the program, so
        // its libraries and file reads resolve as the program's do.
        let temp = self.dir.join(format!(".xetal-gpu-host-{}.xtl", std::process::id()));
        std::fs::write(&temp, text).map_err(|e| Error(format!("cannot write {}: {e}", temp.display())))?;
        let out = std::process::Command::new(xetal)
            .args(["run", "--seed", "1", "--ascii"])
            .arg(temp.file_name().expect("a file name"))
            .current_dir(&self.dir)
            .output();
        if std::env::var_os("XETAL_GPU_KEEP_HOST").is_none() {
            let _ = std::fs::remove_file(&temp);
        }
        let out = out.map_err(|e| Error(format!("cannot run {}: {e}", xetal.display())))?;
        if !out.status.success() {
            return err(format!(
                "{reason}; the evaluator was asked for it and failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).lines().map(str::to_string).collect())
    }

    /// The value of `name` after the statements up to `end`: its shape
    /// and its items, printed by the evaluator and read back (text, when
    /// the items are not numbers).
    fn value(&self, name: &str, end: usize, reason: &str) -> Result<HostValue> {
        let prefix = &self.src[..end.min(self.src.len())];
        let lines = self.run(&format!("{prefix}\ns_hape {name}\nr_avel {name}\n"), reason)?;
        let n = lines.len();
        if n < 2 {
            return err(format!("{reason}; the evaluator printed nothing for {name}"));
        }
        let dims: Vec<usize> = lines[n - 2]
            .split_whitespace()
            .map(|d| d.parse().map_err(|_| Error(format!("{name}: a shape of `{}`?", lines[n - 2]))))
            .collect::<Result<_>>()?;
        let items: Vec<&str> = lines[n - 1].split_whitespace().collect();
        let shape = Shape(dims);
        if items.iter().all(|t| t.parse::<i64>().is_ok()) {
            Ok(HostValue::Array(Array::ints(
                shape,
                items.iter().map(|t| t.parse().unwrap()).collect(),
            )))
        } else if items.iter().any(|t| t.parse::<f64>().is_err() && *t != "inf" && *t != "-inf") {
            Ok(HostValue::Text(lines[n - 1].clone()))
        } else {
            let floats: Vec<f64> = items
                .iter()
                .map(|t| match *t {
                    "inf" => Ok(f64::INFINITY),
                    "-inf" => Ok(f64::NEG_INFINITY),
                    t => t
                        .parse()
                        .map_err(|_| Error(format!("{name}: the evaluator printed `{t}`, not a number"))),
                })
                .collect::<Result<_>>()?;
            Ok(HostValue::Array(Array::floats(shape, floats)))
        }
    }

    /// What the statement from `start` to `end` prints: the program
    /// run with and without it, the difference.
    fn printed(&self, start: usize, end: usize, reason: &str) -> Result<String> {
        let start = start.min(self.src.len());
        let start = self.src[..start].rfind('\n').map_or(0, |n| n + 1);
        let before = &self.src[..start];
        let with = &self.src[..end.min(self.src.len())];
        let n = self.run(before, reason)?.len();
        let lines = self.run(with, reason)?;
        Ok(lines[n.min(lines.len())..].iter().map(|l| l.clone() + "\n").collect())
    }
}

/// A value during translation.
#[derive(Clone)]
enum V {
    /// Known now; the flag says it is shape-like (a shape, a count, a
    /// range, Int arithmetic on those) rather than data, so Int
    /// arithmetic on it folds instead of going to the device.
    Static(Array, bool),
    /// Computed by the device: a value of the XIR program.
    Node(ValueId),
    /// A lambda with its environment (a library function, a local one).
    Fun(Rc<Closure>),
    /// A primitive, with the function arguments it has taken so far
    /// (`'+ r_/` is `r_/` holding `+`), and its axes.
    Prim(Prim),
    /// A dyadic primitive that has its left argument and waits for the right.
    Waiting {
        prim: Prim,
        left: Box<V>,
    },
    /// `'f p_ower`, then with its count: waits for the data.
    Power {
        f: Box<V>,
        count: Option<i64>,
    },
    Str(String),
    Unit,
    Tuple(Vec<V>),
}

#[derive(Clone)]
struct Closure {
    param: Param,
    body: Expr,
    env: Env,
}

#[derive(Clone, Debug)]
struct Prim {
    name: String,
    axes: Vec<u8>,
    /// Function arguments taken (the `f` of `'f r_/`, the `f` and `g` of `'f 'g i_nner`).
    funs: Vec<Prim>,
}

impl V {
    fn what(&self) -> String {
        match self {
            V::Static(a, _) => format!("static {}", a.ty()),
            V::Node(_) => "device array".into(),
            V::Fun(_) => "function".into(),
            V::Prim(p) => format!("primitive {}", p.name),
            V::Waiting { prim, .. } => format!("primitive {} with its left argument", prim.name),
            V::Power { .. } => "p_ower".into(),
            V::Str(_) => "text".into(),
            V::Unit => "unit".into(),
            V::Tuple(_) => "tuple".into(),
        }
    }
}

/// The lexical environment: names to values (persistent, so closures share it).
#[derive(Clone, Default)]
struct Env(Option<Rc<(String, V, Env)>>);

impl Env {
    fn bind(&self, name: &str, v: V) -> Env {
        Env(Some(Rc::new((name.to_string(), v, self.clone()))))
    }

    fn get(&self, name: &str) -> Option<&V> {
        let mut e = self;
        while let Some(cell) = &e.0 {
            if cell.0 == name {
                return Some(&cell.1);
            }
            e = &cell.2;
        }
        None
    }
}

struct Translator {
    program: Program,
    /// The type of every value of `program`, as the checker gives them.
    types: Vec<Ty>,
    /// Library and `u:` definitions, lazily evaluated at each reference.
    globals: HashMap<String, Expr>,
    dir: PathBuf,
    /// How many values carry each base name, for unique XIR names.
    names: HashMap<String, usize>,
    inputs: HashMap<String, Array>,
    printed: Vec<Printed>,
    reasons: Vec<(String, String)>,
    /// The X_eTaL line each value came from.
    lines: HashMap<ValueId, String>,
    /// The line of the statement being translated.
    line: String,
    /// The top-level bindings so far: what a definition's body sees.
    top: Env,
    /// The files `[]N_GET` read, for the cache's key.
    files_read: Vec<PathBuf>,
}

impl Translator {
    // ----- the XIR program being built

    fn name(&mut self, base: &str) -> String {
        let base: String = base
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
            .collect();
        let base = if base.is_empty() || base.starts_with(|c: char| c.is_ascii_digit()) {
            format!("v{base}")
        } else {
            base
        };
        let n = self.names.entry(base.clone()).or_insert(0);
        *n += 1;
        if *n == 1 && self.program.find(&base).is_none() {
            return base;
        }
        loop {
            let candidate = format!("{base}_{}", n);
            if self.program.find(&candidate).is_none() {
                return candidate;
            }
            *n += 1;
        }
    }

    /// Push a value and learn its type from the checker.
    fn push(&mut self, base: &str, ty: Option<Ty>, op: Op) -> Result<ValueId> {
        let name = self.name(base);
        self.program.values.push(Value { name, ty, op });
        let id = ValueId(self.program.values.len() - 1);
        let mut probe = self.program.clone();
        probe.outputs = vec![id];
        match check(probe) {
            Ok(c) => {
                self.types = c.types;
                self.lines.insert(id, self.line.clone());
                Ok(id)
            }
            Err(e) => {
                self.program.values.pop();
                Err(e)
            }
        }
    }

    fn ty(&self, id: ValueId) -> &Ty {
        &self.types[id.0]
    }

    /// Where the program stands, to roll an attempt back to.
    fn mark(&self) -> (usize, HashMap<String, usize>) {
        (self.program.values.len(), self.names.clone())
    }

    fn rollback(&mut self, mark: (usize, HashMap<String, usize>)) {
        let (n, names) = mark;
        self.program.values.truncate(n);
        self.types.truncate(n);
        self.lines.retain(|id, _| id.0 < n);
        self.names = names;
    }

    /// A static array written into the program as a constant.
    fn constant(&mut self, a: Array, base: Option<&str>) -> ValueId {
        let c = match &a.data {
            Data::Int(v) => Const::Int(v.clone()),
            Data::Float(v) => Const::Float(v.clone()),
            Data::Bool(v) => Const::Bool(v.clone()),
        };
        let ty = a.ty();
        self.push(base.unwrap_or("c"), Some(ty), Op::Const(c)).expect("a constant checks")
    }

    /// A value as a node of the program (a static one written in).
    /// The rank of a value's array; no rank for what is not an array.
    fn rank_of(&self, v: &V) -> usize {
        match v {
            V::Static(a, _) => a.shape.rank(),
            V::Node(id) => self.types[id.0].shape.rank(),
            _ => usize::MAX,
        }
    }

    fn node(&mut self, v: V, base: &str) -> Result<ValueId> {
        match v {
            V::Node(id) => Ok(id),
            V::Static(a, _) => Ok(self.constant(a, Some(base))),
            other => err(format!("expected an array, got a {}", other.what())),
        }
    }

    /// An operation on static arrays, computed now by the XIR interpreter.
    fn fold(&self, base: &str, args: Vec<Array>, op: impl Fn(Vec<ValueId>) -> Op) -> Result<Array> {
        let mut p = Program::default();
        let mut ids = Vec::new();
        for (i, a) in args.into_iter().enumerate() {
            let c = match &a.data {
                Data::Int(v) => Const::Int(v.clone()),
                Data::Float(v) => Const::Float(v.clone()),
                Data::Bool(v) => Const::Bool(v.clone()),
            };
            p.values.push(Value {
                name: format!("a{i}"),
                ty: Some(a.ty()),
                op: Op::Const(c),
            });
            ids.push(ValueId(i));
        }
        p.values.push(Value {
            name: base.to_string(),
            ty: None,
            op: op(ids),
        });
        p.outputs.push(ValueId(p.values.len() - 1));
        let checked = check(p)?;
        let mut out = interp::run(&checked, &HashMap::new())?;
        Ok(out.remove(0))
    }

    // ----- evaluation of Core

    fn eval(&mut self, e: &Expr, env: &Env) -> Result<V> {
        match &e.kind {
            Kind::Lit(Number::Int(i)) => Ok(V::Static(Array::ints(Shape::scalar(), vec![*i]), true)),
            Kind::Lit(Number::Float(f)) => Ok(V::Static(Array::floats(Shape::scalar(), vec![*f]), false)),
            Kind::Str(s) => Ok(V::Str(s.clone())),
            Kind::Unit => Ok(V::Unit),
            Kind::Array(items) => {
                let mut ints = Vec::new();
                let mut floats = Vec::new();
                let mut any_float = false;
                for it in items {
                    match self.eval(it, env)? {
                        V::Static(a, _) if a.shape.is_scalar() => match a.data {
                            Data::Int(v) => {
                                ints.push(v[0]);
                                floats.push(v[0] as f64);
                            }
                            Data::Float(v) => {
                                any_float = true;
                                floats.push(v[0]);
                            }
                            Data::Bool(_) => return err("a strand of Bools is outside the subset"),
                        },
                        other => return err(format!("a strand of {}s is outside the subset", other.what())),
                    }
                }
                let n = items.len();
                Ok(V::Static(
                    if any_float {
                        Array::floats(Shape(vec![n]), floats)
                    } else {
                        Array::ints(Shape(vec![n]), ints)
                    },
                    false,
                ))
            }
            Kind::Tuple(parts) => Ok(V::Tuple(parts.iter().map(|p| self.eval(p, env)).collect::<Result<_>>()?)),
            Kind::Proj { index, size, tuple } => match self.eval(tuple, env)? {
                V::Tuple(parts) if parts.len() == *size => Ok(parts[*index].clone()),
                other => err(format!("a tuple of {size} was expected, got a {}", other.what())),
            },
            Kind::Var(name) => env.get(name).cloned().ok_or_else(|| Error(format!("{name} is not bound"))),
            Kind::Global(name) => {
                let Some(def) = self.globals.get(name).cloned() else {
                    return err(format!("{name} is not defined (is its library on XETAL_PATH?)"));
                };
                // A definition sees the globals and the top-level bindings, not the caller's locals.
                let top = self.top.clone();
                self.eval(&def, &top)
            }
            Kind::Prim(name) => Ok(V::Prim(Prim {
                name: name.clone(),
                axes: Vec::new(),
                funs: Vec::new(),
            })),
            Kind::Axes { axes, f, .. } => match self.eval(f, env)? {
                V::Prim(mut p) => {
                    p.axes = axes.clone();
                    Ok(V::Prim(p))
                }
                other => err(format!("an axis subscript on a {} is outside the subset", other.what())),
            },
            Kind::Lam { param, body, .. } => Ok(V::Fun(Rc::new(Closure {
                param: param.clone(),
                body: (**body).clone(),
                env: env.clone(),
            }))),
            Kind::App(f, x) => {
                let f = self.eval(f, env)?;
                let x = self.eval(x, env)?;
                self.apply(f, x)
            }
            Kind::App2 { f, left, right } => {
                // `x f y` is `(f x) y`: the function, the right argument, then the left (E4).
                let f = self.eval(f, env)?;
                let r = self.eval(right, env)?;
                let l = self.eval(left, env)?;
                let g = self.apply(f, l)?;
                self.apply(g, r)
            }
            Kind::Let { name, value, body, .. } => {
                let v = self.eval(value, env)?;
                let inner = env.bind(name, v);
                self.eval(body, &inner)
            }
            Kind::Set { name, .. } => err(format!("{name}!: a variable that changes is outside the subset")),
            Kind::If { cond, then, other } => match self.eval(cond, env)? {
                V::Static(a, _) if a.shape.is_scalar() => {
                    let truth = match &a.data {
                        Data::Bool(v) => v[0],
                        Data::Int(v) => v[0] != 0,
                        Data::Float(v) => v[0] != 0.0,
                    };
                    if truth {
                        self.eval(then, env)
                    } else {
                        self.eval(other, env)
                    }
                }
                _ => err("a guard on a value the device computes is outside the subset (the condition must be known now)"),
            },
            Kind::NoMatch => err("no guard matched"),
        }
    }

    fn apply(&mut self, f: V, x: V) -> Result<V> {
        match f {
            V::Fun(c) => {
                let env = match &c.param {
                    Param::Name(n) => c.env.bind(n, x),
                    Param::Unit => c.env.clone(),
                };
                self.eval(&c.body, &env)
            }
            V::Prim(p) => self.prim(p, x),
            V::Waiting { prim, left } => self.dyadic(&prim, *left, x),
            V::Power { f, count: None } => {
                let count = self.int_of(&x, "p_ower")?;
                Ok(V::Power { f, count: Some(count) })
            }
            V::Power { f, count: Some(count) } => {
                // `k 'f p_ower m`: f applied k times, unrolled (k is known now).
                let mut v = x;
                for _ in 0..count.max(0) {
                    v = self.apply((*f).clone(), v)?;
                }
                Ok(v)
            }
            other => err(format!("a {} cannot be applied", other.what())),
        }
    }

    /// A primitive applied to one more argument: a function argument
    /// is kept (`'+ r_/`), a data argument completes a monadic
    /// primitive, and a dyadic one waits for its right argument as a
    /// closure over the left.
    fn prim(&mut self, mut p: Prim, x: V) -> Result<V> {
        if let V::Prim(f) = &x {
            if takes_functions(&p.name) {
                p.funs.push(f.clone());
                return Ok(V::Prim(p));
            }
        }
        if p.name == "e_ach" && matches!(x, V::Fun(_) | V::Prim(_) | V::Waiting { .. }) {
            return Ok(V::Prim(Prim {
                name: "e_ach".into(),
                axes: Vec::new(),
                funs: vec![Prim {
                    name: "%each".into(),
                    axes: Vec::new(),
                    funs: Vec::new(),
                }],
            }))
            .map(|v| match v {
                V::Prim(mut q) => {
                    q.funs.clear();
                    V::Waiting {
                        prim: q,
                        left: Box::new(x),
                    }
                }
                v => v,
            });
        }
        if p.name == "p_ower" && matches!(x, V::Fun(_) | V::Prim(_) | V::Waiting { .. }) {
            // `'f p_ower`: the function kept until the count and the data arrive.
            return Ok(V::Power {
                f: Box::new(x),
                count: None,
            });
        }
        if is_dyadic(&p.name) {
            // The left argument arrives first: wait for the right one.
            return Ok(V::Waiting {
                prim: p,
                left: Box::new(x),
            });
        }
        self.monadic(&p, x)
    }

    fn monadic(&mut self, p: &Prim, x: V) -> Result<V> {
        match p.name.as_str() {
            "u_se<" => Ok(V::Unit),
            "p_i" => Ok(V::Static(Array::floats(Shape::scalar(), vec![std::f64::consts::PI]), false)),
            "n_eg" | "a_bs" | "e_xp" | "l_og" | "c_os" | "s_in" | "s_qrt" => {
                let op = match p.name.as_str() {
                    "n_eg" => MapOp::Neg,
                    "a_bs" => MapOp::Abs,
                    "e_xp" => MapOp::Exp,
                    "c_os" => MapOp::Cos,
                    "s_in" => MapOp::Sin,
                    "s_qrt" => MapOp::Sqrt,
                    _ => MapOp::Log,
                };
                if let V::Static(a, true) = &x {
                    if a.scalar.is_int() && matches!(op, MapOp::Neg | MapOp::Abs) {
                        return Ok(V::Static(self.fold("n", vec![a.clone()], |ids| Op::Map { op, args: ids })?, true));
                    }
                }
                let id = self.node(x, "x")?;
                let (id, _) = self.float_if_needed(id, matches!(op, MapOp::Exp | MapOp::Log | MapOp::Cos | MapOp::Sin | MapOp::Sqrt))?;
                let base = p.name.replace('_', "");
                Ok(V::Node(self.push(&base, None, Op::Map { op, args: vec![id] })?))
            }
            "f_loat" => match x {
                V::Static(a, _) => Ok(V::Static(
                    self.fold("f", vec![a], |ids| Op::Cast {
                        op: xetal_gpu_xir::ir::CastOp { to: Scalar::F64 },
                        arg: ids[0],
                    })?,
                    false,
                )),
                V::Node(id) => {
                    if self.ty(id).scalar.is_float() {
                        Ok(V::Node(id))
                    } else {
                        Ok(V::Node(self.push(
                            "f",
                            None,
                            Op::Cast {
                                op: xetal_gpu_xir::ir::CastOp { to: Scalar::F64 },
                                arg: id,
                            },
                        )?))
                    }
                }
                other => err(format!("f_loat of a {}", other.what())),
            },
            "f_loor" => match x {
                V::Static(a, shapey) if a.scalar.is_int() => Ok(V::Static(a, shapey)),
                V::Static(a, _) => Ok(V::Static(
                    self.fold("fl", vec![a], |ids| Op::Cast {
                        op: xetal_gpu_xir::ir::CastOp { to: Scalar::I64 },
                        arg: ids[0],
                    })?,
                    false,
                )),
                x => {
                    let id = self.node(x, "x")?;
                    if self.ty(id).scalar.is_int() {
                        return Ok(V::Node(id));
                    }
                    Ok(V::Node(self.push(
                        "fl",
                        None,
                        Op::Cast {
                            op: xetal_gpu_xir::ir::CastOp { to: Scalar::I64 },
                            arg: id,
                        },
                    )?))
                }
            },
            "s_hape" if matches!(x, V::Str(_)) => {
                let V::Str(s) = x else { unreachable!() };
                Ok(V::Static(Array::ints(Shape(vec![1]), vec![s.chars().count() as i64]), true))
            }
            "t_ally" if matches!(x, V::Str(_)) => {
                let V::Str(s) = x else { unreachable!() };
                Ok(V::Static(Array::ints(Shape::scalar(), vec![s.chars().count() as i64]), true))
            }
            "s_hape" => {
                let s = self.shape_of(&x)?;
                Ok(V::Static(
                    Array::ints(Shape(vec![s.rank()]), s.0.iter().map(|d| *d as i64).collect()),
                    true,
                ))
            }
            "t_ally" => {
                let s = self.shape_of(&x)?;
                Ok(V::Static(
                    Array::ints(Shape::scalar(), vec![s.0.first().copied().unwrap_or(1) as i64]),
                    true,
                ))
            }
            "f_irst" => match x {
                V::Static(a, shapey) => {
                    let one = self.fold("first", vec![a], |ids| Op::Take {
                        n: 1,
                        axis: 1,
                        arg: ids[0],
                    })?;
                    scalar_of(one, shapey)
                }
                V::Node(id) => {
                    let t = self.push("first", None, Op::Take { n: 1, axis: 1, arg: id })?;
                    let r = self.push("firstv", None, Op::Ravel { arg: t })?;
                    Ok(V::Node(r))
                }
                other => err(format!("f_irst of a {}", other.what())),
            },
            "r_ange" | "o_ffsets" => {
                let n = self.int_of(&x, &p.name)?;
                let from = if p.name == "r_ange" { 1 } else { 0 };
                Ok(V::Static(
                    Array::ints(Shape(vec![n.max(0) as usize]), (0..n.max(0)).map(|i| i + from).collect()),
                    false,
                ))
            }
            "r_avel" => match x {
                V::Static(a, shapey) => {
                    let n = a.shape.len();
                    Ok(V::Static(Array::new(a.scalar, Shape(vec![n]), a.data).expect("same items"), shapey))
                }
                V::Node(id) => Ok(V::Node(self.push("rv", None, Op::Ravel { arg: id })?)),
                other => err(format!("r_avel of a {}", other.what())),
            },
            "o_\\" => {
                let id = self.node(x, "x")?;
                Ok(V::Node(self.push(
                    "t",
                    None,
                    Op::Transpose {
                        perm: axes_swap(&p.axes),
                        arg: id,
                    },
                )?))
            }
            "n_umbers" => match x {
                V::Str(s) => {
                    let items: Vec<f64> = s
                        .split_whitespace()
                        .map(|t| t.parse::<f64>().map_err(|_| Error(format!("n_umbers: not a number: `{t}`"))))
                        .collect::<Result<_>>()?;
                    let n = items.len();
                    Ok(V::Static(Array::floats(Shape(vec![n]), items), false))
                }
                other => err(format!("n_umbers of a {}", other.what())),
            },
            "[]N_GET" => match x {
                V::Str(path) => {
                    let full = self.dir.join(&path);
                    self.files_read.push(full.clone());
                    std::fs::read_to_string(&full)
                        .map(V::Str)
                        .map_err(|e| Error(format!("[]N_GET: cannot read {}: {e}", full.display())))
                }
                other => err(format!("[]N_GET of a {}", other.what())),
            },
            "r_/" => {
                let f = p.funs.first().ok_or_else(|| Error("r_/ needs a function: '+ r_/".into()))?;
                let op = reduce_op(&f.name)?;
                let x = self.bool_to_int(x)?;
                if let V::Static(a, true) = &x {
                    if a.shape.rank() == 1 && a.scalar.is_int() {
                        return Ok(V::Static(
                            self.fold("r", vec![a.clone()], |ids| Op::Reduce {
                                op,
                                arg: ids[0],
                                axis: None,
                            })?,
                            true,
                        ));
                    }
                }
                let id = self.node(x, "x")?;
                let rank = self.ty(id).shape.rank();
                let axis = match p.axes.as_slice() {
                    [] if rank <= 1 => None,
                    [] => Some(1),
                    [a] => Some(*a as usize),
                    axes if axes.len() == rank => None,
                    axes => {
                        // Several axes: each folded in turn, the highest first so the others keep their numbers.
                        let mut axes: Vec<usize> = axes.iter().map(|a| *a as usize).collect();
                        axes.sort_unstable();
                        let mut cur = id;
                        for a in axes.into_iter().rev() {
                            cur = self.push(
                                &format!("{}r", op.name()),
                                None,
                                Op::Reduce {
                                    op,
                                    arg: cur,
                                    axis: Some(a),
                                },
                            )?;
                        }
                        return Ok(V::Node(cur));
                    }
                };
                let axis = if axis == Some(1) && rank == 1 { None } else { axis };
                Ok(V::Node(self.push(
                    &format!("{}r", op.name()),
                    None,
                    Op::Reduce { op, arg: id, axis },
                )?))
            }
            "s_\\" => err("s_\\ (a scan) is not in the IR yet"),
            "w_here" => err("w_here gives a shape the data decides, which the device cannot (outside the subset)"),
            other => err(format!("`{other}` is outside the subset the device runs")),
        }
    }

    /// A dyadic primitive with both arguments.
    fn dyadic(&mut self, p: &Prim, l: V, r: V) -> Result<V> {
        let axis = p.axes.first().map(|a| *a as usize);
        match p.name.as_str() {
            "+" | "-" | "*" | "/" | "m_ax" | "m_in" | "d_iv" | "m_od" | "=" | "!=" | "<" | "<=" | ">" | ">=" => {
                let op = match p.name.as_str() {
                    "+" => MapOp::Add,
                    "-" => MapOp::Sub,
                    "*" => MapOp::Mul,
                    "/" => MapOp::Div,
                    "m_ax" => MapOp::Max,
                    "m_in" => MapOp::Min,
                    "d_iv" => MapOp::IDiv,
                    "m_od" => MapOp::Mod,
                    "=" => MapOp::Eq,
                    "!=" => MapOp::Ne,
                    "<" => MapOp::Lt,
                    "<=" => MapOp::Le,
                    ">" => MapOp::Gt,
                    _ => MapOp::Ge,
                };
                // A Bool counts as 0 or 1 in arithmetic, as X_eTaL has it (a mask times a value).
                let (l, r) = if !matches!(op.kind(), xetal_gpu_xir::ir::MapKind::EqCompare | xetal_gpu_xir::ir::MapKind::Logic) {
                    (self.bool_to_int(l)?, self.bool_to_int(r)?)
                } else {
                    (l, r)
                };
                // A literal adapts to the type it meets: an Int beside a Float is a Float (`x m_ax 0`).
                let (l, r) = self.adapt_literal(l, r)?;
                // Int arithmetic on shape-like values (shapes, counts) folds; data goes to the device.
                if let (V::Static(a, true), V::Static(b, true)) = (&l, &r) {
                    if a.scalar.is_int()
                        && b.scalar.is_int()
                        && !matches!(op, MapOp::Div)
                        && !matches!(
                            op.kind(),
                            xetal_gpu_xir::ir::MapKind::EqCompare | xetal_gpu_xir::ir::MapKind::OrdCompare
                        )
                    {
                        return Ok(V::Static(
                            self.fold("i", vec![a.clone(), b.clone()], |ids| Op::Map { op, args: ids })?,
                            true,
                        ));
                    }
                }
                // `/` gives a Float, so Ints are made Floats first: statically when known now.
                let (l, r) = if matches!(op, MapOp::Div) {
                    (self.made_float(l)?, self.made_float(r)?)
                } else {
                    (l, r)
                };
                let a = self.node(l, "l")?;
                let b = self.node(r, "r")?;
                Ok(V::Node(self.push(op.name(), None, Op::Map { op, args: vec![a, b] })?))
            }
            "^" => {
                let (l, r) = (self.made_float(l)?, self.made_float(r)?);
                let a = self.node(l, "l")?;
                let b = self.node(r, "r")?;
                Ok(V::Node(self.push(
                    "pow",
                    None,
                    Op::Map {
                        op: MapOp::Pow,
                        args: vec![a, b],
                    },
                )?))
            }
            "r_eshape" => {
                let dims: Vec<usize> = self.ints_of(&l, "r_eshape")?.into_iter().map(|d| d.max(0) as usize).collect();
                match r {
                    V::Static(a, _) => Ok(V::Static(
                        self.fold("rs", vec![a], |ids| Op::Reshape {
                            shape: Shape(dims.clone()),
                            arg: ids[0],
                        })?,
                        false,
                    )),
                    V::Node(id) => Ok(V::Node(self.push(
                        "rs",
                        None,
                        Op::Reshape {
                            shape: Shape(dims),
                            arg: id,
                        },
                    )?)),
                    other => err(format!("r_eshape of a {}", other.what())),
                }
            }
            "c_at" => {
                let axis = axis.unwrap_or(1);
                match (&l, &r) {
                    // Shapes and counts (Int vectors known now) stay shape-like.
                    (V::Static(a, sa), V::Static(b, sb))
                        if a.scalar.is_int() && b.scalar.is_int() && a.shape.rank() <= 1 && b.shape.rank() <= 1 && (*sa || *sb) =>
                    {
                        let (Data::Int(x), Data::Int(y)) = (&a.data, &b.data) else {
                            unreachable!()
                        };
                        let mut v = x.clone();
                        v.extend_from_slice(y);
                        let n = v.len();
                        Ok(V::Static(Array::ints(Shape(vec![n]), v), true))
                    }
                    // Data known now joins now (weights with their bias row); the rest on the device.
                    (V::Static(_, _), V::Static(_, _)) => {
                        let (l, r) = self.adapt_literal(l, r)?;
                        let (V::Static(a, _), V::Static(b, _)) = (l, r) else {
                            unreachable!()
                        };
                        Ok(V::Static(
                            self.fold("cat", vec![a, b], |ids| Op::Concat {
                                axis,
                                a: ids[0],
                                b: ids[1],
                            })?,
                            false,
                        ))
                    }
                    // An empty side known now (a loop's first accumulator) adds
                    // nothing: the other side is the result, and no empty
                    // constant reaches the device.
                    (V::Static(a, _), V::Node(_)) if a.shape.is_empty() && a.shape.rank() == self.rank_of(&r) => Ok(r),
                    (V::Node(_), V::Static(b, _)) if b.shape.is_empty() && b.shape.rank() == self.rank_of(&l) => Ok(l),
                    _ => {
                        let (l, r) = self.adapt_literal(l, r)?;
                        let a = self.node(l, "a")?;
                        let b = self.node(r, "b")?;
                        Ok(V::Node(self.push("cat", None, Op::Concat { axis, a, b })?))
                    }
                }
            }
            "t_ake" | "d_rop" => {
                let n = self.int_of(&l, &p.name)?;
                let axis = axis.unwrap_or(1);
                let take = p.name == "t_ake";
                let base = if take { "take" } else { "drop" };
                match r {
                    V::Static(a, true) => Ok(V::Static(
                        self.fold(base, vec![a], |ids| {
                            if take {
                                Op::Take { n, axis, arg: ids[0] }
                            } else {
                                Op::Drop { n, axis, arg: ids[0] }
                            }
                        })?,
                        true,
                    )),
                    other => {
                        let id = self.node(other, "x")?;
                        Ok(V::Node(self.push(
                            base,
                            None,
                            if take {
                                Op::Take { n, axis, arg: id }
                            } else {
                                Op::Drop { n, axis, arg: id }
                            },
                        )?))
                    }
                }
            }
            "s_elect" => {
                let axis = axis.unwrap_or(1);
                match (l, r) {
                    (V::Static(i, _), V::Static(a, true)) => Ok(V::Static(
                        self.fold("sel", vec![i, a], |ids| Op::Lookup {
                            axis,
                            idx: ids[0],
                            arg: ids[1],
                        })?,
                        true,
                    )),
                    (i, a) => {
                        let idx = self.node(i, "ids")?;
                        let arg = self.node(a, "table")?;
                        Ok(V::Node(self.push("sel", None, Op::Lookup { axis, idx, arg })?))
                    }
                }
            }
            "o_-" => {
                let amounts = self.ints_of(&l, "o_-")?;
                let axis = axis.unwrap_or(1);
                let id = self.node(r, "x")?;
                Ok(V::Node(self.push("rot", None, Op::Rotate { axis, amounts, arg: id })?))
            }
            "t_ranspose" => {
                let perm: Vec<usize> = self.ints_of(&l, "t_ranspose")?.into_iter().map(|a| a.max(0) as usize).collect();
                let id = self.node(r, "x")?;
                Ok(V::Node(self.push("tr", None, Op::Transpose { perm: Some(perm), arg: id })?))
            }
            "i_nner" => {
                let [f, g] = p.funs.as_slice() else {
                    return err("i_nner needs two functions: '+ '* i_nner");
                };
                // `x '+ '* i_nner w` arrives as i_nner holding `*` then `+`.
                if !((f.name == "*" && g.name == "+") || (f.name == "+" && g.name == "*")) {
                    return err(format!(
                        "'{} '{} i_nner is outside the subset (only '+ '* i_nner, a product)",
                        g.name, f.name
                    ));
                }
                let a = self.node(l, "a")?;
                let b = self.node(r, "b")?;
                let (a, b) = self.same_scalar(a, b, false)?;
                Ok(V::Node(self.push("mm", None, Op::Matmul { a, b })?))
            }
            "t_able" => {
                let f = p
                    .funs
                    .first()
                    .ok_or_else(|| Error("t_able needs a function: 'l_eft t_able".into()))?;
                let op = match f.name.as_str() {
                    "l_eft" => TableOp::Left,
                    "r_ight" => TableOp::Right,
                    name => TableOp::parse(&map_name(name)?).ok_or_else(|| Error(format!("'{name} t_able is outside the subset")))?,
                };
                let a = self.node(l, "a")?;
                let b = self.node(r, "b")?;
                let (a, b) = if matches!(op, TableOp::Map(_)) {
                    self.same_scalar(a, b, false)?
                } else {
                    (a, b)
                };
                Ok(V::Node(self.push("tb", None, Op::Table { op, a, b })?))
            }
            "u_se<" => Ok(V::Unit),
            "e_ach" => {
                // `'f e_ach v`: f on each item of v, known now (r_ange n, a strand); each
                // result a single value, the results a vector (a nested result is not an array).
                let f = l;
                let V::Static(items, _) = r else {
                    return err("e_ach over a value the device computes is outside the subset (its items must be known now)");
                };
                if items.shape.rank() != 1 {
                    return err("e_ach over other than a vector is outside the subset");
                }
                let n = items.shape.len();
                let mut acc: Option<V> = None;
                for i in 0..n {
                    let item = match &items.data {
                        Data::Int(v) => Array::ints(Shape::scalar(), vec![v[i]]),
                        Data::Float(v) => Array::floats(Shape::scalar(), vec![v[i]]),
                        Data::Bool(_) => return err("e_ach over Bools is outside the subset"),
                    };
                    let shapey = items.scalar.is_int();
                    let v = self.apply(f.clone(), V::Static(item, shapey))?;
                    let rank = self.shape_of(&v)?.rank();
                    if rank != 0 {
                        return err("e_ach whose results are arrays gives a nested array, which the device cannot (outside the subset)");
                    }
                    acc = Some(match acc {
                        None => v,
                        Some(prev) => {
                            let (prev, v) = self.adapt_literal(prev, v)?;
                            let a = self.node(prev, "each")?;
                            let b = self.node(v, "each")?;
                            V::Node(self.push("each", None, Op::Concat { axis: 1, a, b })?)
                        }
                    });
                }
                match acc {
                    Some(V::Node(id)) if n == 1 => Ok(V::Node(self.push(
                        "each",
                        None,
                        Op::Reshape {
                            shape: Shape(vec![1]),
                            arg: id,
                        },
                    )?)),
                    Some(v) => Ok(v),
                    None => err("e_ach over nothing"),
                }
            }
            "p_ower" => err("p_ower without its function"),
            other => err(format!("`{other}` is outside the subset the device runs")),
        }
    }

    // ----- helpers

    fn shape_of(&self, v: &V) -> Result<Shape> {
        match v {
            V::Static(a, _) => Ok(a.shape.clone()),
            V::Node(id) => Ok(self.ty(*id).shape.clone()),
            other => err(format!("a {} has no shape", other.what())),
        }
    }

    fn int_of(&self, v: &V, what: &str) -> Result<i64> {
        match v {
            V::Static(a, _) if a.shape.len() == 1 => match &a.data {
                Data::Int(x) => Ok(x[0]),
                _ => err(format!("{what} needs a whole number, got a Float")),
            },
            V::Static(a, _) => err(format!("{what} needs one whole number, got {}", a.ty())),
            V::Node(_) => err(format!(
                "{what} needs a count known now, got one the device computes (a dynamic shape is outside the subset)"
            )),
            other => err(format!("{what} needs a whole number, got a {}", other.what())),
        }
    }

    fn ints_of(&self, v: &V, what: &str) -> Result<Vec<i64>> {
        match v {
            V::Static(a, _) => match &a.data {
                Data::Int(x) => Ok(x.clone()),
                _ => err(format!("{what} needs whole numbers, got Floats")),
            },
            V::Node(_) => err(format!(
                "{what} needs a shape known now, got one the device computes (a dynamic shape is outside the subset)"
            )),
            other => err(format!("{what} needs whole numbers, got a {}", other.what())),
        }
    }

    /// Two operands of one scalar type, as X_eTaL's arithmetic needs
    /// (an Int beside a Float is a type error there, except that `/`
    /// gives a Float, so Ints are cast first).
    fn same_scalar(&mut self, a: ValueId, b: ValueId, to_float: bool) -> Result<(ValueId, ValueId)> {
        let (sa, sb) = (self.ty(a).scalar, self.ty(b).scalar);
        let a = if to_float && sa.is_int() {
            self.push(
                "f",
                None,
                Op::Cast {
                    op: xetal_gpu_xir::ir::CastOp { to: Scalar::F64 },
                    arg: a,
                },
            )?
        } else {
            a
        };
        let b = if to_float && sb.is_int() {
            self.push(
                "f",
                None,
                Op::Cast {
                    op: xetal_gpu_xir::ir::CastOp { to: Scalar::F64 },
                    arg: b,
                },
            )?
        } else {
            b
        };
        Ok((a, b))
    }

    /// A Bool value as an Int (0 or 1); anything else as it is.
    fn bool_to_int(&mut self, v: V) -> Result<V> {
        match v {
            V::Static(a, _) if a.scalar == Scalar::Bool => Ok(V::Static(
                self.fold("b", vec![a], |ids| Op::Cast {
                    op: xetal_gpu_xir::ir::CastOp { to: Scalar::I64 },
                    arg: ids[0],
                })?,
                false,
            )),
            V::Node(id) if self.ty(id).scalar == Scalar::Bool => Ok(V::Node(self.push(
                "b",
                None,
                Op::Cast {
                    op: xetal_gpu_xir::ir::CastOp { to: Scalar::I64 },
                    arg: id,
                },
            )?)),
            v => Ok(v),
        }
    }

    /// A value as a Float: a static one cast now, a device one by a kernel.
    fn made_float(&mut self, v: V) -> Result<V> {
        match v {
            V::Static(a, _) if a.scalar.is_int() => Ok(V::Static(
                self.fold("f", vec![a], |ids| Op::Cast {
                    op: xetal_gpu_xir::ir::CastOp { to: Scalar::F64 },
                    arg: ids[0],
                })?,
                false,
            )),
            V::Node(id) if self.ty(id).scalar.is_int() => Ok(V::Node(self.push(
                "f",
                None,
                Op::Cast {
                    op: xetal_gpu_xir::ir::CastOp { to: Scalar::F64 },
                    arg: id,
                },
            )?)),
            v => Ok(v),
        }
    }

    /// A static Int single value beside a Float becomes a Float, as a
    /// literal in X_eTaL takes the type of what it meets.
    fn adapt_literal(&self, l: V, r: V) -> Result<(V, V)> {
        let is_float = |s: &Self, v: &V| match v {
            V::Static(a, _) => a.scalar.is_float(),
            V::Node(id) => s.ty(*id).scalar.is_float(),
            _ => false,
        };
        let to_float = |v: V| -> V {
            match v {
                V::Static(a, _) if a.scalar.is_int() && a.shape.is_scalar() => {
                    let Data::Int(x) = &a.data else { unreachable!() };
                    V::Static(Array::floats(Shape::scalar(), vec![x[0] as f64]), false)
                }
                v => v,
            }
        };
        let (lf, rf) = (is_float(self, &l), is_float(self, &r));
        Ok(match (lf, rf) {
            (true, false) => (l, to_float(r)),
            (false, true) => (to_float(l), r),
            _ => (l, r),
        })
    }

    fn float_if_needed(&mut self, id: ValueId, needed: bool) -> Result<(ValueId, bool)> {
        if needed && self.ty(id).scalar.is_int() {
            Ok((
                self.push(
                    "f",
                    None,
                    Op::Cast {
                        op: xetal_gpu_xir::ir::CastOp { to: Scalar::F64 },
                        arg: id,
                    },
                )?,
                true,
            ))
        } else {
            Ok((id, false))
        }
    }
}

/// A one-item array as a single value.
fn scalar_of(a: Array, shapey: bool) -> Result<V> {
    Array::new(a.scalar, Shape::scalar(), a.data).map(|a| V::Static(a, shapey))
}

fn takes_functions(name: &str) -> bool {
    matches!(name, "r_/" | "i_nner" | "t_able" | "s_\\" | "p_ower" | "e_ach")
}

fn is_dyadic(name: &str) -> bool {
    matches!(
        name,
        "+" | "-"
            | "*"
            | "/"
            | "^"
            | "m_ax"
            | "m_in"
            | "d_iv"
            | "m_od"
            | "="
            | "!="
            | "<"
            | "<="
            | ">"
            | ">="
            | "r_eshape"
            | "c_at"
            | "t_ake"
            | "d_rop"
            | "s_elect"
            | "o_-"
            | "t_ranspose"
            | "i_nner"
            | "t_able"
            | "p_ower"
            | "u_se<"
            | "i_ndexOf"
            | "m_atch"
            | "e_ach"
    )
}

fn reduce_op(name: &str) -> Result<ReduceOp> {
    match name {
        "+" => Ok(ReduceOp::Add),
        "*" => Ok(ReduceOp::Mul),
        "m_ax" => Ok(ReduceOp::Max),
        "m_in" => Ok(ReduceOp::Min),
        other => err(format!("'{other} r_/ is outside the subset (+, *, m_ax, m_in)")),
    }
}

fn map_name(name: &str) -> Result<String> {
    Ok(match name {
        "+" => "add",
        "-" => "sub",
        "*" => "mul",
        "/" => "div",
        "m_ax" => "max",
        "m_in" => "min",
        "=" => "eq",
        "!=" => "ne",
        "<" => "lt",
        "<=" => "le",
        ">" => "gt",
        ">=" => "ge",
        other => return err(format!("'{other} is outside the subset as a table function")),
    }
    .to_string())
}

/// `o_\_jk`: the two axes swapped, as a permutation; none for the whole reversal.
fn axes_swap(axes: &[u8]) -> Option<Vec<usize>> {
    match axes {
        [j, k] => {
            let (j, k) = (*j as usize, *k as usize);
            let n = j.max(k);
            let mut perm: Vec<usize> = (1..=n).collect();
            perm.swap(j - 1, k - 1);
            Some(perm)
        }
        _ => None,
    }
}
