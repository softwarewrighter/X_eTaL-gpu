//! gpu: XIR programs run on OpenCL devices, or on the reference
//! interpreter, from X_eTaL code through X_eTaL-extensions' ABI V1.
//!
//! ABI V1 functions take at most two arguments and give one result,
//! so the extension keeps a session per thread: a program loaded
//! (`load`), its inputs bound one by one (`bind`), a schedule
//! (`schedule`, else the defaults), a run on a device (`run`), and
//! its outputs read one by one (`output`); `explain` and `kernel`
//! describe the last plan. The facade, `lib/Gpu.xtl`, gives each an
//! X_eTaL name and type.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use xetal_ext_sdk::{float_vector, number, text, Array as ExtArray, ArrayData, OwnedError, Value};
use xetal_gpu_opencl::{explain as explain_plan, plan, Plan, Schedule};
use xetal_gpu_runtime::{devices as list_devices, Prepared};
use xetal_gpu_xir::{
    binary, check, run as interpret, text as xir_text, Array, Checked, Data, Scalar, Shape, Ty,
};

/// A function given as source (`gp:a_ccel<`): lowered when its
/// arguments are bound, since their shapes are its inputs' shapes;
/// each distinct set of argument types lowers once.
struct Source {
    function: String,
    params: Vec<String>,
    kinds: Vec<String>,
    uses: String,
    dir: std::path::PathBuf,
    lowered: HashMap<Vec<Ty>, Checked>,
}

#[derive(Default)]
struct Session {
    program: Option<Checked>,
    source: Option<Source>,
    inputs: HashMap<String, Array>,
    schedule: Schedule,
    outputs: Vec<Array>,
    last_plan: Option<Plan>,
    prepared: Option<(usize, Prepared)>,
    dirty: HashSet<String>,
}

thread_local! {
    static SESSION: RefCell<Session> = RefCell::new(Session::default());
}

fn failure(e: impl std::fmt::Display) -> OwnedError {
    OwnedError::failure(e.to_string())
}

/// A whole number given as an Int or an integral Float.
fn whole(v: &Value, what: &str) -> Result<i64, OwnedError> {
    let x = number(v)?;
    if x.fract() != 0.0 {
        return Err(OwnedError::invalid_argument(format!(
            "{what} is a whole number, got {x}"
        )));
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(x as i64)
}

/// The OpenCL devices, one line each, as `xetal-gpu devices` lists them.
#[allow(clippy::unnecessary_wraps)]
fn devices(_: &[Value]) -> Result<Value, OwnedError> {
    let found = list_devices().map_err(failure)?;
    if found.is_empty() {
        return Ok(Value::Text("no OpenCL device found".into()));
    }
    let lines: Vec<String> = found
        .iter()
        .map(|d| {
            format!(
                "opencl:{}  {} ({}, {}; {}), {} compute units, work-groups to {}, {}",
                d.index,
                d.name,
                d.kind,
                d.vendor,
                d.version,
                d.compute_units,
                d.max_work_group,
                if d.fp64 { "fp64" } else { "no fp64" }
            )
        })
        .collect();
    Ok(Value::Text(lines.join("\n")))
}

/// Load an XIR program (its text): parsed and checked; the session's
/// inputs and outputs are cleared. The number of inputs.
fn load(args: &[Value]) -> Result<Value, OwnedError> {
    let src = text(&args[0])?;
    let checked = check(xir_text::parse(&src).map_err(OwnedError::invalid_argument_from)?)
        .map_err(OwnedError::invalid_argument_from)?;
    let n = checked.program.inputs().len();
    SESSION.with(|s| {
        let mut s = s.borrow_mut();
        s.program = Some(checked);
        s.inputs.clear();
        s.outputs.clear();
        s.last_plan = None;
        s.prepared = None;
        s.dirty.clear();
    });
    Ok(Value::Int(i64::try_from(n).unwrap_or(i64::MAX)))
}

/// The function's parameter names, from its text (`{ x w -> ...}`).
fn parameters(function: &str) -> Result<Vec<String>, OwnedError> {
    let body = function.trim();
    let inner = body.strip_prefix('{').ok_or_else(|| {
        OwnedError::invalid_argument(
            "the function must be written `{ x -> ... }` or `{ x w -> ... }`",
        )
    })?;
    let head = inner
        .split("->")
        .next()
        .ok_or_else(|| OwnedError::invalid_argument("the function has no `->`"))?;
    let params: Vec<String> = head.split_whitespace().map(str::to_string).collect();
    if params.is_empty()
        || params.len() > 2
        || params
            .iter()
            .any(|p| p == "@" || p.contains(',') || p.contains('('))
    {
        return Err(OwnedError::invalid_argument(
            "the function takes one or two named arrays (a tuple pattern or @ is not accelerated)",
        ));
    }
    Ok(params)
}

/// Load a function from its source text, to be lowered when its
/// arguments are bound: the file it is written in gives its imports
/// (every `u_se<` line) and the directory its libraries resolve from;
/// the kinds (`"float"`, `"int float"`, or empty for all Floats) say
/// how each argument is taken. The number of parameters.
fn lower(args: &[Value]) -> Result<Value, OwnedError> {
    let function = text(&args[0])?;
    let spec = text(&args[1])?;
    // "FILE" or "FILE|KINDS": the file the call is written in, then the kinds.
    let (file, kinds_text) = match spec.split_once('|') {
        Some((f, k)) => (f.to_string(), k.to_string()),
        None => (spec.clone(), String::new()),
    };
    let params = parameters(&function)?;
    let mut kinds: Vec<String> = kinds_text.split_whitespace().map(str::to_string).collect();
    if kinds.is_empty() {
        kinds = vec!["float".to_string(); params.len()];
    }
    if kinds.len() != params.len() || kinds.iter().any(|k| k != "float" && k != "int") {
        return Err(OwnedError::invalid_argument(format!(
            "the kinds say `{kinds_text}`; one of float or int per argument ({} here)",
            params.len()
        )));
    }
    let path = std::path::Path::new(&file);
    let program_text = std::fs::read_to_string(path).unwrap_or_default();
    let uses: String = program_text
        .lines()
        .filter(|l| l.contains("u_se<") && !l.trim_start().starts_with('#'))
        .map(|l| l.to_string() + "\n")
        .collect();
    let dir = path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let n = params.len();
    SESSION.with(|s| {
        let mut s = s.borrow_mut();
        s.program = None;
        s.source = Some(Source {
            function,
            params,
            kinds,
            uses,
            dir,
            lowered: HashMap::new(),
        });
        s.inputs.clear();
        s.outputs.clear();
        s.last_plan = None;
        s.prepared = None;
        s.dirty.clear();
    });
    Ok(Value::Int(i64::try_from(n).unwrap_or(i64::MAX)))
}

/// Bind input n (from 1, in program order) to an array of numbers of
/// its declared shape; an Int input takes whole numbers only. n.
/// For a function loaded from source, argument n takes the array as
/// it is (its kind says Int or Float), under the parameter's name.
fn bind(args: &[Value]) -> Result<Value, OwnedError> {
    let n = whole(&args[0], "an input's number")?;
    let (shape, items) = float_vector(&args[1])?;
    SESSION.with(|s| {
        let mut s = s.borrow_mut();
        if let Some(source) = &s.source {
            let i = usize::try_from(n - 1)
                .ok()
                .filter(|i| *i < source.params.len())
                .ok_or_else(|| {
                    OwnedError::invalid_argument(format!(
                        "the function takes {} argument(s); no argument {n}",
                        source.params.len()
                    ))
                })?;
            let name = source.params[i].clone();
            let data = if source.kinds[i] == "int" {
                if let Some(x) = items.iter().find(|x| x.fract() != 0.0) {
                    return Err(OwnedError::invalid_argument(format!(
                        "argument {n} ({name}) is int: {x} is not whole"
                    )));
                }
                #[allow(clippy::cast_possible_truncation)]
                Data::Int(items.iter().map(|x| *x as i64).collect())
            } else {
                Data::Float(items)
            };
            let scalar = if source.kinds[i] == "int" {
                Scalar::I64
            } else {
                Scalar::F64
            };
            let array = Array::new(scalar, Shape(shape), data).map_err(failure)?;
            s.dirty.insert(name.clone());
            s.inputs.insert(name, array);
            return Ok(Value::Int(n));
        }
        let checked = s
            .program
            .as_ref()
            .ok_or_else(|| OwnedError::invalid_argument("no program loaded (gp:l_oad first)"))?;
        let inputs = checked.program.inputs();
        let id = usize::try_from(n - 1)
            .ok()
            .and_then(|i| inputs.get(i).copied())
            .ok_or_else(|| {
                OwnedError::invalid_argument(format!(
                    "the program has {} input(s); no input {n}",
                    inputs.len()
                ))
            })?;
        let ty = checked.ty(id).clone();
        let name = checked.program.value(id).name.clone();
        let given = Shape(shape);
        if given.len() != ty.shape.len() || (given.rank() != 0 && given != ty.shape) {
            return Err(OwnedError::invalid_argument(format!(
                "input {n} (%{name}) is {ty}, given shape {given}"
            )));
        }
        let data = match ty.scalar {
            Scalar::F32 | Scalar::F64 => Data::Float(items),
            Scalar::I32 | Scalar::I64 => {
                if let Some(x) = items.iter().find(|x| x.fract() != 0.0) {
                    return Err(OwnedError::invalid_argument(format!(
                        "input {n} (%{name}) is {ty}: {x} is not whole"
                    )));
                }
                #[allow(clippy::cast_possible_truncation)]
                Data::Int(items.iter().map(|x| *x as i64).collect())
            }
            Scalar::Bool => Data::Bool(items.iter().map(|x| *x != 0.0).collect()),
        };
        let array = Array::new(ty.scalar, ty.shape.clone(), data).map_err(failure)?;
        s.dirty.insert(name.clone());
        s.inputs.insert(name, array);
        Ok(Value::Int(n))
    })
}

/// Bind a checked binary array from a host path; only the path crosses the bridge.
fn bind_file(args: &[Value]) -> Result<Value, OwnedError> {
    let n = whole(&args[0], "an input's number")?;
    let path = text(&args[1])?;
    SESSION.with(|s| {
        let mut s = s.borrow_mut();
        let checked = s
            .program
            .as_ref()
            .ok_or_else(|| OwnedError::invalid_argument("no program loaded (gp:l_oad first)"))?;
        let ids = checked.program.inputs();
        let id = n
            .checked_sub(1)
            .and_then(|i| usize::try_from(i).ok())
            .and_then(|i| ids.get(i).copied())
            .ok_or_else(|| {
                OwnedError::invalid_argument(format!(
                    "the program has {} input(s); no input {n}",
                    ids.len()
                ))
            })?;
        let a = binary::for_input(
            binary::read(&path).map_err(OwnedError::invalid_argument_from)?,
            checked.ty(id),
        )
        .map_err(OwnedError::invalid_argument_from)?;
        let name = checked.program.value(id).name.clone();
        s.dirty.insert(name.clone());
        s.inputs.insert(name, a);
        Ok(Value::Int(n))
    })
}

/// The schedule for later runs, as a schedules/*.toml file's text
/// ("" for the defaults). 0.
fn schedule(args: &[Value]) -> Result<Value, OwnedError> {
    let src = text(&args[0])?;
    let sch = Schedule::from_toml(&src).map_err(OwnedError::invalid_argument_from)?;
    SESSION.with(|s| {
        let mut s = s.borrow_mut();
        if s.schedule != sch {
            s.prepared = None;
            s.last_plan = None;
        }
        s.schedule = sch;
    });
    Ok(Value::Int(0))
}

/// Run the loaded program on opencl:d, or on the reference interpreter
/// for d = -1. The number of outputs.
fn run(args: &[Value]) -> Result<Value, OwnedError> {
    run_impl(args, false)
}

/// Reuse the last device's kernels and buffers; upload only rebound inputs.
/// Program, schedule or device changes cause a new preparation.
fn rerun(args: &[Value]) -> Result<Value, OwnedError> {
    run_impl(args, true)
}

fn run_impl(args: &[Value], reuse: bool) -> Result<Value, OwnedError> {
    let d = whole(&args[0], "a device")?;
    SESSION.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let s = &mut *borrow;
        // A function from source: lowered for the bound arguments' types, once per set of types.
        if let Some(source) = s.source.as_mut() {
            let mut params: Vec<(String, Ty)> = Vec::new();
            for name in &source.params {
                let a = s.inputs.get(name).ok_or_else(|| {
                    OwnedError::invalid_argument(format!("argument {name} is not bound (gp:b_ind)"))
                })?;
                params.push((name.clone(), a.ty()));
            }
            let key: Vec<Ty> = params.iter().map(|(_, t)| t.clone()).collect();
            let fresh = !source.lowered.contains_key(&key);
            if fresh {
                let lowered = xetal_gpu_lower::lower_function(
                    &source.function,
                    &params,
                    &source.dir,
                    &source.uses,
                )
                .map_err(|e| OwnedError::invalid_argument(format!("gp:a_ccel<: {e}")))?;
                let checked = check(lowered.program).map_err(failure)?;
                source.lowered.insert(key.clone(), checked);
            }
            let checked = source.lowered[&key].clone();
            if fresh || s.program.as_ref() != Some(&checked) {
                s.program = Some(checked);
                s.last_plan = None;
                s.prepared = None;
            }
        }
        let checked = s
            .program
            .as_ref()
            .ok_or_else(|| OwnedError::invalid_argument("no program loaded (gp:l_oad first)"))?;
        let p = if reuse {
            s.last_plan
                .clone()
                .unwrap_or_else(|| plan(checked, &s.schedule))
        } else {
            plan(checked, &s.schedule)
        };
        let outputs = if d < 0 {
            s.prepared = None;
            interpret(checked, &s.inputs).map_err(failure)?
        } else {
            let index = usize::try_from(d).map_err(failure)?;
            if !reuse || s.prepared.as_ref().map(|(i, _)| *i) != Some(index) {
                let (prepared, _) =
                    Prepared::new(checked, &p, &s.inputs, index).map_err(failure)?;
                s.prepared = Some((index, prepared));
            } else {
                let (_, prepared) = s.prepared.as_mut().expect("matching session");
                for name in &s.dirty {
                    prepared.rebind(name, &s.inputs[name]).map_err(failure)?;
                }
            }
            s.prepared
                .as_mut()
                .expect("prepared session")
                .1
                .run_timed()
                .map_err(failure)?
                .0
        };
        let n = outputs.len();
        s.outputs = outputs;
        s.last_plan = Some(p);
        s.dirty.clear();
        Ok(Value::Int(i64::try_from(n).unwrap_or(i64::MAX)))
    })
}

/// Output i (from 1) of the last run: Ints as Ints, Floats as Floats,
/// Bools as Bools, a single value as a scalar.
fn output(args: &[Value]) -> Result<Value, OwnedError> {
    let i = whole(&args[0], "an output's number")?;
    SESSION.with(|s| {
        let s = s.borrow();
        let a = usize::try_from(i - 1)
            .ok()
            .and_then(|k| s.outputs.get(k))
            .ok_or_else(|| {
                OwnedError::invalid_argument(format!(
                    "the last run has {} output(s); no output {i}",
                    s.outputs.len()
                ))
            })?;
        to_value(a)
    })
}

fn to_value(a: &Array) -> Result<Value, OwnedError> {
    if a.shape.is_scalar() {
        return Ok(match &a.data {
            Data::Int(v) => Value::Int(v[0]),
            Data::Float(v) => Value::Float(v[0]),
            Data::Bool(v) => Value::Bool(v[0]),
        });
    }
    let data = match &a.data {
        Data::Int(v) => ArrayData::Int(v.clone()),
        Data::Float(v) => ArrayData::Float(v.clone()),
        Data::Bool(v) => ArrayData::Bool(v.clone()),
    };
    Ok(Value::Array(
        ExtArray::new(a.shape.0.clone(), data).map_err(failure)?,
    ))
}

/// The last run's plan in words (`xetal-gpu explain`).
fn explain(_: &[Value]) -> Result<Value, OwnedError> {
    SESSION.with(|s| {
        let s = s.borrow();
        match (&s.program, &s.last_plan) {
            (Some(c), Some(p)) => Ok(Value::Text(explain_plan(c, p))),
            _ => Err(OwnedError::invalid_argument(
                "nothing has run yet (gp:r_un first)",
            )),
        }
    })
}

/// The last run's OpenCL C (`xetal-gpu kernel`).
fn kernel(_: &[Value]) -> Result<Value, OwnedError> {
    SESSION.with(|s| match &s.borrow().last_plan {
        Some(p) => Ok(Value::Text(p.source.clone())),
        None => Err(OwnedError::invalid_argument(
            "nothing has run yet (gp:r_un first)",
        )),
    })
}

/// The SDK's error constructors take a message; ours carry a Display.
trait FromDisplay {
    fn invalid_argument_from(e: impl std::fmt::Display) -> Self;
}

impl FromDisplay for OwnedError {
    fn invalid_argument_from(e: impl std::fmt::Display) -> Self {
        OwnedError::invalid_argument(e.to_string())
    }
}

xetal_ext_sdk::xetal_extension! {
    name: "gpu",
    version: env!("CARGO_PKG_VERSION"),
    functions: {
        devices: 0, "Unit -> Char", "The OpenCL devices, one line each (opencl:N).";
        load: 1, "Char -> Int", "Load an XIR program (its text); the number of inputs.";
        lower: 2, "Char -> Char -> Int", "Load a function from its X_eTaL text (and 'FILE|KINDS': the file it is written in, the arguments' kinds); lowered when bound; the number of parameters.";
        bind: 2, "Num a => Int -> a -> Int", "Bind input n (from 1) to an array of its shape.";
        bind_file: 2, "Int -> Char -> Int", "Bind input n from a checked XTLARR01 binary array file.";
        schedule: 1, "Char -> Int", "The schedule for later runs (a schedules/*.toml text; empty for the defaults).";
        run: 1, "Int -> Int", "Run on opencl:d (-1: the reference interpreter); the number of outputs.";
        rerun: 1, "Int -> Int", "Rerun with resident inputs and kernels; upload only rebound inputs.";
        output: 1, "Num a => Int -> a", "Output i (from 1) of the last run.";
        explain: 0, "Unit -> Char", "The last run's plan in words.";
        kernel: 0, "Unit -> Char", "The last run's OpenCL C.";
    }
}
