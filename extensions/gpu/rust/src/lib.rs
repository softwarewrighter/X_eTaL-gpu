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
use std::collections::HashMap;

use xetal_ext_sdk::{float_vector, number, text, Array as ExtArray, ArrayData, OwnedError, Value};
use xetal_gpu_opencl::{explain as explain_plan, plan, Plan, Schedule};
use xetal_gpu_runtime::{devices as list_devices, execute};
use xetal_gpu_xir::{
    check, run as interpret, text as xir_text, Array, Checked, Data, Scalar, Shape,
};

#[derive(Default)]
struct Session {
    program: Option<Checked>,
    inputs: HashMap<String, Array>,
    schedule: Schedule,
    outputs: Vec<Array>,
    last_plan: Option<Plan>,
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
    });
    Ok(Value::Int(i64::try_from(n).unwrap_or(i64::MAX)))
}

/// Bind input n (from 1, in program order) to an array of numbers of
/// its declared shape; an Int input takes whole numbers only. n.
fn bind(args: &[Value]) -> Result<Value, OwnedError> {
    let n = whole(&args[0], "an input's number")?;
    let (shape, items) = float_vector(&args[1])?;
    SESSION.with(|s| {
        let mut s = s.borrow_mut();
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
        s.inputs.insert(name, array);
        Ok(Value::Int(n))
    })
}

/// The schedule for later runs, as a schedules/*.toml file's text
/// ("" for the defaults). 0.
fn schedule(args: &[Value]) -> Result<Value, OwnedError> {
    let src = text(&args[0])?;
    let sch = Schedule::from_toml(&src).map_err(OwnedError::invalid_argument_from)?;
    SESSION.with(|s| s.borrow_mut().schedule = sch);
    Ok(Value::Int(0))
}

/// Run the loaded program on opencl:d, or on the reference interpreter
/// for d = -1. The number of outputs.
fn run(args: &[Value]) -> Result<Value, OwnedError> {
    let d = whole(&args[0], "a device")?;
    SESSION.with(|s| {
        let mut s = s.borrow_mut();
        let checked = s
            .program
            .as_ref()
            .ok_or_else(|| OwnedError::invalid_argument("no program loaded (gp:l_oad first)"))?;
        let p = plan(checked, &s.schedule);
        let outputs = if d < 0 {
            interpret(checked, &s.inputs).map_err(failure)?
        } else {
            execute(checked, &p, &s.inputs, usize::try_from(d).unwrap_or(0)).map_err(failure)?
        };
        let n = outputs.len();
        s.outputs = outputs;
        s.last_plan = Some(p);
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
        bind: 2, "Num a => Int -> a -> Int", "Bind input n (from 1) to an array of its shape.";
        schedule: 1, "Char -> Int", "The schedule for later runs (a schedules/*.toml text; empty for the defaults).";
        run: 1, "Int -> Int", "Run on opencl:d (-1: the reference interpreter); the number of outputs.";
        output: 1, "Num a => Int -> a", "Output i (from 1) of the last run.";
        explain: 0, "Unit -> Char", "The last run's plan in words.";
        kernel: 0, "Unit -> Char", "The last run's OpenCL C.";
    }
}
