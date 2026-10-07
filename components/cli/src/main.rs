//! `xetal-gpu`: the X_eTaL-gpu tool.
//!
//! ```text
//! xetal-gpu check FILE.xir                    parse and check; every value with its type
//! xetal-gpu devices                           the OpenCL devices found
//! xetal-gpu run FILE.xir [--device cpu|opencl:N] [--bind NAME=1,2,3 | NAME=@FILE]...
//!                                             run; each output printed as xetal prints it
//! xetal-gpu print FILE.xir                    the program in its canonical text form
//! xetal-gpu kernel FILE.xir [SCHEDULE]        the OpenCL C it would run
//! xetal-gpu explain FILE.xir [SCHEDULE]       the plan: buffers, kernels, launches
//!   SCHEDULE: --float f32|f64  --int i32|i64  --work-group N
//! ```

use std::collections::HashMap;
use std::process::ExitCode;

use xetal_gpu_opencl::{explain, plan, Schedule, Width};
use xetal_gpu_runtime::{devices, execute};
use xetal_gpu_xir::{check, format, run, text, Array, Checked, Data, Error, Op, Program, Result, Scalar, Shape};

const USAGE: &str = "\
usage: xetal-gpu COMMAND FILE.xir [OPTIONS]
  check FILE.xir                 parse and check: every value with its type
  devices                        the OpenCL devices found (opencl:N)
  run FILE.xir [--device cpu|opencl:N] [SCHEDULE]   run; each output printed as xetal prints it
      [--bind NAME=1,2,3 | --bind NAME=@FILE]...   an input's items (a file: whitespace-separated)
  print FILE.xir                 the program in its canonical text form
  kernel FILE.xir [SCHEDULE]     the OpenCL C 1.2 source it would run
  explain FILE.xir [SCHEDULE]    the plan: buffers, kernels, launches, in words
    SCHEDULE: --float f32|f64 (f32)  --int i32|i64 (i64)  --work-group N (256)  --tile T (0)
  version";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("xetal-gpu: {e}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(args: &[String]) -> Result<()> {
    let Some(cmd) = args.first() else { return Err(Error(USAGE.into())) };
    match cmd.as_str() {
        "version" | "--version" | "-V" => {
            println!("xetal-gpu {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        "check" => {
            let checked = load(args.get(1))?;
            for (i, v) in checked.program.values.iter().enumerate() {
                println!("%{} : {} = {}", v.name, checked.types[i], describe(&checked.program, &v.op));
            }
            for o in &checked.program.outputs {
                println!("output %{}", checked.program.value(*o).name);
            }
            Ok(())
        }
        "print" => {
            let checked = load(args.get(1))?;
            print!("{}", text::print(&checked.program));
            Ok(())
        }
        "devices" => {
            let found = devices()?;
            if found.is_empty() {
                println!("no OpenCL device found (no OpenCL library, or no platform reports a device)");
            }
            for d in found {
                println!(
                    "opencl:{}  {} ({}, {}; {}), {} compute units, work-groups to {}, {} MB, {}",
                    d.index,
                    d.name,
                    d.kind,
                    d.vendor,
                    d.version,
                    d.compute_units,
                    d.max_work_group,
                    d.global_mem_bytes / (1024 * 1024),
                    if d.fp64 { "fp64" } else { "no fp64" }
                );
            }
            Ok(())
        }
        "kernel" | "explain" => {
            let checked = load(args.get(1))?;
            let schedule = schedule(&args[2.min(args.len())..])?;
            let p = plan(&checked, &schedule);
            if cmd == "kernel" {
                print!("{}", p.source);
            } else {
                print!("{}", explain(&checked, &p));
            }
            Ok(())
        }
        "run" => {
            let checked = load(args.get(1))?;
            let mut device = "cpu".to_string();
            let mut binds: Vec<String> = Vec::new();
            let mut rest: Vec<String> = Vec::new();
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--device" => {
                        device = args
                            .get(i + 1)
                            .cloned()
                            .ok_or_else(|| Error("--device needs a value (cpu, opencl:N)".into()))?;
                        i += 2;
                    }
                    "--bind" => {
                        binds.push(args.get(i + 1).cloned().ok_or_else(|| Error("--bind needs NAME=items".into()))?);
                        i += 2;
                    }
                    _ => {
                        rest.push(args[i].clone());
                        i += 1;
                    }
                }
            }
            let schedule = schedule(&rest)?;
            let inputs = bindings(&checked, &binds)?;
            let outputs = if device == "cpu" {
                run(&checked, &inputs)?
            } else if let Some(n) = device.strip_prefix("opencl:") {
                let index: usize = n
                    .parse()
                    .map_err(|_| Error(format!("--device opencl:N takes a number, got `{device}`")))?;
                execute(&checked, &plan(&checked, &schedule), &inputs, index)?
            } else {
                return Err(Error(format!(
                    "unknown device `{device}` (cpu, or opencl:N from `xetal-gpu devices`)"
                )));
            };
            for out in outputs {
                println!("{}", format::show(&out));
            }
            Ok(())
        }
        other => Err(Error(format!("unknown command `{other}`\n{USAGE}"))),
    }
}

/// `--float f32|f64 --int i32|i64 --work-group N --tile T`, the defaults otherwise.
fn schedule(args: &[String]) -> Result<Schedule> {
    let mut s = Schedule::default();
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1).ok_or_else(|| Error(format!("{} needs a value", args[i])))?;
        match args[i].as_str() {
            "--float" => {
                s.float = match value.as_str() {
                    "f32" => Width::W32,
                    "f64" => Width::W64,
                    other => return Err(Error(format!("--float takes f32 or f64, got `{other}`"))),
                }
            }
            "--int" => {
                s.int = match value.as_str() {
                    "i32" => Width::W32,
                    "i64" => Width::W64,
                    other => return Err(Error(format!("--int takes i32 or i64, got `{other}`"))),
                }
            }
            "--work-group" => {
                s.work_group = value
                    .parse()
                    .ok()
                    .filter(|n: &usize| n.is_power_of_two())
                    .ok_or_else(|| Error(format!("--work-group takes a power of two, got `{value}`")))?;
            }
            "--tile" => {
                s.tile = value
                    .parse()
                    .ok()
                    .filter(|t: &usize| *t == 0 || t.is_power_of_two())
                    .ok_or_else(|| Error(format!("--tile takes 0 (no tiling) or a power of two, got `{value}`")))?;
            }
            other => return Err(Error(format!("unknown option `{other}`\n{USAGE}"))),
        }
        i += 2;
    }
    Ok(s)
}

fn load(path: Option<&String>) -> Result<Checked> {
    let Some(path) = path else { return Err(Error(USAGE.into())) };
    let src = std::fs::read_to_string(path).map_err(|e| Error(format!("{path}: {e}")))?;
    let program = text::parse(&src).map_err(|e| Error(format!("{path}:{e}")))?;
    check(program).map_err(|e| Error(format!("{path}: {e}")))
}

fn describe(program: &Program, op: &Op) -> String {
    let line = text::print(&Program {
        values: vec![],
        outputs: vec![],
    });
    debug_assert!(line.is_empty());
    let name = |id| format!("%{}", program.value(id).name);
    match op {
        Op::Input => "input".to_string(),
        Op::Const(c) => format!("const ({} item{})", c.len(), if c.len() == 1 { "" } else { "s" }),
        Op::Map { op, args } => format!("map {op} {}", args.iter().map(|a| name(*a)).collect::<Vec<_>>().join(" ")),
        Op::Select { cond, a, b } => format!("select {} {} {}", name(*cond), name(*a), name(*b)),
        Op::Reduce { op, arg, axis: None } => format!("reduce {op} {}", name(*arg)),
        Op::Reduce { op, arg, axis: Some(a) } => format!("reduce {op} axis={a} {}", name(*arg)),
        Op::Cast { op, arg } => format!("cast {} {}", op.to, name(*arg)),
        Op::Matmul { a, b } => format!("matmul {} {}", name(*a), name(*b)),
    }
}

/// `NAME=1,2,3` or `NAME=@FILE` for each input, typed as the program
/// declares it.
fn bindings(checked: &Checked, binds: &[String]) -> Result<HashMap<String, Array>> {
    let mut inputs = HashMap::new();
    for b in binds {
        let Some((name, items)) = b.split_once('=') else {
            return Err(Error(format!("--bind takes NAME=items, got `{b}`")));
        };
        let Some(id) = checked.program.find(name) else {
            return Err(Error(format!("--bind {name}: no such value")));
        };
        if !matches!(checked.program.value(id).op, Op::Input) {
            return Err(Error(format!("--bind {name}: %{name} is not an input")));
        }
        let ty = checked.ty(id);
        let text = match items.strip_prefix('@') {
            Some(path) => std::fs::read_to_string(path).map_err(|e| Error(format!("--bind {name}: {path}: {e}")))?,
            None => items.replace(',', " "),
        };
        let toks: Vec<&str> = text.split_whitespace().collect();
        let data = match ty.scalar {
            Scalar::I32 | Scalar::I64 => Data::Int(parse_all(name, &toks)?),
            Scalar::F32 | Scalar::F64 => Data::Float(parse_all(name, &toks)?),
            Scalar::Bool => Data::Bool(parse_all::<u8>(name, &toks)?.into_iter().map(|x| x != 0).collect()),
        };
        let array = Array::new(ty.scalar, Shape(ty.shape.0.clone()), data).map_err(|e| Error(format!("--bind {name}: {e}")))?;
        inputs.insert(name.to_string(), array);
    }
    Ok(inputs)
}

fn parse_all<T: std::str::FromStr>(name: &str, toks: &[&str]) -> Result<Vec<T>> {
    toks.iter()
        .map(|t| t.parse::<T>().map_err(|_| Error(format!("--bind {name}: not a number: `{t}`"))))
        .collect()
}
