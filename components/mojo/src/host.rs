//! The host side: find `mojo`, build the generated source as a shared
//! library (cached by the source's hash under `work/mojo-cache/`), load
//! it with `dlopen`, and call its C exports.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use xetal_gpu_xir::{Array, Checked, Data, Scalar, Shape};

use crate::{Abi, Error, Result};

/// The repository root above the running tool (the directory holding
/// `XETAL_COMMIT`), if any.
fn repo_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let mut d = exe.parent();
    while let Some(p) = d {
        if p.join("XETAL_COMMIT").exists() {
            return Some(p.to_path_buf());
        }
        d = p.parent();
    }
    None
}

/// The Mojo compiler: `XETAL_GPU_MOJO`, else the repository's `bin/mojo`
/// (`scripts/mojo.sh`), else `mojo` on the path. None when none is there.
pub fn mojo_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("XETAL_GPU_MOJO") {
        return Some(PathBuf::from(p));
    }
    if let Some(root) = repo_root() {
        let p = root.join("bin/mojo");
        if p.exists() {
            return Some(p);
        }
    }
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths).map(|d| d.join("mojo")).find(|p| p.is_file())
}

pub fn lib_ext() -> &'static str {
    if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    }
}

/// Where built libraries are kept: `XETAL_GPU_MOJO_CACHE`, else the
/// repository's `work/mojo-cache`, else a directory under the system's.
fn cache_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("XETAL_GPU_MOJO_CACHE") {
        return PathBuf::from(d);
    }
    match repo_root() {
        Some(root) => root.join("work/mojo-cache"),
        None => std::env::temp_dir().join("xetal-gpu-mojo-cache"),
    }
}

/// `mojo build SRC --emit shared-lib -o OUT` for the source, unless the
/// cache holds a library built from the same source by the same
/// compiler (its path and modification time). The library's path and
/// the build's time (zero on a cache hit).
pub fn build(mojo: &Path, source: &str, name: &str) -> Result<(PathBuf, Duration)> {
    let mut h = DefaultHasher::new();
    source.hash(&mut h);
    mojo.hash(&mut h);
    if let Ok(m) = std::fs::metadata(mojo).and_then(|m| m.modified()) {
        m.hash(&mut h);
    }
    let dir = cache_dir().join(format!("{:016x}", h.finish()));
    let lib = dir.join(format!("lib{name}.{}", lib_ext()));
    if lib.exists() {
        return Ok((lib, Duration::ZERO));
    }
    std::fs::create_dir_all(&dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    let src = dir.join(format!("{name}.mojo"));
    std::fs::write(&src, source).map_err(|e| Error(format!("{}: {e}", src.display())))?;
    let start = Instant::now();
    let output = Command::new(mojo)
        .arg("build")
        .arg(&src)
        .args(["--emit", "shared-lib", "-o"])
        .arg(&lib)
        .output()
        .map_err(|e| {
            Error(format!(
                "could not run {}: {e} (scripts/mojo.sh installs it, or set XETAL_GPU_MOJO)",
                mojo.display()
            ))
        })?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let noise: Vec<&str> = stderr.lines().filter(|l| !l.contains("Crashpad") && !l.trim().is_empty()).collect();
    if !output.status.success() {
        return Err(Error(format!("mojo build failed on {}:\n{}", src.display(), noise.join("\n"))));
    }
    Ok((lib, start.elapsed()))
}

type InitFn = unsafe extern "C" fn() -> i32;
type InfoFn = unsafe extern "C" fn(*mut u8, i64) -> i64;
type RunFn = unsafe extern "C" fn(*const *const c_void, *const *mut c_void) -> i32;

/// A built program, loaded.
pub struct Library {
    lib: libloading::Library,
}

impl Library {
    pub fn load(path: &Path) -> Result<Library> {
        let lib = unsafe { libloading::Library::new(path) }.map_err(|e| Error(format!("dlopen {}: {e}", path.display())))?;
        Ok(Library { lib })
    }

    fn sym<T>(&self, name: &[u8]) -> Result<libloading::Symbol<'_, T>> {
        unsafe { self.lib.get(name) }.map_err(|e| Error(format!("symbol {}: {e}", String::from_utf8_lossy(name))))
    }

    /// 1 when the build targets a GPU, 0 otherwise.
    pub fn init(&self) -> Result<i32> {
        let f: libloading::Symbol<InitFn> = self.sym(b"xir_init")?;
        Ok(unsafe { f() })
    }

    /// `api=...;name=...` of the device the kernels run on.
    pub fn device_info(&self) -> Result<String> {
        let f: libloading::Symbol<InfoFn> = self.sym(b"xir_device_info")?;
        let mut buf = vec![0u8; 512];
        let n = unsafe { f(buf.as_mut_ptr(), buf.len() as i64) };
        if n < 0 {
            return Err(Error("the library could not open a device (DeviceContext failed)".into()));
        }
        Ok(String::from_utf8_lossy(&buf[..n as usize]).into_owned())
    }

    /// Run the program on the arrays the ABI lists (the bound inputs by
    /// name, the constants from the program), giving the outputs as
    /// arrays of the checked types. Host arrays cross as f64, i64 or
    /// i32 Bool items, preserving Ints beyond 2^53. The time is the call's.
    pub fn run(&self, checked: &Checked, abi: &Abi, inputs: &HashMap<String, Array>) -> Result<(Vec<Array>, Duration)> {
        check_positions(checked, inputs)?;
        let mut ins: Vec<HostArray> = Vec::new();
        for (id, name) in &abi.inputs {
            let items = match name {
                Some(n) => {
                    let a = inputs.get(n).ok_or_else(|| Error(format!("input %{n} is not bound")))?;
                    let ty = checked.ty(*id);
                    if a.ty() != *ty {
                        return Err(Error(format!("input %{n} is {ty}, bound to {}", a.ty())));
                    }
                    HostArray::from_data(&a.data)
                }
                None => match &checked.program.values[id.0].op {
                    xetal_gpu_xir::Op::Const(c) => match c {
                        xetal_gpu_xir::Const::Int(v) => HostArray::Int(v.clone()),
                        xetal_gpu_xir::Const::Float(v) => HostArray::Float(v.clone()),
                        xetal_gpu_xir::Const::Bool(v) => HostArray::Bool(v.iter().map(|&b| i32::from(b)).collect()),
                    },
                    _ => unreachable!("the ABI's unnamed arrays are constants"),
                },
            };
            ins.push(items);
        }
        let mut outs: Vec<HostArray> = abi
            .outputs
            .iter()
            .map(|o| {
                let ty = checked.ty(*o);
                let n = ty.shape.len().max(1);
                match ty.scalar {
                    Scalar::F32 | Scalar::F64 => HostArray::Float(vec![0.0; n]),
                    Scalar::I32 | Scalar::I64 => HostArray::Int(vec![0; n]),
                    Scalar::Bool => HostArray::Bool(vec![0; n]),
                }
            })
            .collect();
        let f: libloading::Symbol<RunFn> = self.sym(b"xir_run")?;
        let in_ptrs: Vec<*const c_void> = ins.iter().map(HostArray::ptr).collect();
        let out_ptrs: Vec<*mut c_void> = outs.iter_mut().map(HostArray::mut_ptr).collect();
        let start = Instant::now();
        let rc = unsafe { f(in_ptrs.as_ptr(), out_ptrs.as_ptr()) };
        let took = start.elapsed();
        if rc != 0 {
            return Err(Error(format!("xir_run returned {rc} (the device failed; its message is above)")));
        }
        let mut arrays = Vec::new();
        for (o, items) in abi.outputs.iter().zip(outs) {
            let ty = checked.ty(*o);
            let n = ty.shape.len();
            let data = match items {
                HostArray::Float(v) => Data::Float(v[..n].to_vec()),
                HostArray::Int(v) => Data::Int(v[..n].to_vec()),
                HostArray::Bool(v) => Data::Bool(v[..n].iter().map(|&x| x != 0).collect()),
            };
            arrays.push(Array::new(ty.scalar, Shape(ty.shape.0.clone()), data).map_err(|e| Error(format!("an output: {e}")))?);
        }
        Ok((arrays, took))
    }
}

// The allocation's element type matches the generated library's
// host pointer casts; no numeric conversion happens at the C ABI.
enum HostArray {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Bool(Vec<i32>),
}

impl HostArray {
    fn from_data(data: &Data) -> Self {
        match data {
            Data::Int(v) => Self::Int(v.clone()),
            Data::Float(v) => Self::Float(v.clone()),
            Data::Bool(v) => Self::Bool(v.iter().map(|&b| i32::from(b)).collect()),
        }
    }

    fn ptr(&self) -> *const c_void {
        match self {
            Self::Int(v) => v.as_ptr().cast(),
            Self::Float(v) => v.as_ptr().cast(),
            Self::Bool(v) => v.as_ptr().cast(),
        }
    }

    fn mut_ptr(&mut self) -> *mut c_void {
        match self {
            Self::Int(v) => v.as_mut_ptr().cast(),
            Self::Float(v) => v.as_mut_ptr().cast(),
            Self::Bool(v) => v.as_mut_ptr().cast(),
        }
    }
}

/// Emit, build, load and run a checked program in one go: the outputs,
/// the device's description, the build's and the run's times.
pub fn execute(
    checked: &Checked,
    schedule: &xetal_gpu_opencl::Schedule,
    name: &str,
    inputs: &HashMap<String, Array>,
) -> Result<(Vec<Array>, String, Duration, Duration)> {
    let mojo = mojo_path()
        .ok_or_else(|| Error("no Mojo compiler: scripts/mojo.sh installs the pinned one as bin/mojo (or set XETAL_GPU_MOJO)".into()))?;
    let (source, abi) = crate::emit(checked, schedule, name)?;
    let (lib, built) = build(&mojo, &source, name)?;
    let library = Library::load(&lib)?;
    if library.init()? == 0 {
        return Err(Error("the Mojo build targets no GPU (has_accelerator is false)".into()));
    }
    let info = library.device_info()?;
    let (outputs, ran) = library.run(checked, &abi, inputs)?;
    Ok((outputs, info, built, ran))
}

/// A lookup's positions, when they are a bound input, must be within
/// 1..len of the axis they pick along: refused here with the
/// evaluator's message, since the device clamps a stray one (it cannot
/// refuse) and would give a wrong answer silently. Positions written in
/// the program were checked by the checker; computed ones cannot be
/// checked before they exist (ask G13 asks X_eTaL to type them).
fn check_positions(checked: &Checked, inputs: &HashMap<String, Array>) -> Result<()> {
    let program = &checked.program;
    for v in &program.values {
        let xetal_gpu_xir::Op::Lookup { axis, idx, arg } = &v.op else {
            continue;
        };
        let source = &program.values[idx.0];
        if !matches!(source.op, xetal_gpu_xir::Op::Input) {
            continue;
        }
        let Some(bound) = inputs.get(&source.name) else { continue };
        let len = checked.ty(*arg).shape.0[axis - 1] as i64;
        if let Data::Int(items) = &bound.data {
            if let Some(bad) = items.iter().find(|p| **p < 1 || **p > len) {
                return Err(Error(format!("%{}: lookup: index {bad} is outside 1..{len}", v.name)));
            }
        }
    }
    Ok(())
}
