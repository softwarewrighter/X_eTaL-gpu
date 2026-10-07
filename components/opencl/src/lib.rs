//! The schedule and the OpenCL C 1.2 emitter.
//!
//! A checked XIR program plus a [`Schedule`] (how the device
//! represents each scalar type, the work-group size) becomes a
//! [`Plan`]: the kernel source as text, the device buffers (inputs,
//! constants, and the values the kernels write), and the launches in
//! order. Nothing here touches a device: the runtime component
//! executes a plan, and the tests pin the source for every twin.
//!
//! The schedule, not the IR, decides:
//! - **fusion**: consecutive elementwise values of one shape become
//!   one kernel, one work-item per element; only values an output or
//!   a later kernel needs are written to buffers, the rest stay in
//!   registers; a single-value constant is a literal in the source;
//! - **reductions**: a tree reduction in local memory, one partial
//!   per work-group, launched again on the partials until one value
//!   is left (so a reduce of a million items is three launches at a
//!   work-group of 256);
//! - **widths**: X_eTaL's Float (f64) computes as `float` unless the
//!   schedule asks for `double`; its Int (i64) as `long` unless `int`
//!   is asked; Bool is `int`.

pub mod emit;
pub mod explain;

use std::fmt;

use xetal_gpu_xir::{Const, Scalar, ValueId};

pub use emit::plan;
pub use explain::explain;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_files() {
        assert_eq!(Schedule::from_toml("").unwrap(), Schedule::default());
        let s = Schedule::from_toml("device = \"x\"\nfloat = \"f64\"\nint = \"i32\"\nwork_group = 1024\ntile = 32\n").unwrap();
        assert_eq!(
            s,
            Schedule {
                float: Width::W64,
                int: Width::W32,
                work_group: 1024,
                tile: 32
            }
        );
        assert_eq!(
            Schedule::from_toml("tile = 32\n").unwrap_err(),
            "tile = 32: a 32 by 32 tile is a work-group of 1024, more than work_group = 256"
        );
        assert_eq!(Schedule::from_toml("float = \"f16\"\n").unwrap_err(), "float = \"f16\": f32 or f64");
        assert_eq!(
            Schedule::from_toml("work_group = 100\n").unwrap_err(),
            "work_group = 100: a power of two (the reduction tree needs it)"
        );
        assert!(Schedule::from_toml("workgroup = 64\n")
            .unwrap_err()
            .contains("unknown field `workgroup`"));
    }

    #[test]
    fn every_schedule_file_reads() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schedules");
        let mut n = 0;
        for e in std::fs::read_dir(&dir).expect("schedules/").flatten() {
            if e.path().extension().is_some_and(|x| x == "toml") {
                let text = std::fs::read_to_string(e.path()).unwrap();
                Schedule::from_toml(&text).unwrap_or_else(|err| panic!("{}: {err}", e.path().display()));
                n += 1;
            }
        }
        assert!(n >= 2, "schedule files found: {n}");
        let apple = std::fs::read_to_string(dir.join("apple-m1-max.toml")).unwrap();
        assert_eq!(
            Schedule::from_toml(&apple).unwrap(),
            Schedule::default(),
            "the Apple file is the defaults"
        );
    }
}

/// How wide a number is on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    W32,
    W64,
}

/// What the device is asked to do with the program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    /// Float values as `float` (W32) or `double` (W64, needs cl_khr_fp64).
    pub float: Width,
    /// Int values as `int` (W32) or `long` (W64).
    pub int: Width,
    /// Work-items per work-group; a power of two (the reduction tree needs it).
    pub work_group: usize,
    /// Inner products in T by T tiles staged in local memory (a work-group
    /// of T * T items); 0 computes each result item straight from global
    /// memory. A power of two, T * T at most the work-group.
    pub tile: usize,
}

impl Default for Schedule {
    fn default() -> Self {
        Schedule {
            float: Width::W32,
            int: Width::W64,
            work_group: 256,
            tile: 0,
        }
    }
}

/// A schedule as a file (`schedules/<device>.toml`): every field
/// optional, the defaults otherwise; an unknown key is an error.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleFile {
    /// What the file is for (shown by `explain`; not used otherwise).
    #[allow(dead_code)]
    device: Option<String>,
    /// Whether the file has been run on its device (documentation).
    #[allow(dead_code)]
    tested: Option<bool>,
    float: Option<String>,
    int: Option<String>,
    work_group: Option<usize>,
    tile: Option<usize>,
}

impl Schedule {
    /// A schedule from the text of a TOML file: `float = "f32"`,
    /// `int = "i64"`, `work_group = 256`, `tile = 16` (each optional),
    /// and `device = "..."`, `tested = true` as documentation.
    pub fn from_toml(text: &str) -> Result<Schedule, String> {
        let f: ScheduleFile = toml::from_str(text).map_err(|e| e.to_string().trim().to_string())?;
        let mut s = Schedule::default();
        if let Some(v) = f.float {
            s.float = match v.as_str() {
                "f32" => Width::W32,
                "f64" => Width::W64,
                _ => return Err(format!("float = \"{v}\": f32 or f64")),
            };
        }
        if let Some(v) = f.int {
            s.int = match v.as_str() {
                "i32" => Width::W32,
                "i64" => Width::W64,
                _ => return Err(format!("int = \"{v}\": i32 or i64")),
            };
        }
        if let Some(v) = f.work_group {
            s.work_group = v;
        }
        if let Some(v) = f.tile {
            s.tile = v;
        }
        s.validate()?;
        Ok(s)
    }

    /// Whether the fields fit together, naming the one that does not.
    pub fn validate(&self) -> Result<(), String> {
        if self.work_group == 0 || !self.work_group.is_power_of_two() {
            return Err(format!(
                "work_group = {}: a power of two (the reduction tree needs it)",
                self.work_group
            ));
        }
        if self.tile != 0 && !self.tile.is_power_of_two() {
            return Err(format!("tile = {}: 0 (untiled) or a power of two", self.tile));
        }
        if self.tile * self.tile > self.work_group {
            return Err(format!(
                "tile = {}: a {0} by {0} tile is a work-group of {}, more than work_group = {}",
                self.tile,
                self.tile * self.tile,
                self.work_group
            ));
        }
        Ok(())
    }

    /// The device element type for a scalar type.
    pub fn elem(&self, s: Scalar) -> Elem {
        match s {
            Scalar::I32 => Elem::I32,
            Scalar::I64 => match self.int {
                Width::W32 => Elem::I32,
                Width::W64 => Elem::I64,
            },
            Scalar::F32 => Elem::F32,
            Scalar::F64 => match self.float {
                Width::W32 => Elem::F32,
                Width::W64 => Elem::F64,
            },
            Scalar::Bool => Elem::I32,
        }
    }
}

impl fmt::Display for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Float as {}, Int as {}, work-group {}, {}",
            match self.float {
                Width::W32 => "float",
                Width::W64 => "double",
            },
            match self.int {
                Width::W32 => "int",
                Width::W64 => "long",
            },
            self.work_group,
            if self.tile == 0 {
                "products untiled".to_string()
            } else {
                format!("products in {0} by {0} tiles", self.tile)
            }
        )
    }
}

/// A device element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Elem {
    I32,
    I64,
    F32,
    F64,
}

impl Elem {
    /// The OpenCL C type name.
    pub fn c(self) -> &'static str {
        match self {
            Elem::I32 => "int",
            Elem::I64 => "long",
            Elem::F32 => "float",
            Elem::F64 => "double",
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Elem::I32 | Elem::F32 => 4,
            Elem::I64 | Elem::F64 => 8,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Elem::F32 | Elem::F64)
    }
}

impl fmt::Display for Elem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.c())
    }
}

/// A buffer's position in the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub usize);

/// Where a buffer's items come from.
#[derive(Debug, Clone, PartialEq)]
pub enum Init {
    /// Bound by the host (the input's name).
    Input(String),
    /// Items written in the program, converted to the element type.
    Const(Const),
    /// Written by a kernel.
    Device,
}

/// A device buffer: `len` items of `elem`.
#[derive(Debug, Clone, PartialEq)]
pub struct Buffer {
    pub id: BufferId,
    /// The value it holds, when it holds one (partials have none).
    pub value: Option<ValueId>,
    /// The name in the kernel source (`b3`, `p0`).
    pub name: String,
    pub elem: Elem,
    pub len: usize,
    pub init: Init,
}

/// A kernel argument.
#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    Buffer(BufferId),
    /// A `long` by value.
    Long(i64),
    /// Local memory of this many bytes.
    Local(usize),
}

/// One kernel launch.
#[derive(Debug, Clone, PartialEq)]
pub struct Launch {
    pub kernel: String,
    pub args: Vec<Arg>,
    /// Work-items in all (a multiple of `local`).
    pub global: usize,
    /// Work-items per work-group.
    pub local: usize,
    /// What it is, for `explain`.
    pub note: String,
}

/// A kernel in the source, for `explain`.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelInfo {
    pub name: String,
    /// The values it computes, in order.
    pub computes: Vec<ValueId>,
    /// The values it writes to buffers.
    pub writes: Vec<ValueId>,
    /// Elements, one work-item each (0 for a kernel shared by several values).
    pub elements: usize,
    /// What a shared kernel does, for `explain` (empty for an elementwise one).
    pub about: String,
}

/// The whole plan: source, buffers, launches, outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub schedule: Schedule,
    pub source: String,
    pub buffers: Vec<Buffer>,
    pub kernels: Vec<KernelInfo>,
    pub launches: Vec<Launch>,
    /// Each output value and the buffer holding it.
    pub outputs: Vec<(ValueId, BufferId)>,
}

impl Plan {
    pub fn buffer(&self, id: BufferId) -> &Buffer {
        &self.buffers[id.0]
    }

    /// Whether the source needs double precision.
    pub fn needs_fp64(&self) -> bool {
        self.buffers.iter().any(|b| b.elem == Elem::F64) || self.source.contains("double")
    }
}
