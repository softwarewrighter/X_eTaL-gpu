//! The OpenCL host runtime: find the devices, and execute a plan on
//! one of them. X_eTaL arrays go to device buffers in the element
//! type the schedule chose, the kernels run in the plan's order on
//! an in-order queue, and the outputs come back as arrays of the
//! program's types (an f32 result widened to f64, a Bool from an
//! int).
//!
//! The OpenCL library is loaded at run time (opencl3's `dynamic`
//! feature): on a machine without one, [`devices`] reports none and
//! nothing else is lost.

use std::collections::HashMap;
use std::ptr;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{Device, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::{cl_long, CL_BLOCKING};

use xetal_gpu_opencl::{Arg, Elem, Init, Plan};
use xetal_gpu_xir::{Array, Checked, Const, Data, Error, Result, Scalar, Shape};

/// One OpenCL device, as `xetal-gpu devices` lists it; `index` is
/// the N of `--device opencl:N`.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceInfo {
    pub index: usize,
    pub platform: String,
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub kind: String,
    pub compute_units: u32,
    pub max_work_group: usize,
    pub global_mem_bytes: u64,
    pub fp64: bool,
}

/// Every device of every platform, in a fixed order; empty (not an
/// error) when there is no OpenCL library or no platform.
pub fn devices() -> Result<Vec<DeviceInfo>> {
    let Ok(platforms) = get_platforms() else { return Ok(Vec::new()) };
    let mut out = Vec::new();
    for p in platforms {
        let platform = p.name().unwrap_or_default();
        let Ok(ids) = p.get_devices(CL_DEVICE_TYPE_ALL) else { continue };
        for id in ids {
            let d = Device::new(id);
            let kind = match d.dev_type().unwrap_or(0) {
                t if t & CL_DEVICE_TYPE_GPU != 0 => "GPU",
                t if t & CL_DEVICE_TYPE_CPU != 0 => "CPU",
                t if t & CL_DEVICE_TYPE_ACCELERATOR != 0 => "accelerator",
                _ => "other",
            };
            out.push(DeviceInfo {
                index: out.len(),
                platform: platform.clone(),
                name: d.name().unwrap_or_default().trim().to_string(),
                vendor: d.vendor().unwrap_or_default().trim().to_string(),
                version: d.version().unwrap_or_default().trim().to_string(),
                kind: kind.to_string(),
                compute_units: d.max_compute_units().unwrap_or(0),
                max_work_group: d.max_work_group_size().unwrap_or(0),
                global_mem_bytes: d.global_mem_size().unwrap_or(0),
                fp64: d.extensions().unwrap_or_default().contains("cl_khr_fp64") || d.double_fp_config().unwrap_or(0) != 0,
            });
        }
    }
    Ok(out)
}

fn device(index: usize) -> Result<(DeviceInfo, Device)> {
    let infos = devices()?;
    if infos.is_empty() {
        return Err(Error(
            "no OpenCL device found (no OpenCL library, or no platform reports a device)".into(),
        ));
    }
    let Some(info) = infos.get(index).cloned() else {
        return Err(Error(format!(
            "no device opencl:{index}; `xetal-gpu devices` lists {} (0 to {})",
            infos.len(),
            infos.len() - 1
        )));
    };
    // Find the id again by walking the platforms in the same order.
    let mut n = 0;
    for p in get_platforms().map_err(cl)? {
        for id in p.get_devices(CL_DEVICE_TYPE_ALL).map_err(cl)? {
            if n == index {
                return Ok((info, Device::new(id)));
            }
            n += 1;
        }
    }
    unreachable!("the device was listed")
}

fn cl(e: opencl3::error_codes::ClError) -> Error {
    Error(format!("OpenCL: {e}"))
}

/// A device buffer of the plan's element type.
enum DevBuf {
    I32(Buffer<i32>),
    I64(Buffer<i64>),
    F32(Buffer<f32>),
    F64(Buffer<f64>),
}

/// Execute a plan on device `index`, with the inputs bound by name;
/// the outputs in order, as arrays of the program's types.
pub fn execute(checked: &Checked, plan: &Plan, inputs: &HashMap<String, Array>, index: usize) -> Result<Vec<Array>> {
    let (info, dev) = device(index)?;
    if plan.needs_fp64() && !info.fp64 {
        return Err(Error(format!(
            "device opencl:{index} ({}) has no double precision; schedule Float as f32",
            info.name
        )));
    }
    if plan.schedule.work_group > info.max_work_group {
        return Err(Error(format!(
            "device opencl:{index} ({}) allows work-groups of at most {}; schedule --work-group {} or less",
            info.name,
            info.max_work_group,
            info.max_work_group.next_power_of_two() / 2
        )));
    }
    let context = Context::from_device(&dev).map_err(cl)?;
    #[allow(deprecated)]
    let queue = CommandQueue::create_default(&context, 0).map_err(cl)?;
    let program = Program::create_and_build_from_source(&context, &plan.source, "")
        .map_err(|log| Error(format!("the OpenCL compiler of {} refused the kernels:\n{}", info.name, log.trim())))?;
    let mut kernels: HashMap<String, Kernel> = HashMap::new();
    for k in &plan.kernels {
        kernels.insert(k.name.clone(), Kernel::create(&program, &k.name).map_err(cl)?);
    }
    // Buffers: allocate each, upload inputs and constants.
    let mut bufs: Vec<DevBuf> = Vec::with_capacity(plan.buffers.len());
    for b in &plan.buffers {
        let len = b.len.max(1);
        let host: Option<Array> = match &b.init {
            Init::Input(name) => {
                let id = b.value.expect("an input buffer holds a value");
                let ty = checked.ty(id);
                let Some(a) = inputs.get(name) else {
                    return Err(Error(format!("input %{name} ({ty}) is not bound")));
                };
                if a.ty() != *ty {
                    return Err(Error(format!("input %{name} is {ty}, bound to {}", a.ty())));
                }
                Some(a.clone())
            }
            Init::Const(c) => Some(const_array(c, checked.ty(b.value.expect("a const holds a value")).scalar, len)),
            Init::Device => None,
        };
        let buf = unsafe {
            match b.elem {
                Elem::I32 => {
                    let mut d = Buffer::<i32>::create(&context, CL_MEM_READ_WRITE, len, ptr::null_mut()).map_err(cl)?;
                    if let Some(a) = &host {
                        let v: Vec<i32> = to_i64(a).into_iter().map(|x| x as i32).collect();
                        queue.enqueue_write_buffer(&mut d, CL_BLOCKING, 0, &v, &[]).map_err(cl)?;
                    }
                    DevBuf::I32(d)
                }
                Elem::I64 => {
                    let mut d = Buffer::<i64>::create(&context, CL_MEM_READ_WRITE, len, ptr::null_mut()).map_err(cl)?;
                    if let Some(a) = &host {
                        queue.enqueue_write_buffer(&mut d, CL_BLOCKING, 0, &to_i64(a), &[]).map_err(cl)?;
                    }
                    DevBuf::I64(d)
                }
                Elem::F32 => {
                    let mut d = Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, len, ptr::null_mut()).map_err(cl)?;
                    if let Some(a) = &host {
                        let v: Vec<f32> = to_f64(a).into_iter().map(|x| x as f32).collect();
                        queue.enqueue_write_buffer(&mut d, CL_BLOCKING, 0, &v, &[]).map_err(cl)?;
                    }
                    DevBuf::F32(d)
                }
                Elem::F64 => {
                    let mut d = Buffer::<f64>::create(&context, CL_MEM_READ_WRITE, len, ptr::null_mut()).map_err(cl)?;
                    if let Some(a) = &host {
                        queue.enqueue_write_buffer(&mut d, CL_BLOCKING, 0, &to_f64(a), &[]).map_err(cl)?;
                    }
                    DevBuf::F64(d)
                }
            }
        };
        bufs.push(buf);
    }
    // Launches, in order, on the in-order queue.
    for l in &plan.launches {
        let kernel = &kernels[&l.kernel];
        let max = kernel.get_work_group_size(dev.id()).map_err(cl)?;
        if l.local > max {
            return Err(Error(format!(
                "kernel {} allows work-groups of at most {max} on {}; schedule --work-group {max} or less",
                l.kernel, info.name
            )));
        }
        let mut ex = ExecuteKernel::new(kernel);
        unsafe {
            for a in &l.args {
                match a {
                    Arg::Buffer(id) => match &bufs[id.0] {
                        DevBuf::I32(b) => ex.set_arg(b),
                        DevBuf::I64(b) => ex.set_arg(b),
                        DevBuf::F32(b) => ex.set_arg(b),
                        DevBuf::F64(b) => ex.set_arg(b),
                    },
                    Arg::Long(n) => ex.set_arg(&(*n as cl_long)),
                    Arg::Local(bytes) => ex.set_arg_local_buffer(*bytes),
                };
            }
            ex.set_global_work_size(l.global)
                .set_local_work_size(l.local)
                .enqueue_nd_range(&queue)
                .map_err(cl)?;
        }
    }
    queue.finish().map_err(cl)?;
    // Read the outputs back as the program's types.
    let mut outputs = Vec::with_capacity(plan.outputs.len());
    for (value, buf) in &plan.outputs {
        let b = plan.buffer(*buf);
        let ty = checked.ty(*value);
        let data = unsafe {
            match &bufs[buf.0] {
                DevBuf::I32(d) => {
                    let mut v = vec![0i32; b.len.max(1)];
                    queue.enqueue_read_buffer(d, CL_BLOCKING, 0, &mut v, &[]).map_err(cl)?;
                    v.truncate(b.len);
                    from_ints(v.into_iter().map(i64::from).collect(), ty.scalar)
                }
                DevBuf::I64(d) => {
                    let mut v = vec![0i64; b.len.max(1)];
                    queue.enqueue_read_buffer(d, CL_BLOCKING, 0, &mut v, &[]).map_err(cl)?;
                    v.truncate(b.len);
                    from_ints(v, ty.scalar)
                }
                DevBuf::F32(d) => {
                    let mut v = vec![0f32; b.len.max(1)];
                    queue.enqueue_read_buffer(d, CL_BLOCKING, 0, &mut v, &[]).map_err(cl)?;
                    v.truncate(b.len);
                    Data::Float(v.into_iter().map(f64::from).collect())
                }
                DevBuf::F64(d) => {
                    let mut v = vec![0f64; b.len.max(1)];
                    queue.enqueue_read_buffer(d, CL_BLOCKING, 0, &mut v, &[]).map_err(cl)?;
                    v.truncate(b.len);
                    Data::Float(v)
                }
            }
        };
        outputs.push(Array::new(ty.scalar, Shape(ty.shape.0.clone()), data)?);
    }
    Ok(outputs)
}

fn const_array(c: &Const, scalar: Scalar, len: usize) -> Array {
    let fill = |n: usize| if n == len { None } else { Some(len) };
    let data = match c {
        Const::Int(v) => Data::Int(match fill(v.len()) {
            Some(n) => vec![v[0]; n],
            None => v.clone(),
        }),
        Const::Float(v) => Data::Float(match fill(v.len()) {
            Some(n) => vec![v[0]; n],
            None => v.clone(),
        }),
        Const::Bool(v) => Data::Bool(match fill(v.len()) {
            Some(n) => vec![v[0]; n],
            None => v.clone(),
        }),
    };
    Array::new(scalar, Shape(vec![len]), data).expect("a checked constant")
}

fn to_i64(a: &Array) -> Vec<i64> {
    match &a.data {
        Data::Int(v) => v.clone(),
        Data::Bool(v) => v.iter().map(|x| i64::from(*x)).collect(),
        Data::Float(v) => v.iter().map(|x| *x as i64).collect(),
    }
}

fn to_f64(a: &Array) -> Vec<f64> {
    match &a.data {
        Data::Float(v) => v.clone(),
        Data::Int(v) => v.iter().map(|x| *x as f64).collect(),
        Data::Bool(v) => v.iter().map(|x| f64::from(u8::from(*x))).collect(),
    }
}

fn from_ints(v: Vec<i64>, scalar: Scalar) -> Data {
    if scalar == Scalar::Bool {
        Data::Bool(v.into_iter().map(|x| x != 0).collect())
    } else {
        Data::Int(v)
    }
}

#[cfg(test)]
mod tests {
    //! These run only when a device is present; `cargo test` on a
    //! machine without one prints that they were skipped.

    use super::*;
    use xetal_gpu_opencl::{plan, Schedule};
    use xetal_gpu_xir::{check, format, text};

    fn on_device(src: &str, schedule: &Schedule) -> Option<Vec<String>> {
        if devices().unwrap().is_empty() {
            eprintln!("skipped: no OpenCL device");
            return None;
        }
        let checked = check(text::parse(src).unwrap()).unwrap();
        let p = plan(&checked, schedule);
        Some(
            execute(&checked, &p, &HashMap::new(), 0)
                .unwrap()
                .iter()
                .map(format::show)
                .collect(),
        )
    }

    #[test]
    fn elementwise_ints_exactly() {
        let Some(out) = on_device(
            "%a = const i64 [4] 1 2 3 4\n%k = const i64 [] 10\n%b = map mul %k %a\n%c = map add %a %b\n%m = map mod %c %k\n%d = map idiv %c %k\n%n = map neg %c\n%g = map gt %c %k\noutput %c\noutput %m\noutput %d\noutput %n\noutput %g\n",
            &Schedule::default(),
        ) else {
            return;
        };
        assert_eq!(out, ["11 22 33 44", "1 2 3 4", "1 2 3 4", "-11 -22 -33 -44", "1 1 1 1"]);
    }

    #[test]
    fn floats_within_f32() {
        let Some(out) = on_device(
            "%x = const f64 [4] -0.75 0.0 0.5 1.0\n%t = const f64 [] 0.25\n%m = map gt %x %t\n%z = const f64 [] 0.0\n%s = select %m %x %z\n%q = map div %x %t\n%r = reduce add %x\noutput %s\noutput %q\noutput %r\n",
            &Schedule::default(),
        ) else {
            return;
        };
        assert_eq!(out, ["0.0 0.0 0.5 1.0", "-3.0 0.0 2.0 4.0", "0.75"]);
    }

    #[test]
    fn a_reduction_over_many_work_groups() {
        let n = 70_000;
        let src = format!("%a = const i64 [{n}] 3\n%r = reduce add %a\n%m = reduce max %a\noutput %r\noutput %m\n");
        let Some(out) = on_device(&src, &Schedule::default()) else { return };
        assert_eq!(out, [(3 * n).to_string(), "3".to_string()]);
    }

    #[test]
    fn inputs_bound_by_name() {
        if devices().unwrap().is_empty() {
            eprintln!("skipped: no OpenCL device");
            return;
        }
        let checked = check(text::parse("%a = input i64 [3]\n%k = const i64 [] 2\n%b = map mul %k %a\noutput %b\n").unwrap()).unwrap();
        let p = plan(&checked, &Schedule::default());
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), Array::ints(Shape(vec![3]), vec![1, 2, 3]));
        let out = execute(&checked, &p, &inputs, 0).unwrap();
        assert_eq!(format::show(&out[0]), "2 4 6");
        assert_eq!(
            execute(&checked, &p, &HashMap::new(), 0).unwrap_err().0,
            "input %a (i64 [3]) is not bound"
        );
    }
}
