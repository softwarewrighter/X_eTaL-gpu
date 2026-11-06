//! A strictly lowered mask expression agrees on CPU and installed GPUs.
use std::collections::HashMap;
use std::path::Path;
use xetal_gpu_opencl::{plan, Schedule};
use xetal_gpu_xir::{check, interp, Array, Scalar, Shape, Ty};

#[test]
fn lowered_boolean_masks_on_installed_devices() {
    let ty = Ty {
        scalar: Scalar::I64,
        shape: Shape(vec![4]),
    };
    let lowered = xetal_gpu_lower::lower_function(
        "{ x y -> (n_ot (x = 0)) & ((y > 0) | (x < 0)) }",
        &[("x".into(), ty.clone()), ("y".into(), ty)],
        Path::new("."),
        "",
    )
    .unwrap();
    assert!(lowered.reasons.is_empty());
    let checked = check(lowered.program).unwrap();
    let schedule = Schedule::default();
    let opencl = xetal_gpu_runtime::devices().unwrap();
    let mojo = if xetal_gpu_mojo::host::mojo_path().is_some() {
        Some(xetal_gpu_mojo::host::Prepared::new(&checked, &schedule, "boolean_masks").unwrap())
    } else {
        eprintln!("skipped: boolean masks on Mojo (no toolchain)");
        None
    };
    if opencl.is_empty() {
        eprintln!("skipped: boolean masks on OpenCL (no device)");
    }
    for x in [vec![-1, 0, 1, 2], vec![1, 2, -1, 0]] {
        let inputs = HashMap::from([
            ("x".into(), Array::ints(Shape(vec![4]), x)),
            ("y".into(), Array::ints(Shape(vec![4]), vec![0, 1, 0, 1])),
        ]);
        let expected = interp::run(&checked, &inputs).unwrap();
        for device in &opencl {
            assert_eq!(
                xetal_gpu_runtime::execute(&checked, &plan(&checked, &schedule), &inputs, device.index).unwrap(),
                expected
            );
        }
        if let Some(device) = &mojo {
            assert_eq!(device.run_timed(&inputs).unwrap().0, expected);
        }
    }
}
