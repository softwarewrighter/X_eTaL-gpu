//! Validation reductions and whole-array matching depend on device inputs.
use std::collections::HashMap;
use std::path::Path;
use xetal_gpu_opencl::{plan, Schedule};
use xetal_gpu_xir::{check, interp, Array, Data, Scalar, Shape, Ty};

#[test]
fn validation_operators_on_installed_devices() {
    let ty = Ty {
        scalar: Scalar::I64,
        shape: Shape(vec![2, 3]),
    };
    let devices = xetal_gpu_runtime::devices().unwrap();
    if devices.is_empty() {
        eprintln!("skipped: validation operators on OpenCL (no device)");
    }
    let schedule = Schedule::default();
    let cases = [
        (
            "{ x y -> n_ot (2 3 m_atch s_hape x) ? @ p_anic< \"bad shape\"; '& r_/ x > 0 }",
            vec![vec![true, false, true], vec![true; 3], vec![false; 3]],
        ),
        ("{ x y -> '| r_/_2 x > 0 }", vec![vec![true; 2], vec![true; 2], vec![false; 2]]),
        ("{ x y -> x m_atch y }", vec![vec![false], vec![true], vec![false]]),
    ];
    for (source, expected) in cases {
        let lowered =
            xetal_gpu_lower::lower_function(source, &[("x".into(), ty.clone()), ("y".into(), ty.clone())], Path::new("."), "").unwrap();
        assert!(lowered.reasons.is_empty());
        let checked = check(lowered.program).unwrap();
        let mojo = if xetal_gpu_mojo::host::mojo_path().is_some() {
            Some(xetal_gpu_mojo::host::Prepared::new(&checked, &schedule, "validation_ops").unwrap())
        } else {
            eprintln!("skipped: validation operators on Mojo (no toolchain)");
            None
        };
        for (x, expected) in [vec![1, -1, 2, 3, 4, 5], vec![1; 6], vec![0; 6]].into_iter().zip(expected) {
            let inputs = HashMap::from([
                ("x".into(), Array::ints(Shape(vec![2, 3]), x)),
                ("y".into(), Array::ints(Shape(vec![2, 3]), vec![1; 6])),
            ]);
            let reference = interp::run(&checked, &inputs).unwrap();
            assert_eq!(reference[0].data, Data::Bool(expected));
            for device in &devices {
                assert_eq!(
                    xetal_gpu_runtime::execute(&checked, &plan(&checked, &schedule), &inputs, device.index).unwrap(),
                    reference
                );
            }
            if let Some(device) = &mojo {
                assert_eq!(device.run_timed(&inputs).unwrap().0, reference);
            }
        }
    }
}
