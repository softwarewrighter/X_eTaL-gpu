//! Runtime validation stays ordered, including unused results and repeated calls.
use std::collections::HashMap;
use std::path::Path;
use xetal_gpu_opencl::{plan, Schedule};
use xetal_gpu_xir::{check, interp, text, Array, Data, Op, Scalar, Shape, Ty};

#[test]
fn lowered_guards_stop_before_division_and_recover_after_rebinding() {
    let source = "{ x -> divisor := f_irst x; divisor = 0 ? @ p_anic< \"zero divisor\"; 12 d_iv divisor }";
    let lowered = xetal_gpu_lower::lower_function(
        source,
        &[(
            "x".into(),
            Ty {
                scalar: Scalar::I64,
                shape: Shape(vec![3]),
            },
        )],
        Path::new("."),
        "",
    )
    .unwrap();
    assert!(lowered.reasons.is_empty());
    assert!(lowered.program.values.iter().any(|v| matches!(v.op, Op::Guard { .. })));
    let checked = check(lowered.program).unwrap();
    let schedule = Schedule::default();
    let p = plan(&checked, &schedule);
    assert_eq!(p.guard_checks.len(), 1);
    assert!(p.guard_checks[0].before_launch > 0 && p.guard_checks[0].before_launch < p.launches.len());
    let inputs = |x| HashMap::from([("x".into(), Array::ints(Shape(vec![3]), vec![x, 99, 99]))]);
    let mut opencl = Vec::new();
    for device in xetal_gpu_runtime::devices().unwrap() {
        opencl.push(xetal_gpu_runtime::Prepared::new(&checked, &p, &inputs(4), device.index).unwrap().0);
    }
    if opencl.is_empty() {
        eprintln!("skipped: guard execution on OpenCL (no device)");
    }
    let mojo = if xetal_gpu_mojo::host::mojo_path().is_some() {
        Some(xetal_gpu_mojo::host::Prepared::new(&checked, &schedule, "ordered_guards").unwrap())
    } else {
        eprintln!("skipped: guard execution on Mojo (no toolchain)");
        None
    };
    for x in [4, 0, 3, 0, -2] {
        let bound = inputs(x);
        let expected = interp::run(&checked, &bound);
        if x == 0 {
            assert_eq!(expected.as_ref().unwrap_err().0, "zero divisor");
        } else {
            assert_eq!(expected.as_ref().unwrap()[0].data, Data::Int(vec![12 / x]));
        }
        for device in &mut opencl {
            device.rebind("x", &bound["x"]).unwrap();
            assert_eq!(device.run_timed().map(|v| v.0), expected);
        }
        if let Some(device) = &mojo {
            assert_eq!(
                device.run_timed(&bound).map(|v| v.0).map_err(|e| xetal_gpu_xir::Error(e.0)),
                expected
            );
        }
    }
}

#[test]
fn constant_unused_guard_and_messages_survive_text_round_trip() {
    let message = "first # guard\n\"quoted\"";
    let source = format!(
        "%bad = const bool [] true\n%g = guard %bad {}\n%later = guard %bad {}\n%result = const i64 [] 7\noutput %result\n",
        text::encode_message(message),
        text::encode_message("second")
    );
    let program = text::parse(&source).unwrap();
    assert_eq!(text::parse(&text::print(&program)).unwrap(), program);
    let checked = check(program).unwrap();
    assert_eq!(interp::run(&checked, &HashMap::new()).unwrap_err().0, message);
    let schedule = Schedule::default();
    let p = plan(&checked, &schedule);
    assert_eq!(p.guard_checks[0].before_launch, 0);
    for device in xetal_gpu_runtime::devices().unwrap() {
        assert_eq!(
            xetal_gpu_runtime::execute(&checked, &p, &HashMap::new(), device.index)
                .unwrap_err()
                .0,
            message
        );
    }
    if xetal_gpu_mojo::host::mojo_path().is_some() {
        let device = xetal_gpu_mojo::host::Prepared::new(&checked, &schedule, "constant_guard").unwrap();
        assert_eq!(device.run_timed(&HashMap::new()).unwrap_err().0, message);
    }
    for token in ["x0", "xgg", "xff", "plain"] {
        assert!(text::parse(&format!("%bad = const bool [] true\n%g = guard %bad {token}\noutput %g\n")).is_err());
    }
    for input in ["i64 [] 1", "bool [1] true"] {
        let source = format!("%bad = const {input}\n%g = guard %bad x\noutput %g\n");
        assert!(check(text::parse(&source).unwrap()).unwrap_err().0.contains("scalar Bool"));
    }
}

#[test]
fn numeric_boolean_values_are_checked_on_the_device() {
    let schedule = Schedule::default();
    let devices = xetal_gpu_runtime::devices().unwrap();
    for scalar in [Scalar::I64, Scalar::F64] {
        let lowered = xetal_gpu_lower::lower_function(
            "{ x -> n_ot '& r_/ x }",
            &[(
                "x".into(),
                Ty {
                    scalar,
                    shape: Shape(vec![4]),
                },
            )],
            Path::new("."),
            "",
        )
        .unwrap();
        let checked = check(lowered.program).unwrap();
        let p = plan(&checked, &schedule);
        let mojo = if xetal_gpu_mojo::host::mojo_path().is_some() {
            Some(xetal_gpu_mojo::host::Prepared::new(&checked, &schedule, "numeric_boolean_guard").unwrap())
        } else {
            None
        };
        for n in [0, 1, 2, 1] {
            let array = if scalar == Scalar::I64 {
                Array::ints(Shape(vec![4]), vec![n; 4])
            } else {
                Array::floats(Shape(vec![4]), vec![n as f64; 4])
            };
            let inputs = HashMap::from([("x".into(), array)]);
            let expected = interp::run(&checked, &inputs);
            if n == 2 {
                assert_eq!(expected.as_ref().unwrap_err().0, "expected a Bool (or 1 / 0)");
            } else {
                assert_eq!(expected.as_ref().unwrap()[0].data, Data::Bool(vec![n == 0]));
            }
            for device in &devices {
                assert_eq!(xetal_gpu_runtime::execute(&checked, &p, &inputs, device.index), expected);
            }
            if let Some(device) = &mojo {
                assert_eq!(
                    device.run_timed(&inputs).map(|v| v.0).map_err(|e| xetal_gpu_xir::Error(e.0)),
                    expected
                );
            }
        }
    }
}
