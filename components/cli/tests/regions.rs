//! Device-side loops must retain their body once, use bound inputs, and
//! report the first mapped lane's first failure before unsafe work.
use std::{collections::HashMap, path::Path};
use xetal_gpu_opencl::{plan, Schedule};
use xetal_gpu_xir::{check, interp, text, Array, Data, Scalar, Shape, Ty};

fn devices() -> Vec<xetal_gpu_runtime::DeviceInfo> {
    let devices = xetal_gpu_runtime::devices().unwrap();
    if devices.is_empty() {
        eprintln!("skipped: structured region OpenCL checks (no device)");
    }
    devices
}

fn lower(src: &str, shape: &[usize], scalar: Scalar) -> xetal_gpu_xir::Checked {
    let ty = Ty {
        scalar,
        shape: Shape(shape.to_vec()),
    };
    let l = xetal_gpu_lower::lower_function_regions(src, &[("x".into(), ty)], Path::new("."), "", true).unwrap();
    assert!(l.reasons.is_empty());
    assert_eq!(text::parse(&text::print(&l.program)).unwrap(), l.program);
    check(l.program).unwrap()
}

#[test]
fn repeated_body_is_one_dispatch_and_does_not_grow_with_iteration_count() {
    let mut sizes = Vec::new();
    for count in [2, 44, 322] {
        let c = lower(
            &format!("{{ x -> {count} '{{ state -> state * 1.0001 + 0.01 }} p_ower x }}"),
            &[7],
            Scalar::F64,
        );
        let p = plan(&c, &Schedule::default());
        assert_eq!(p.launches.iter().filter(|l| l.kernel.starts_with("region_")).count(), 1);
        sizes.push(p.source.len());
        let inputs = HashMap::from([("x".into(), Array::floats(Shape(vec![7]), vec![0., 1., 2., 3., 4., 5., 6.]))]);
        let expected = interp::run(&c, &inputs).unwrap();
        for d in devices() {
            let (mut prepared, _) = xetal_gpu_runtime::Prepared::new(&c, &p, &inputs, d.index).unwrap();
            let (out, _) = prepared.run_timed().unwrap();
            let (Data::Float(a), Data::Float(b)) = (&out[0].data, &expected[0].data) else {
                panic!()
            };
            for (a, b) in a.iter().zip(b) {
                assert!((a - b).abs() < 1e-4 * (1. + b.abs()), "{a} != {b}");
            }
        }
    }
    assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() < 16, "{sizes:?}");
}

#[test]
fn mapped_nested_loop_preserves_failure_order_and_recovers_after_rebinding() {
    let c=lower("{ x -> '{ item -> 3 '{ state -> state = 0 ? @ p_anic< \"zero\"; state = 1 ? @ p_anic< \"one\"; state - 1 } p_ower item } e_ach x }", &[3],Scalar::I64);
    let p = plan(&c, &Schedule::default());
    assert_eq!(p.launches.len(), 1);
    assert_eq!(p.guard_checks.len(), 1, "one host read for the region");
    let good = Array::ints(Shape(vec![3]), vec![5, 6, 7]);
    let inputs = HashMap::from([("x".into(), good.clone())]);
    assert_eq!(interp::run(&c, &inputs).unwrap()[0].data, Data::Int(vec![2, 3, 4]));
    let mut prepared: Vec<_> = devices()
        .into_iter()
        .map(|d| xetal_gpu_runtime::Prepared::new(&c, &p, &inputs, d.index).unwrap().0)
        .collect();
    for values in [vec![2, 0, 8], vec![0, 1, 8], vec![5, 6, 7]] {
        let array = Array::ints(Shape(vec![3]), values);
        let bound = HashMap::from([("x".into(), array.clone())]);
        let expected = interp::run(&c, &bound);
        for device in &mut prepared {
            device.rebind("x", &array).unwrap();
            match &expected {
                Ok(out) => assert_eq!(device.run_timed().unwrap().0, *out),
                Err(e) => assert_eq!(device.run_timed().unwrap_err().0, e.0),
            }
        }
    }
}

#[test]
fn each_supports_a_different_result_type_and_global_array_capture() {
    let c = lower(
        "{ x -> table := 3 2 r_eshape 1.0 2.0 3.0 4.0 5.0 6.0; '{ item -> '+ r_/ item s_elect table } e_ach x }",
        &[3],
        Scalar::I64,
    );
    let p = plan(&c, &Schedule::default());
    let inputs = HashMap::from([("x".into(), Array::ints(Shape(vec![3]), vec![3, 1, 2]))]);
    let expected = interp::run(&c, &inputs).unwrap();
    assert_eq!(expected[0].data, Data::Float(vec![11., 3., 7.]));
    for d in devices() {
        let (mut prepared, _) = xetal_gpu_runtime::Prepared::new(&c, &p, &inputs, d.index).unwrap();
        assert_eq!(prepared.run_timed().unwrap().0, expected);
    }
}

#[test]
fn shared_raytrace_nearest_keeps_geometry_loop_on_device() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let libs = root.join("work/xetal-libraries/libs");
    let paths: Vec<_> = std::fs::read_dir(libs)
        .unwrap()
        .map(|p| p.unwrap().path().join("src"))
        .filter(|p| p.is_dir())
        .collect();
    std::env::set_var("XETAL_PATH", std::env::join_paths(paths).unwrap());
    let src = r#"{ triangles origins directions indices ->
        clip := 1e-6 100.0 1e-10 1e-5;
        '{ index ->
            o := 1 3 r_eshape index s_elect origins;
            d := 1 3 r_eshape index s_elect directions;
            f_irst (((clip rt:n_earest triangles)_ o)_ d)
        } e_ach indices
    }"#;
    let mut counts = Vec::new();
    let mut sources = Vec::new();
    for triangles in [2, 44, 322] {
        let params = [
            ("triangles", Scalar::F64, vec![triangles, 3, 3]),
            ("origins", Scalar::F64, vec![7, 3]),
            ("directions", Scalar::F64, vec![7, 3]),
            ("indices", Scalar::I64, vec![7]),
        ]
        .map(|(n, s, shape)| {
            (
                n.into(),
                Ty {
                    scalar: s,
                    shape: Shape(shape),
                },
            )
        });
        let l = xetal_gpu_lower::lower_function_regions(src, &params, Path::new("."), "\"rt:\" u_se< \"RayTrace\"", true).unwrap();
        let c = check(l.program).unwrap();
        let p = plan(&c, &Schedule::default());
        counts.push(p.launches.len());
        sources.push(p.source.len());
        std::fs::write(root.join(format!("work/ray-region-{triangles}.cl")), &p.source).unwrap();
        let face = [-1., -1., 0., 1., -1., 0., 1., 1., 0.];
        let inputs = HashMap::from([
            (
                "triangles".into(),
                Array::floats(Shape(vec![triangles, 3, 3]), face.repeat(triangles)),
            ),
            (
                "origins".into(),
                Array::floats(
                    Shape(vec![7, 3]),
                    vec![
                        0.5, 0., 3., -0.5, 0., 3., 3., 0., 3., 0.5, 0., -3., 0.5, 0., 3., 0.5, 0., 3., 0.5, 0., 3.,
                    ],
                ),
            ),
            ("directions".into(), Array::floats(Shape(vec![7, 3]), [0., 0., -1.].repeat(7))),
            ("indices".into(), Array::ints(Shape(vec![7]), vec![1, 2, 3, 4, 5, 6, 7])),
        ]);
        let expected = interp::run(&c, &inputs).unwrap();
        assert_eq!(expected[0].data, Data::Int(vec![1, 0, 0, 0, 1, 1, 1]));
        for d in devices() {
            let (mut device, setup) = xetal_gpu_runtime::Prepared::new(&c, &p, &inputs, d.index).unwrap();
            let (out, warm) = device.run_timed().unwrap();
            assert_eq!(out, expected);
            eprintln!(
                "RayTrace {triangles} triangles: {} launches, {} source bytes, {:.3} ms setup, {:.3} ms device",
                p.launches.len(),
                p.source.len(),
                setup.setup_ms,
                warm.kernels_ms
            );
            let zero = Array::floats(Shape(vec![7, 3]), vec![0.; 21]);
            device.rebind("directions", &zero).unwrap();
            assert_eq!(device.run_timed().unwrap_err().0, "RayTrace directions must be nonzero");
            device.rebind("directions", &inputs["directions"]).unwrap();
            assert_eq!(device.run_timed().unwrap().0, expected);
        }
    }
    assert_eq!(counts, vec![counts[0]; 3], "dispatches must not grow per triangle");
    assert!(sources.iter().max().unwrap() - sources.iter().min().unwrap() < 1024, "{sources:?}");
}

#[test]
fn shared_shading_region_matches_batch_cpu_math() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let paths: Vec<_> = std::fs::read_dir(root.join("work/xetal-libraries/libs"))
        .unwrap()
        .map(|p| p.unwrap().path().join("src"))
        .filter(|p| p.is_dir())
        .collect();
    std::env::set_var("XETAL_PATH", std::env::join_paths(paths).unwrap());
    let adapter = std::fs::read_to_string(root.join("examples/ray/region-shade.xtl")).unwrap();
    let uses = adapter;
    let src = r#"{ triangles origins directions indices ->
        clip := 1e-6 100.0 1e-10 1e-5;
        materials := 2 4 r_eshape 0.8 0.3 0.1 0.0 0.1 0.3 0.8 0.0;
        lights := 1 6 r_eshape 0.0 1.0 3.0 30.0 30.0 30.0;
        env := 2 3 r_eshape 0.05 0.05 0.05 0.01 0.02 0.04;
        ((((((((clip u:s_hadeChannels triangles)_ 1 2)_ materials)_ lights)_ env)_ origins)_ directions)_ indices)
    }"#;
    let params = [
        ("triangles", Scalar::F64, vec![2, 3, 3]),
        ("origins", Scalar::F64, vec![4, 3]),
        ("directions", Scalar::F64, vec![4, 3]),
        ("indices", Scalar::I64, vec![12]),
    ]
    .map(|(n, s, shape)| {
        (
            n.into(),
            Ty {
                scalar: s,
                shape: Shape(shape),
            },
        )
    });
    let l = xetal_gpu_lower::lower_function_regions(src, &params, Path::new("."), &uses, true).unwrap();
    let c = check(l.program).unwrap();
    let p = plan(&c, &Schedule::default());
    let mesh = vec![-1., -1., 0., 1., -1., 0., 1., 1., 0., -1., -1., 0., 1., 1., 0., -1., 1., 0.];
    let inputs = HashMap::from([
        ("triangles".into(), Array::floats(Shape(vec![2, 3, 3]), mesh)),
        (
            "origins".into(),
            Array::floats(Shape(vec![4, 3]), vec![-0.5, -0.25, 3., 0.5, 0.25, 3., 2., 0., 3., 0., 0., 3.]),
        ),
        ("directions".into(), Array::floats(Shape(vec![4, 3]), [0., 0., -1.].repeat(4))),
        ("indices".into(), Array::ints(Shape(vec![12]), (1..=12).collect())),
    ]);
    let reference = interp::run(&c, &inputs).unwrap();
    let batch = r#"{ triangles origins directions indices ->
        clip := 1e-6 100.0 1e-10 1e-5;
        materials := 2 4 r_eshape 0.8 0.3 0.1 0.0 0.1 0.3 0.8 0.0;
        lights := 1 6 r_eshape 0.0 1.0 3.0 30.0 30.0 30.0;
        env := 2 3 r_eshape 0.05 0.05 0.05 0.01 0.02 0.04;
        r_avel (((((((clip rt:s_hade triangles)_ 1 2)_ materials)_ lights)_ env)_ origins)_ directions)
    }"#;
    let baseline = check(
        xetal_gpu_lower::lower_function(batch, &params, Path::new("."), &uses)
            .unwrap()
            .program,
    )
    .unwrap();
    assert_eq!(reference, interp::run(&baseline, &inputs).unwrap());
    for d in devices() {
        let (mut device, setup) = xetal_gpu_runtime::Prepared::new(&c, &p, &inputs, d.index).unwrap();
        let (out, warm) = device.run_timed().unwrap();
        let (Data::Float(a), Data::Float(b)) = (&out[0].data, &reference[0].data) else {
            panic!()
        };
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() < 1e-5 * (1. + b.abs()), "{a} != {b}");
        }
        eprintln!(
            "Shading: {} launches, {} source bytes, {:.3} ms setup, {:.3} ms device",
            p.launches.len(),
            p.source.len(),
            setup.setup_ms,
            warm.kernels_ms
        );
    }
}

#[test]
fn zero_iterations_empty_maps_and_capture_rebinding() {
    let empty = lower(
        "{ x -> '{ item -> 3 '{ state -> state + 1 } p_ower item } e_ach x }",
        &[0],
        Scalar::I64,
    );
    let inputs = HashMap::from([("x".into(), Array::ints(Shape(vec![0]), vec![]))]);
    let p = plan(&empty, &Schedule::default());
    assert_eq!(interp::run(&empty, &inputs).unwrap()[0].data, Data::Int(vec![]));
    for d in devices() {
        assert_eq!(
            xetal_gpu_runtime::Prepared::new(&empty, &p, &inputs, d.index)
                .unwrap()
                .0
                .run_timed()
                .unwrap()
                .0[0]
                .data,
            Data::Int(vec![])
        );
    }
    let zero = lower("{ x -> 0 '{ state -> @ p_anic< \"never\" } p_ower x }", &[3], Scalar::I64);
    assert!(!zero
        .program
        .values
        .iter()
        .any(|v| matches!(v.op, xetal_gpu_xir::Op::Region { .. } | xetal_gpu_xir::Op::Guard { .. })));
    let ty = Ty {
        scalar: Scalar::I64,
        shape: Shape::scalar(),
    };
    let params = [
        (
            "x".into(),
            Ty {
                scalar: Scalar::I64,
                shape: Shape(vec![5]),
            },
        ),
        ("increment".into(), ty),
    ];
    let src = "{ x increment -> '{ item -> 4 '{ state -> state + increment } p_ower item } e_ach x }";
    let c = check(
        xetal_gpu_lower::lower_function_regions(src, &params, Path::new("."), "", true)
            .unwrap()
            .program,
    )
    .unwrap();
    let p = plan(&c, &Schedule::default());
    let mut bound = HashMap::from([
        ("x".into(), Array::ints(Shape(vec![5]), vec![1, 2, 3, 4, 5])),
        ("increment".into(), Array::ints(Shape::scalar(), vec![1])),
    ]);
    for d in devices() {
        let (mut prepared, _) = xetal_gpu_runtime::Prepared::new(&c, &p, &bound, d.index).unwrap();
        for increment in [1, 7, -2] {
            let a = Array::ints(Shape::scalar(), vec![increment]);
            bound.insert("increment".into(), a.clone());
            prepared.rebind("increment", &a).unwrap();
            assert_eq!(prepared.run_timed().unwrap().0, interp::run(&c, &bound).unwrap());
        }
    }
}

#[test]
fn local_integer_division_checks_before_unsafe_operation() {
    let c = lower("{ x -> '{ item -> '+ r_/ (1 2) d_iv item } e_ach x }", &[2], Scalar::I64);
    let p = plan(&c, &Schedule::default());
    let bad = HashMap::from([("x".into(), Array::ints(Shape(vec![2]), vec![1, 0]))]);
    let error = interp::run(&c, &bad).unwrap_err().0;
    for d in devices() {
        let (mut prepared, _) = xetal_gpu_runtime::Prepared::new(&c, &p, &bad, d.index).unwrap();
        assert_eq!(prepared.run_timed().unwrap_err().0, error);
        let good = Array::ints(Shape(vec![2]), vec![1, 2]);
        prepared.rebind("x", &good).unwrap();
        assert_eq!(
            prepared.run_timed().unwrap().0,
            interp::run(&c, &HashMap::from([("x".into(), good)])).unwrap()
        );
    }
}

#[test]
fn private_float64_region_values_enable_fp64_even_with_integer_buffers() {
    let c = lower("{ x -> '{ item -> (0.25 + f_loat item) > 0.0 } e_ach x }", &[3], Scalar::I64);
    let schedule = Schedule {
        float: xetal_gpu_opencl::Width::W64,
        ..Schedule::default()
    };
    let p = plan(&c, &schedule);
    assert!(p.buffers.iter().all(|b| b.elem != xetal_gpu_opencl::Elem::F64));
    assert!(p.source.contains("#pragma OPENCL EXTENSION cl_khr_fp64 : enable"));
    assert!(p.needs_fp64());
    let inputs = HashMap::from([("x".into(), Array::ints(Shape(vec![3]), vec![-1, 0, 1]))]);
    for d in devices().into_iter().filter(|d| d.fp64) {
        let (mut device, _) = xetal_gpu_runtime::Prepared::new(&c, &p, &inputs, d.index).unwrap();
        assert_eq!(device.run_timed().unwrap().0, interp::run(&c, &inputs).unwrap());
    }
}
