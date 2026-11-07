//! Actual shared glass optics on bound inputs, including strict refusal paths.
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use xetal_gpu_opencl::{plan, Schedule, Width};
use xetal_gpu_xir::{check, interp, text, Array, Data, Scalar, Shape, Ty};
fn libraries() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let paths: Vec<_> = std::fs::read_dir(root.join("work/xetal-libraries/libs"))
        .unwrap()
        .map(|p| p.unwrap().path().join("src"))
        .filter(|p| p.is_dir())
        .collect();
    std::env::set_var("XETAL_PATH", std::env::join_paths(paths).unwrap());
    root
}
fn close(a: &[Array], b: &[Array]) {
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.ty(), b.ty());
        let (Data::Float(a), Data::Float(b)) = (&a.data, &b.data) else {
            panic!()
        };
        for (a, b) in a.iter().zip(b) {
            assert!(a.is_finite() && (*a - *b).abs() <= 1e-12 * (1. + b.abs()), "{a} != {b}");
        }
    }
}
#[test]
fn shared_interface_snell_fresnel_tir_absorption_and_validation() {
    libraries();
    let src = r#"{ ni nt normals directions sigma distances ->
 (reflected,transmitted,valid,fresnel) := (((ni gl:i_nterface nt)_ normals)_ directions);
 (r_avel reflected) c_at (r_avel transmitted) c_at (f_loat valid) c_at fresnel c_at r_avel sigma gl:a_bsorb distances
 }"#;
    let params = [
        ("ni", vec![3]),
        ("nt", vec![3]),
        ("normals", vec![3, 3]),
        ("directions", vec![3, 3]),
        ("sigma", vec![3, 3]),
        ("distances", vec![3]),
    ]
    .map(|(n, s)| {
        (
            n.into(),
            Ty {
                scalar: Scalar::F64,
                shape: Shape(s),
            },
        )
    });
    let l = xetal_gpu_lower::lower_function_regions(src, &params, Path::new("."), "\"gl:\" u_se< \"Glass\"", true).unwrap();
    let c = check(l.program).unwrap();
    assert_eq!(text::parse(&text::print(&c.program)).unwrap(), c.program);
    let f = |s: Vec<usize>, v| Array::floats(Shape(s), v);
    let bound = HashMap::from([
        ("ni".into(), f(vec![3], vec![1., 1., 1.5])),
        ("nt".into(), f(vec![3], vec![1.5, 1.5, 1.])),
        ("normals".into(), f(vec![3, 3], [0., 0., 1.].repeat(3))),
        (
            "directions".into(),
            f(
                vec![3, 3],
                vec![0., 0., -1., 0.5, 0., -0.8660254037844386, 0.8660254037844386, 0., -0.5],
            ),
        ),
        ("sigma".into(), f(vec![3, 3], [0., 0.5, 1.].repeat(3))),
        ("distances".into(), f(vec![3], vec![2.; 3])),
    ]);
    let expected = interp::run(&c, &bound).unwrap();
    let Data::Float(v) = &expected[0].data else { panic!() };
    assert_eq!(&v[18..21], &[1., 1., 0.]);
    assert!((v[21] - 0.04).abs() < 1e-14);
    assert_eq!(v[23], 1.);
    assert!((v[12] - 1. / 3.).abs() < 1e-14);
    assert!((v[25] - (-1_f64).exp()).abs() < 1e-14);
    let schedule = Schedule {
        float: Width::W64,
        ..Schedule::default()
    };
    let p = plan(&c, &schedule);
    let devices = xetal_gpu_runtime::devices().unwrap();
    if devices.is_empty() {
        eprintln!("skipped: glass OpenCL (no device)");
    }
    for d in devices.into_iter().filter(|d| d.fp64) {
        let (mut dev, _) = xetal_gpu_runtime::Prepared::new(&c, &p, &bound, d.index).unwrap();
        close(&dev.run_timed().unwrap().0, &expected);
        let bad = f(vec![3, 3], [0., 0., 2.].repeat(3));
        dev.rebind("normals", &bad).unwrap();
        assert_eq!(dev.run_timed().unwrap_err().0, "Glass interface vectors must be unit length");
        dev.rebind("normals", &bound["normals"]).unwrap();
        close(&dev.run_timed().unwrap().0, &expected);
    }
}
#[test]
fn lazy_tuple_branches_skip_unsafe_work_and_inactive_guards() {
    let src = r#"{ x ->
 branch_ := { item -> item = 0 ? (2, 3); item = 1 ? @ p_anic< "one"; (9 d_iv item, 4) };
 '{ item -> (a, b) := branch_ item; a + b } e_ach x
 }"#;
    let params = [(
        "x".into(),
        Ty {
            scalar: Scalar::I64,
            shape: Shape(vec![3]),
        },
    )];
    let c = check(
        xetal_gpu_lower::lower_function_regions(src, &params, Path::new("."), "", true)
            .unwrap()
            .program,
    )
    .unwrap();
    assert_eq!(text::parse(&text::print(&c.program)).unwrap(), c.program);
    let bound = HashMap::from([("x".into(), Array::ints(Shape(vec![3]), vec![0, 3, -3]))]);
    let expected = interp::run(&c, &bound).unwrap();
    assert_eq!(expected[0].data, Data::Int(vec![5, 7, 1]));
    for d in xetal_gpu_runtime::devices().unwrap() {
        let p = plan(&c, &Schedule::default());
        let (mut dev, _) = xetal_gpu_runtime::Prepared::new(&c, &p, &bound, d.index).unwrap();
        assert_eq!(dev.run_timed().unwrap().0, expected);
        dev.rebind("x", &Array::ints(Shape(vec![3]), vec![0, 1, 0])).unwrap();
        assert_eq!(dev.run_timed().unwrap_err().0, "one");
        dev.rebind("x", &bound["x"]).unwrap();
        assert_eq!(dev.run_timed().unwrap().0, expected);
    }
}
#[test]
fn bounded_id_search_preserves_missing_and_first_match() {
    libraries();
    let src = r#"{ x -> ids := 17 9 17; '{ id -> 0 = id ? 0; at := w_here ids = id; 0 = t_ally at ? 0; f_irst at } e_ach x }"#;
    let params = [(
        "x".into(),
        Ty {
            scalar: Scalar::I64,
            shape: Shape(vec![4]),
        },
    )];
    let c = check(
        xetal_gpu_lower::lower_function_regions(src, &params, Path::new("."), "", true)
            .unwrap()
            .program,
    )
    .unwrap();
    assert_eq!(text::parse(&text::print(&c.program)).unwrap(), c.program);
    let bound = HashMap::from([("x".into(), Array::ints(Shape(vec![4]), vec![0, 17, 9, 88]))]);
    let expected = interp::run(&c, &bound).unwrap();
    assert_eq!(expected[0].data, Data::Int(vec![0, 1, 2, 0]));
    for d in xetal_gpu_runtime::devices().unwrap() {
        let p = plan(&c, &Schedule::default());
        assert_eq!(p.launches.iter().filter(|l| l.kernel.starts_with("region_")).count(), 1);
        let (mut dev, _) = xetal_gpu_runtime::Prepared::new(&c, &p, &bound, d.index).unwrap();
        assert_eq!(dev.run_timed().unwrap().0, expected);
    }
}
#[test]
fn actual_glass_path_enters_exits_and_retains_medium_identity() {
    libraries();
    let src = r#"{ origins directions mediumids sampleids indices far ->
 geometry := (2 3 3 r_eshape -5.0 -5.0 -5.0 5.0 -5.0 5.0 5.0 -5.0 -5.0 -5.0 -5.0 -5.0 -5.0 -5.0 5.0 5.0 -5.0 5.0, 1 4 r_eshape 0.0 0.0 0.0 1.0);
 links := (1 2, 1 r_eshape 10, 0 0, 1 r_eshape 7, 2 2, 1 r_eshape 1);
 shading := (2 4 r_eshape 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0, 1 0, 2 5 r_eshape 1.0 1.5 0.5 0.25 0.0 0.0 1.0 0.0 0.0 0.0, 2 3 r_eshape 0.0, 1 6 r_eshape 0.0 10.0 10.0 0.0 0.0 0.0);
 scene := (geometry gl:p_reflight links)_ shading;
 clip := 0.00001 c_at far c_at 1e-10 0.000001;
 env := 2 3 r_eshape 0.0 0.0 0.0 1.0 1.0 1.0;
 '{ component ->
 (result, no, nd, beta, medium, active, counts, clipped) := (((((((scene gl:t_race clip)_ env)_ origins)_ directions)_ mediumids)_ sampleids)_ 0 3);
 packed := (r_avel result) c_at (r_avel beta) c_at (f_loat medium) c_at (f_loat active) c_at (f_loat counts) c_at f_loat clipped
 component s_elect packed
 } e_ach indices
 }"#;
    let params = [
        ("origins", Scalar::F64, vec![2, 3]),
        ("directions", Scalar::F64, vec![2, 3]),
        ("mediumids", Scalar::I64, vec![2]),
        ("sampleids", Scalar::I64, vec![2]),
        ("indices", Scalar::I64, vec![20]),
        ("far", Scalar::F64, vec![]),
    ]
    .map(|(n, scalar, shape)| {
        (
            n.into(),
            Ty {
                scalar,
                shape: Shape(shape),
            },
        )
    });
    let c = check(
        xetal_gpu_lower::lower_function_regions(src, &params, Path::new("."), "\"gl:\" u_se< \"Glass\"", true)
            .unwrap()
            .program,
    )
    .unwrap();
    let bound = HashMap::from([
        ("origins".into(), Array::floats(Shape(vec![2, 3]), vec![0., 0., 3., 0., 0., 0.])),
        ("directions".into(), Array::floats(Shape(vec![2, 3]), [0., 0., -1.].repeat(2))),
        ("mediumids".into(), Array::ints(Shape(vec![2]), vec![0, 7])),
        ("sampleids".into(), Array::ints(Shape(vec![2]), vec![1, 1])),
        ("indices".into(), Array::ints(Shape(vec![20]), (1..=20).collect())),
        ("far".into(), Array::floats(Shape::scalar(), vec![100.])),
    ]);
    let reference = interp::run(&c, &bound).unwrap();
    let Data::Float(values) = &reference[0].data else { panic!() };
    assert_eq!(&values[12..14], &[0., 0.]);
    assert_eq!(&values[16..18], &[2., 1.]);
    // Both transmitted paths attenuate red more than green, with no blue absorption.
    assert!(values[6] < values[7] && values[7] < values[8]);
    assert!(values[9] < values[10] && values[10] < values[11]);
    let mut clipped = bound.clone();
    clipped.insert("far".into(), Array::floats(Shape::scalar(), vec![0.5]));
    let clipped_reference = interp::run(&c, &clipped).unwrap();
    let Data::Float(short) = &clipped_reference[0].data else { panic!() };
    assert_eq!(&short[12..14], &[0., 7.]);
    assert_eq!(&short[16..20], &[0., 0., 0., 1.]);
    let p = plan(
        &c,
        &Schedule {
            float: Width::W64,
            ..Schedule::default()
        },
    );
    for d in xetal_gpu_runtime::devices().unwrap().into_iter().filter(|d| d.fp64) {
        let (mut dev, _) = xetal_gpu_runtime::Prepared::new(&c, &p, &bound, d.index).unwrap();
        close(&dev.run_timed().unwrap().0, &reference);
        dev.rebind("far", &Array::floats(Shape::scalar(), vec![0.5])).unwrap();
        close(&dev.run_timed().unwrap().0, &clipped_reference);
        dev.rebind("far", &bound["far"]).unwrap();
        close(&dev.run_timed().unwrap().0, &reference);
    }
}
#[test]
fn device_counted_repeat_preserves_sampling_and_rejects_negative_counts() {
    let params = [
        (
            "x".into(),
            Ty {
                scalar: Scalar::I64,
                shape: Shape(vec![3]),
            },
        ),
        (
            "count".into(),
            Ty {
                scalar: Scalar::I64,
                shape: Shape::scalar(),
            },
        ),
    ];
    let source = "{ x count -> '{ value -> count '{ state -> (48271 * state) m_od 2147483647 } p_ower value } e_ach x }";
    let c = check(
        xetal_gpu_lower::lower_function_regions(source, &params, Path::new("."), "", true)
            .unwrap()
            .program,
    )
    .unwrap();
    assert_eq!(text::parse(&text::print(&c.program)).unwrap(), c.program);
    let mut bound = HashMap::from([
        ("x".into(), Array::ints(Shape(vec![3]), vec![1, 12, 99])),
        ("count".into(), Array::ints(Shape::scalar(), vec![3])),
    ]);
    let p = plan(&c, &Schedule::default());
    assert_eq!(p.launches.len(), 1);
    for d in xetal_gpu_runtime::devices().unwrap() {
        let (mut dev, _) = xetal_gpu_runtime::Prepared::new(&c, &p, &bound, d.index).unwrap();
        for n in [0, 1, 7, -1, 3] {
            let count = Array::ints(Shape::scalar(), vec![n]);
            bound.insert("count".into(), count.clone());
            dev.rebind("count", &count).unwrap();
            if n < 0 {
                assert_eq!(dev.run_timed().unwrap_err().0, "p_ower requires a nonnegative count");
            } else {
                assert_eq!(dev.run_timed().unwrap().0, interp::run(&c, &bound).unwrap());
            }
        }
    }
}
#[test]
fn empty_setup_arrays_do_not_reuse_kernel_names() {
    let p = text::parse(
        "%x = input f64 [0]\n%a = map eq %x %x\n%b = map not %a\n%v = input i64 [1]\n%n = map add %v %v\noutput %a\noutput %b\noutput %n\n",
    )
    .unwrap();
    let c = check(p).unwrap();
    let p = plan(&c, &Schedule::default());
    let names: std::collections::HashSet<_> = p.kernels.iter().map(|k| k.name.as_str()).collect();
    assert_eq!(names.len(), p.kernels.len());
    let bound = HashMap::from([
        ("x".into(), Array::floats(Shape(vec![0]), vec![])),
        ("v".into(), Array::ints(Shape(vec![1]), vec![3])),
    ]);
    for d in xetal_gpu_runtime::devices().unwrap() {
        let (mut dev, _) = xetal_gpu_runtime::Prepared::new(&c, &p, &bound, d.index).unwrap();
        assert_eq!(dev.run_timed().unwrap().0, interp::run(&c, &bound).unwrap());
    }
}

#[test]
fn first_of_empty_dynamic_positions_returns_the_evaluator_zero() {
    let params = [(
        "x".into(),
        Ty {
            scalar: Scalar::I64,
            shape: Shape(vec![3]),
        },
    )];
    let c = check(
        xetal_gpu_lower::lower_function_regions(
            "{ x -> '{ item -> f_irst w_here (17 9 17) = item } e_ach x }",
            &params,
            Path::new("."),
            "",
            true,
        )
        .unwrap()
        .program,
    )
    .unwrap();
    let bound = HashMap::from([("x".into(), Array::ints(Shape(vec![3]), vec![0, 17, 9]))]);
    let expected = interp::run(&c, &bound).unwrap();
    assert_eq!(expected[0].data, Data::Int(vec![0, 1, 2]));
    for d in xetal_gpu_runtime::devices().unwrap() {
        let p = plan(&c, &Schedule::default());
        let (mut dev, _) = xetal_gpu_runtime::Prepared::new(&c, &p, &bound, d.index).unwrap();
        assert_eq!(dev.run_timed().unwrap().0, expected);
    }
}
