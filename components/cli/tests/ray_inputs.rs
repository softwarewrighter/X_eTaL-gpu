//! Actual shared RayTrace entry points must respond to bound GPU inputs.
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use xetal_gpu_opencl::{plan, Schedule};
use xetal_gpu_xir::{check, interp, Array, Data, Scalar, Shape, Ty};

fn floats(shape: &[usize], values: Vec<f64>) -> Array {
    Array::floats(Shape(shape.to_vec()), values)
}

fn close(actual: &[Array], expected: &[Array]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert_eq!(a.ty(), e.ty());
        let (Data::Float(a), Data::Float(e)) = (&a.data, &e.data) else {
            panic!("Float outputs")
        };
        for (i, (a, e)) in a.iter().zip(e).enumerate() {
            assert!(a.is_finite() && (a - e).abs() <= 1e-5 * (1.0 + e.abs()), "item {i}: {a} != {e}");
        }
        assert_eq!(&a[12..], &e[12..], "primitive IDs must agree exactly");
    }
}

#[test]
fn renderer_rebinding_changes_geometry_rays_and_materials_and_preserves_refusals() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let libraries = root.join("work/xetal-libraries/libs");
    assert!(
        libraries.join("RayTrace/src/RayTrace.xtl").is_file(),
        "run just xetal-libraries first"
    );
    let paths: Vec<_> = std::fs::read_dir(&libraries)
        .unwrap()
        .map(|p| p.unwrap().path().join("src"))
        .filter(|p| p.is_dir())
        .collect();
    std::env::set_var("XETAL_PATH", std::env::join_paths(paths).unwrap());
    let source = r#"{ triangles origins directions materials ->
      clip := 1e-6 100.0 1e-10 1e-5;
      lights := 1 6 r_eshape 0.0 1.0 3.0 30.0 30.0 30.0;
      environment := 2 3 r_eshape 0.05 0.05 0.05 0.01 0.02 0.04;
      ids := (((clip rt:n_earest triangles)_ origins)_ directions);
      color := (((((((clip rt:r_ender triangles)_ 1 2)_ materials)_ lights)_ environment)_ origins)_ directions);
      (r_avel color) c_at f_loat ids
    }"#;
    let params = [
        ("triangles", vec![2, 3, 3]),
        ("origins", vec![4, 3]),
        ("directions", vec![4, 3]),
        ("materials", vec![2, 4]),
    ]
    .map(|(name, shape)| {
        (
            name.into(),
            Ty {
                scalar: Scalar::F64,
                shape: Shape(shape),
            },
        )
    });
    let lowered = xetal_gpu_lower::lower_function(source, &params, Path::new("."), "\"rt:\" u_se< \"RayTrace\"").unwrap();
    assert!(lowered.reasons.is_empty());
    assert_eq!(lowered.program.inputs().len(), 4);
    let checked = check(lowered.program).unwrap();
    let mesh = vec![-1., -1., 0., 1., -1., 0., 1., 1., 0., -1., -1., 0., 1., 1., 0., -1., 1., 0.];
    let bound = HashMap::from([
        ("triangles".into(), floats(&[2, 3, 3], mesh.clone())),
        (
            "origins".into(),
            floats(&[4, 3], vec![-0.5, -0.25, 3., 0.5, 0.25, 3., 2., 0., 3., 0., 0., 3.]),
        ),
        ("directions".into(), floats(&[4, 3], [0., 0., -1.].repeat(4))),
        ("materials".into(), floats(&[2, 4], vec![0.8, 0.3, 0.1, 0., 0.1, 0.3, 0.8, 0.35])),
    ]);
    let reference = interp::run(&checked, &bound).unwrap();
    let Data::Float(first) = &reference[0].data else { panic!() };
    assert_eq!(&first[12..], &[2., 1., 0., 1.]);
    let schedule = Schedule::default();
    let p = plan(&checked, &schedule);
    let mut devices = Vec::new();
    for d in xetal_gpu_runtime::devices().unwrap() {
        devices.push(xetal_gpu_runtime::Prepared::new(&checked, &p, &bound, d.index).unwrap().0);
    }
    if devices.is_empty() {
        eprintln!("skipped: actual RayTrace input checks on OpenCL (no device)");
    }
    let cases = [
        ("origins", floats(&[4, 3], [10., 0., 3.].repeat(4)), None),
        (
            "triangles",
            floats(&[2, 3, 3], vec![f64::NAN; 18]),
            Some("RayTrace requires finite numbers and results"),
        ),
        (
            "triangles",
            floats(
                &[2, 3, 3],
                mesh.iter()
                    .enumerate()
                    .map(|(i, n)| n + if i % 3 == 0 { 10. } else { 0. })
                    .collect(),
            ),
            None,
        ),
        ("directions", floats(&[4, 3], [0., 0., 1.].repeat(4)), None),
        ("materials", floats(&[2, 4], vec![0.2, 0.9, 0.2, 0., 0.9, 0.2, 0.1, 0.]), None),
        (
            "directions",
            floats(&[4, 3], vec![0.; 12]),
            Some("RayTrace directions must be nonzero"),
        ),
        (
            "materials",
            floats(&[2, 4], vec![2.; 8]),
            Some("RayTrace material RGB and reflection must be in 0..1"),
        ),
    ];
    // Return to valid inputs after every mutation/failure, using one prepared session.
    for (name, array, error) in cases {
        let mut inputs = bound.clone();
        inputs.insert(name.into(), array);
        let expected = interp::run(&checked, &inputs);
        if let Some(message) = error {
            assert_eq!(expected.as_ref().unwrap_err().0, message);
        } else {
            assert_ne!(expected.as_ref().unwrap(), &reference, "changed {name} had no effect");
        }
        for device in &mut devices {
            device.rebind(name, &inputs[name]).unwrap();
            let actual = device.run_timed().map(|v| v.0);
            match &expected {
                Ok(value) => close(&actual.unwrap(), value),
                Err(error) => assert_eq!(actual.unwrap_err(), *error),
            }
            device.rebind(name, &bound[name]).unwrap();
            close(&device.run_timed().unwrap().0, &reference);
        }
    }
}
