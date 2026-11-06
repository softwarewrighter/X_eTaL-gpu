//! Strict lowering and changed-input checks for renderer mask operations.

use std::collections::HashMap;
use std::path::Path;
use xetal_gpu_xir::{check, interp, Array, Data, Scalar, Shape, Ty};

#[test]
fn boolean_truth_tables_and_scalar_extension_use_bound_inputs() {
    let ty = Ty {
        scalar: Scalar::Bool,
        shape: Shape(vec![4]),
    };
    let cases = [
        ("{ x y -> n_ot x }", vec![true, true, false, false]),
        ("{ x y -> x & y }", vec![false, false, false, true]),
        ("{ x y -> x | y }", vec![false, true, true, true]),
    ];
    for (source, expected) in cases {
        let lowered =
            xetal_gpu_lower::lower_function(source, &[("x".into(), ty.clone()), ("y".into(), ty.clone())], Path::new("."), "").unwrap();
        assert!(lowered.reasons.is_empty());
        let checked = check(lowered.program).unwrap();
        let array = |v| Array::new(Scalar::Bool, Shape(vec![4]), Data::Bool(v)).unwrap();
        let mut inputs = HashMap::from([
            ("x".into(), array(vec![false, false, true, true])),
            ("y".into(), array(vec![false, true, false, true])),
        ]);
        let first = interp::run(&checked, &inputs).unwrap();
        assert_eq!(first[0].data, Data::Bool(expected), "{source}");
        inputs.insert("x".into(), array(vec![true, true, false, false]));
        let second = interp::run(&checked, &inputs).unwrap();
        assert_ne!(first, second, "changing x must change {source}");
    }
}

#[test]
fn boolean_scalar_extends_over_a_mask() {
    let params = [
        (
            "x".into(),
            Ty {
                scalar: Scalar::Bool,
                shape: Shape::scalar(),
            },
        ),
        (
            "y".into(),
            Ty {
                scalar: Scalar::Bool,
                shape: Shape(vec![4]),
            },
        ),
    ];
    let lowered = xetal_gpu_lower::lower_function("{ x y -> x | y }", &params, Path::new("."), "").unwrap();
    let checked = check(lowered.program).unwrap();
    for flag in [false, true] {
        let inputs = HashMap::from([
            (
                "x".into(),
                Array::new(Scalar::Bool, Shape::scalar(), Data::Bool(vec![flag])).unwrap(),
            ),
            (
                "y".into(),
                Array::new(Scalar::Bool, Shape(vec![4]), Data::Bool(vec![false, true, false, true])).unwrap(),
            ),
        ]);
        assert_eq!(
            interp::run(&checked, &inputs).unwrap()[0].data,
            Data::Bool(vec![flag, true, flag, true])
        );
    }
}

#[test]
fn logic_keeps_numeric_value_and_shape_checks() {
    let ty = Ty {
        scalar: Scalar::I64,
        shape: Shape(vec![4]),
    };
    for source in ["{ x -> n_ot x }", "{ x -> x & x }", "{ x -> x | x }"] {
        let lowered = xetal_gpu_lower::lower_function(source, &[("x".into(), ty.clone())], Path::new("."), "").unwrap();
        let checked = check(lowered.program).unwrap();
        let valid = HashMap::from([("x".into(), Array::ints(Shape(vec![4]), vec![0, 1, 0, 1]))]);
        assert!(interp::run(&checked, &valid).is_ok());
        let invalid = HashMap::from([("x".into(), Array::ints(Shape(vec![4]), vec![0, 1, 2, 1]))]);
        assert!(interp::run(&checked, &invalid).unwrap_err().0.contains("expected a Bool"));
    }
    let params = [
        (
            "x".into(),
            Ty {
                scalar: Scalar::Bool,
                shape: Shape(vec![4]),
            },
        ),
        (
            "y".into(),
            Ty {
                scalar: Scalar::Bool,
                shape: Shape(vec![3]),
            },
        ),
    ];
    assert!(xetal_gpu_lower::lower_function("{ x y -> x & y }", &params, Path::new("."), "").is_err());
}
