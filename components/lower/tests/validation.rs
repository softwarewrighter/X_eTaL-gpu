//! Metadata guards decide during lowering; data validation remains explicit.
use std::collections::HashMap;
use std::path::Path;
use xetal_gpu_xir::{check, interp, Array, Data, Scalar, Shape, Ty};

fn lower(source: &str, shape: Vec<usize>, scalar: Scalar) -> xetal_gpu_xir::Result<xetal_gpu_lower::Lowered> {
    xetal_gpu_lower::lower_function(
        source,
        &[(
            "x".into(),
            Ty {
                scalar,
                shape: Shape(shape),
            },
        )],
        Path::new("."),
        "",
    )
}

#[test]
fn shape_guards_resolve_and_preserve_the_failure_message() {
    let source = "{ x -> n_ot (2 3 m_atch s_hape x) ? @ p_anic< \"expected 2 by 3\"; x + 1 }";
    let result = lower(source, vec![2, 3], Scalar::I64).unwrap();
    assert!(result.reasons.is_empty());
    let checked = check(result.program).unwrap();
    for n in [1, 4] {
        let inputs = HashMap::from([("x".into(), Array::ints(Shape(vec![2, 3]), vec![n; 6]))]);
        assert_eq!(interp::run(&checked, &inputs).unwrap()[0].data, Data::Int(vec![n + 1; 6]));
    }
    assert_eq!(lower(source, vec![3, 2], Scalar::I64).unwrap_err().0, "expected 2 by 3");
    let source = "{ x -> n_ot '& r_/ (s_hape x) > 0 ? @ p_anic< \"empty axis\"; x + 1 }";
    assert!(lower(source, vec![2, 3], Scalar::I64).is_ok());
    assert_eq!(lower(source, vec![0, 3], Scalar::I64).unwrap_err().0, "empty axis");
}

#[test]
fn general_dynamic_branches_remain_refused() {
    let source = "{ x -> x > 0 ? x + 1; x - 1 }";
    let error = lower(source, vec![], Scalar::F64).unwrap_err().0;
    assert!(error.contains("general dynamic branches"), "{error}");
}

#[test]
fn logical_reductions_match_known_axes_and_empty_identities() {
    let inputs = HashMap::from([("x".into(), Array::ints(Shape(vec![2, 3]), vec![1, -1, 2, 3, 4, 5]))]);
    for (source, expected) in [
        ("{ x -> '& r_/ x > 0 }", vec![true, false, true]),
        ("{ x -> '| r_/ x > 0 }", vec![true, true, true]),
        ("{ x -> '& r_/_2 x > 0 }", vec![false, true]),
        ("{ x -> '| r_/_2 x > 0 }", vec![true, true]),
        ("{ x -> '& r_/_12 x > 0 }", vec![false]),
        ("{ x -> '| r_/_12 x > 0 }", vec![true]),
    ] {
        let result = lower(source, vec![2, 3], Scalar::I64).unwrap();
        assert_eq!(
            interp::run(&check(result.program).unwrap(), &inputs).unwrap()[0].data,
            Data::Bool(expected),
            "{source}"
        );
    }
    for shape in [vec![0], vec![0, 3], vec![2, 0]] {
        for (op, identity) in [("&", true), ("|", false)] {
            let source = format!("{{ x -> '{op} r_/ x }}");
            let result = lower(&source, shape.clone(), Scalar::Bool).unwrap();
            let inputs = HashMap::from([(
                "x".into(),
                Array::new(Scalar::Bool, Shape(shape.clone()), Data::Bool(vec![])).unwrap(),
            )]);
            let out = interp::run(&check(result.program).unwrap(), &inputs).unwrap();
            let count = if shape.len() == 1 { 1 } else { shape[1] };
            assert_eq!(out[0].data, Data::Bool(vec![identity; count]));
        }
    }
}

#[test]
fn match_is_shape_sensitive_and_compares_all_items() {
    for (source, shape, expected) in [
        ("{ x -> x m_atch x }", vec![0], true),
        ("{ x -> x m_atch x }", vec![2, 3], true),
        ("{ x -> x m_atch (6 r_eshape x) }", vec![2, 3], false),
        ("{ x -> x m_atch (x + 1) }", vec![2, 3], false),
    ] {
        let result = lower(source, shape.clone(), Scalar::I64).unwrap();
        let n = shape.iter().product();
        let inputs = HashMap::from([("x".into(), Array::ints(Shape(shape), vec![1; n]))]);
        assert_eq!(
            interp::run(&check(result.program).unwrap(), &inputs).unwrap()[0].data,
            Data::Bool(vec![expected])
        );
    }
}

#[test]
fn match_keeps_numeric_limits_explicit_and_nan_unequal() {
    assert!(lower("{ x -> x m_atch f_loat x }", vec![2], Scalar::I64)
        .unwrap_err()
        .0
        .contains("different numeric scalar types"));
    assert!(lower("{ x -> '& r_/ x }", vec![2], Scalar::I64).is_ok());
    let lowered = lower("{ x -> x m_atch x }", vec![2], Scalar::F64).unwrap();
    let inputs = HashMap::from([("x".into(), Array::floats(Shape(vec![2]), vec![1.0, f64::NAN]))]);
    assert_eq!(
        interp::run(&check(lowered.program).unwrap(), &inputs).unwrap()[0].data,
        Data::Bool(vec![false])
    );
    assert!(lower(
        "{ x -> n_ot (\"a\" m_atch \"a\") ? @ p_anic< \"wrong\"; x + 1 }",
        vec![2],
        Scalar::I64
    )
    .is_ok());
}
