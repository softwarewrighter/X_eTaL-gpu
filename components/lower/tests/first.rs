//! f_irst removes the leading axis, as the evaluator specifies.
use std::collections::HashMap;
use std::path::Path;
use xetal_gpu_xir::{check, interp, Array, Data, Scalar, Shape, Ty};

#[test]
fn first_preserves_the_remaining_shape_and_empty_prototype() {
    for (shape, data, result_shape, expected) in [
        (vec![], vec![4], vec![], vec![4]),
        (vec![3], vec![1, 2, 3], vec![], vec![1]),
        (vec![2, 3], vec![1, 2, 3, 4, 5, 6], vec![3], vec![1, 2, 3]),
        (vec![0], vec![], vec![], vec![0]),
        (vec![0, 3], vec![], vec![3], vec![0, 0, 0]),
    ] {
        let lowered = xetal_gpu_lower::lower_function(
            "{ x -> f_irst x }",
            &[(
                "x".into(),
                Ty {
                    scalar: Scalar::I64,
                    shape: Shape(shape.clone()),
                },
            )],
            Path::new("."),
            "",
        )
        .unwrap();
        let checked = check(lowered.program).unwrap();
        let inputs = HashMap::from([("x".into(), Array::ints(Shape(shape), data))]);
        let output = interp::run(&checked, &inputs).unwrap();
        assert_eq!(output[0].shape, Shape(result_shape));
        assert_eq!(output[0].data, Data::Int(expected));
    }
    let axis = xetal_gpu_lower::lower_function(
        "{ x -> f_irst_2 x }",
        &[(
            "x".into(),
            Ty {
                scalar: Scalar::I64,
                shape: Shape(vec![2, 3]),
            },
        )],
        Path::new("."),
        "",
    )
    .unwrap();
    let checked_axis = check(axis.program).unwrap();
    let bound = HashMap::from([("x".into(), Array::ints(Shape(vec![2, 3]), vec![1, 2, 3, 4, 5, 6]))]);
    assert_eq!(
        interp::run(&checked_axis, &bound).unwrap()[0],
        Array::ints(Shape(vec![2]), vec![1, 4])
    );
    let lowered = xetal_gpu_lower::lower_function(
        "{ x -> (f_irst 2 3 r_eshape 1 2 3 4 5 6) + x }",
        &[(
            "x".into(),
            Ty {
                scalar: Scalar::I64,
                shape: Shape::scalar(),
            },
        )],
        Path::new("."),
        "",
    )
    .unwrap();
    let checked = check(lowered.program).unwrap();
    let inputs = HashMap::from([("x".into(), Array::ints(Shape::scalar(), vec![4]))]);
    assert_eq!(
        interp::run(&checked, &inputs).unwrap()[0],
        Array::ints(Shape(vec![3]), vec![5, 6, 7])
    );
}
