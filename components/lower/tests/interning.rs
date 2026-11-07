//! Repeated immutable validation stays correct across input changes and errors.
use std::{collections::HashMap, path::Path};
use xetal_gpu_xir::{check, interp, Array, Data, Op, Scalar, Shape, Ty};

#[test]
fn repeated_checks_share_work_without_merging_messages_or_inputs() {
    let source = "{ x y -> x = 0 ? @ p_anic< \"first\"; x = 0 ? @ p_anic< \"first\"; x = 0 ? @ p_anic< \"later\"; y = 0 ? @ p_anic< \"second\"; (x + y) * (x + y) }";
    let ty = Ty {
        scalar: Scalar::I64,
        shape: Shape::scalar(),
    };
    let result = xetal_gpu_lower::lower_function(source, &[("x".into(), ty.clone()), ("y".into(), ty)], Path::new("."), "").unwrap();
    assert!(result.reasons.is_empty());
    assert_eq!(result.program.inputs().len(), 2);
    assert_eq!(result.program.values.iter().filter(|v| matches!(v.op, Op::Guard { .. })).count(), 3);
    let checked = check(result.program).unwrap();
    for (x, y, expected) in [(2, 3, Ok(25)), (0, 0, Err("first")), (2, 0, Err("second")), (3, 4, Ok(49))] {
        let inputs = HashMap::from([
            ("x".into(), Array::ints(Shape::scalar(), vec![x])),
            ("y".into(), Array::ints(Shape::scalar(), vec![y])),
        ]);
        let output = interp::run(&checked, &inputs);
        match expected {
            Ok(n) => assert_eq!(output.unwrap()[0].data, Data::Int(vec![n])),
            Err(message) => assert_eq!(output.unwrap_err().0, message),
        }
    }
}
