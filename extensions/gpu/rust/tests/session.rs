//! The gpu extension through the raw ABI, as xetal-x's loader calls
//! it: its descriptor validates, and a session (load, bind, run,
//! output) gives the reference interpreter's answers and, when an
//! OpenCL device is present, the device's.
#![allow(unsafe_code)]

use xetal_ext_sdk::abi::{
    copy_foreign_error, copy_foreign_value, validate_descriptor, AbiErrorV1, AbiValue,
    EncodedValue, ErrorCode, ValidatedExtension,
};
use xetal_ext_sdk::{Array, ArrayData, OwnedError, Value};

fn gpu() -> ValidatedExtension {
    unsafe { validate_descriptor(xetal_ext_gpu::__xetal_extension::descriptor()) }.unwrap()
}

fn call(name: &str, args: &[Value]) -> Result<Value, OwnedError> {
    let ext = gpu();
    let f = ext.function(name).unwrap();
    let encoded: Vec<EncodedValue> = args.iter().map(EncodedValue::new).collect();
    let raw: Vec<AbiValue> = encoded.iter().map(|e| *e.as_raw()).collect();
    let mut out = AbiValue::zero();
    let mut err = AbiErrorV1::none();
    let code = unsafe { (f.invoke())(raw.as_ptr(), raw.len(), &mut out, &mut err) };
    if code == ErrorCode::Ok as u32 {
        Ok(unsafe { copy_foreign_value(&out) }.unwrap())
    } else {
        Err(unsafe { copy_foreign_error(&err) }.unwrap())
    }
}

fn floats(shape: &[usize], v: &[f64]) -> Value {
    Value::Array(Array::new(shape.to_vec(), ArrayData::Float(v.to_vec())).unwrap())
}

const MATVEC: &str = "%a = input f64 [2 3]\n%v = input f64 [3]\n%y = matmul %a %v\n%s = reduce add %y\noutput %y\noutput %s\n";

#[test]
fn descriptor_lists_every_function() {
    let names: Vec<String> = gpu()
        .functions()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    assert_eq!(
        names,
        ["devices", "load", "bind", "schedule", "run", "output", "explain", "kernel"]
    );
}

#[test]
fn a_session_on_the_interpreter() {
    assert_eq!(
        call("load", &[Value::Text(MATVEC.into())]).unwrap(),
        Value::Int(2)
    );
    assert_eq!(
        call(
            "bind",
            &[
                Value::Int(1),
                floats(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            ]
        )
        .unwrap(),
        Value::Int(1)
    );
    assert_eq!(
        call("bind", &[Value::Int(2), floats(&[3], &[1.0, 0.0, 2.0])]).unwrap(),
        Value::Int(2)
    );
    assert_eq!(call("run", &[Value::Int(-1)]).unwrap(), Value::Int(2));
    assert_eq!(
        call("output", &[Value::Int(1)]).unwrap(),
        floats(&[2], &[7.0, 16.0])
    );
    assert_eq!(
        call("output", &[Value::Int(2)]).unwrap(),
        Value::Float(23.0)
    );
    let Value::Text(plan) = call("explain", &[]).unwrap() else {
        panic!("text")
    };
    assert!(plan.contains("matmul_float"), "{plan}");
}

#[test]
fn ints_come_back_as_ints() {
    call(
        "load",
        &[Value::Text(
            "%a = input i64 [3]\n%k = const i64 [] 2\n%b = map mul %k %a\noutput %b\n".into(),
        )],
    )
    .unwrap();
    // The bridge sends numbers as Floats; whole ones bind to an Int input.
    call("bind", &[Value::Int(1), floats(&[3], &[1.0, 2.0, 3.0])]).unwrap();
    call("run", &[Value::Int(-1)]).unwrap();
    assert_eq!(
        call("output", &[Value::Int(1)]).unwrap(),
        Value::Array(Array::new(vec![3], ArrayData::Int(vec![2, 4, 6])).unwrap())
    );
    let e = call("bind", &[Value::Int(1), floats(&[3], &[1.0, 2.5, 3.0])]).unwrap_err();
    assert_eq!(e.message(), "input 1 (%a) is i64 [3]: 2.5 is not whole");
}

#[test]
fn errors_name_what_is_wrong() {
    let e = call("load", &[Value::Text("%a = frob\n".into())]).unwrap_err();
    assert_eq!(e.code(), ErrorCode::InvalidArgument);
    assert!(
        e.message().starts_with("line 1: unknown operation `frob`"),
        "{}",
        e.message()
    );
    call("load", &[Value::Text(MATVEC.into())]).unwrap();
    let e = call("bind", &[Value::Int(3), floats(&[3], &[1.0, 2.0, 3.0])]).unwrap_err();
    assert_eq!(e.message(), "the program has 2 input(s); no input 3");
    let e = call(
        "bind",
        &[Value::Int(2), floats(&[4], &[1.0, 2.0, 3.0, 4.0])],
    )
    .unwrap_err();
    assert_eq!(e.message(), "input 2 (%v) is f64 [3], given shape [4]");
    let e = call("run", &[Value::Int(-1)]).unwrap_err();
    assert_eq!(e.message(), "input %a (f64 [2 3]) is not bound");
    let e = call("schedule", &[Value::Text("tile = 32\n".into())]).unwrap_err();
    assert!(e.message().starts_with("tile = 32"), "{}", e.message());
}

#[test]
fn a_session_on_the_device() {
    let Value::Text(found) = call("devices", &[]).unwrap() else {
        panic!("text")
    };
    if !found.starts_with("opencl:0") {
        eprintln!("skipped: no OpenCL device");
        return;
    }
    call("load", &[Value::Text(MATVEC.into())]).unwrap();
    call(
        "bind",
        &[
            Value::Int(1),
            floats(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ],
    )
    .unwrap();
    call("bind", &[Value::Int(2), floats(&[3], &[1.0, 0.0, 2.0])]).unwrap();
    call("schedule", &[Value::Text("tile = 2\n".into())]).unwrap();
    assert_eq!(call("run", &[Value::Int(0)]).unwrap(), Value::Int(2));
    assert_eq!(
        call("output", &[Value::Int(1)]).unwrap(),
        floats(&[2], &[7.0, 16.0])
    );
    let Value::Text(src) = call("kernel", &[]).unwrap() else {
        panic!("text")
    };
    assert!(src.contains("matmul_tiled2_float"), "{src}");
}
