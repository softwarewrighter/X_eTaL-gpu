//! Device regressions at the type and shape boundaries, beyond the demos.

use std::collections::HashMap;

use xetal_gpu_mojo::host;
use xetal_gpu_opencl::{Schedule, Width};
use xetal_gpu_xir::{check, interp, text, Array, Data, Scalar, Shape};

#[test]
fn typed_abi_axis_folds_and_indexed_reads_match_the_interpreter() {
    if host::mojo_path().is_none() {
        eprintln!("skipped: Mojo semantics device tests (no toolchain)");
        return;
    }
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        Array::new(
            Scalar::I64,
            Shape(vec![2]),
            Data::Int(vec![9_007_199_254_740_993, -9_007_199_254_740_993]),
        )
        .unwrap(),
    );
    let cases = [
        // Both bound inputs and uploaded constants must retain every i64 bit.
        ("exact_ints", "%x = input i64 [2]\n%c = const i64 [2] 9007199254740995 -9007199254740995\n%y = map add %x %c\n%b = cast bool %y\n%n = cast i64 %b\noutput %x\noutput %c\noutput %y\noutput %b\noutput %n\n"),
        // Strided folds on all axes, all four reductions, singleton lines,
        // and length changes between elementwise stages.
        ("axes", "%x = const i64 [2 3 2] 1 2 3 4 5 6 7 8 9 10 11 12\n%a = reduce add axis=1 %x\n%b = reduce mul axis=2 %x\n%c = reduce min axis=3 %x\n%d = reduce max axis=2 %x\n%k = const i64 [] -2\n%y = map mul %a %k\n%z = reduce add %y\n%one = const i64 [2 1] 4 5\n%r = reduce add axis=2 %one\noutput %a\noutput %b\noutput %c\noutput %d\noutput %z\noutput %r\n"),
        // Bool buffers, scalar extension in tables, repeated arguments,
        // slices along a later axis, and zero-sized results.
        ("indexed", "%x = const i64 [2 3] 1 2 3 4 5 6\n%a = take -2 axis=2 %x\n%b = drop 1 axis=2 %x\n%e = take 0 %x\n%v = ravel %a\n%t = const bool [2] true false\n%u = table and %t %t\n%k = const i64 [] 3\n%w = table mul %v %k\n%single = const i64 [1] 3\n%s = reduce add %single\n%f = const f64 [3] -1.75 0.0 2.9\n%i = cast i64 %f\noutput %a\noutput %b\noutput %e\noutput %u\noutput %w\noutput %s\noutput %i\n"),
    ];
    for (name, source) in cases {
        let checked = check::check(text::parse(source).unwrap()).unwrap();
        let expected = interp::run(&checked, &inputs).unwrap();
        let (got, info, _, _) = host::execute(&checked, &Schedule::default(), name, &inputs).unwrap();
        assert_eq!(got, expected, "{name} on {info}");
        eprintln!("ok: {name} on {info}");
        if name != "exact_ints" {
            let schedule = Schedule {
                int: Width::W32,
                ..Schedule::default()
            };
            let (got, _, _, _) = host::execute(&checked, &schedule, &format!("{name}_i32"), &inputs).unwrap();
            assert_eq!(got, expected, "{name} with i32 schedule");
        }
    }
}

/// The structural operations of step 3 against the interpreter: concat
/// (a single value joining a vector, and along a later axis), the TopK
/// operation (ties, a line with fewer distinct values than k, along the
/// first axis), reshape cycling and filling, rotations by negative and
/// oversized amounts, lookups with bound positions (a stray one
/// refused), and inner products in tiles at sizes no tile divides,
/// exact for Ints and the evaluator's right fold for Floats.
#[test]
fn structure_and_tiled_products_match_the_interpreter() {
    if host::mojo_path().is_none() {
        eprintln!("skipped: Mojo structure device tests (no toolchain)");
        return;
    }
    let mut inputs = HashMap::new();
    inputs.insert(
        "ids".into(),
        Array::new(Scalar::I64, Shape(vec![4]), Data::Int(vec![3, 1, 1, 2])).unwrap(),
    );
    let cases = [
        ("concat", "%v = const i64 [3] 1 2 3\n%s = const i64 [] 9\n%a = concat %s %v\n%m = const f64 [2 2] 1.0 2.0 3.0 4.0\n%n = const f64 [2 1] 5.0 6.0\n%b = concat axis=2 %m %n\n%t = const bool [2] true false\n%u = concat %t %t\noutput %a\noutput %b\noutput %u\n"),
        ("topk", "%m = const f64 [3 4] 5.0 1.0 9.0 9.0 2.0 2.0 2.0 2.0 7.0 8.0 6.0 8.0\n%a = topk 2 %m\n%b = topk 3 %m\n%c = topk 1 axis=1 %m\n%i = const i64 [5] 4 -1 4 0 7\n%d = topk 2 %i\noutput %a\noutput %b\noutput %c\noutput %d\n"),
        ("reshape", "%v = const i64 [5] 1 2 3 4 5\n%a = reshape [3 4] %v\n%k = const f64 [] 2.5\n%b = reshape [2 3] %k\n%c = reshape [2 2] %v\noutput %a\noutput %b\noutput %c\n"),
        ("rotate", "%v = const i64 [5] 1 2 3 4 5\n%a = rotate -7 %v\n%b = rotate 12 0 -1 %v\n%m = const f64 [2 3] 1.0 2.0 3.0 4.0 5.0 6.0\n%c = rotate axis=1 1 %m\n%d = rotate axis=2 -1 1 %m\noutput %a\noutput %b\noutput %c\noutput %d\n"),
        ("lookup", "%ids = input i64 [4]\n%m = const i64 [3 2] 1 2 3 4 5 6\n%a = lookup %ids %m\n%p = const i64 [2] 2 1\n%b = lookup axis=2 %p %m\noutput %a\noutput %b\n"),
        ("products", "%a = const f64 [5 7] 0.1 0.2 0.3 0.4 0.5 0.6 0.7 1.1 1.2 1.3 1.4 1.5 1.6 1.7 2.1 2.2 2.3 2.4 2.5 2.6 2.7 3.1 3.2 3.3 3.4 3.5 3.6 3.7 4.1 4.2 4.3 4.4 4.5 4.6 4.7\n%b = const f64 [7 3] 1.0 -1.0 0.5 2.0 0.25 -0.5 3.0 1.5 1.0 -2.0 0.75 2.0 1.0 -1.0 0.5 0.5 2.0 -1.5 -1.0 1.0 1.0\n%c = matmul %a %b\n%i = const i64 [3 5] 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\n%j = const i64 [5 2] 3037000499 -1 2 3 4 5 6 7 8 9\n%k = matmul %i %j\noutput %c\noutput %k\n"),
    ];
    let close = |got: &[Array], want: &[Array]| -> bool {
        got.len() == want.len()
            && got.iter().zip(want).all(|(g, w)| match (&g.data, &w.data) {
                (Data::Float(x), Data::Float(y)) => {
                    g.shape == w.shape && x.iter().zip(y).all(|(p, q)| (p - q).abs() <= 1e-5 * q.abs().max(1.0))
                }
                _ => g == w,
            })
    };
    for (name, source) in cases {
        let checked = check::check(text::parse(source).unwrap()).unwrap();
        let expected = interp::run(&checked, &inputs).unwrap();
        let schedules = [
            ("", Schedule::default()),
            (
                "_t4",
                Schedule {
                    tile: 4,
                    ..Schedule::default()
                },
            ),
        ];
        for (suffix, schedule) in schedules {
            if !suffix.is_empty() && name != "products" {
                continue;
            }
            let (got, info, _, _) = host::execute(&checked, &schedule, &format!("{name}{suffix}"), &inputs).unwrap();
            assert!(close(&got, &expected), "{name}{suffix} on {info}:\n{got:?}\nexpected\n{expected:?}");
            eprintln!("ok: {name}{suffix} on {info}");
        }
    }
    // A stray bound position is refused before the device runs, with
    // the evaluator's message.
    let checked = check::check(text::parse(cases[4].1).unwrap()).unwrap();
    let mut stray = inputs.clone();
    stray.insert(
        "ids".into(),
        Array::new(Scalar::I64, Shape(vec![4]), Data::Int(vec![3, 4, 1, 2])).unwrap(),
    );
    let err = host::execute(&checked, &Schedule::default(), "lookup", &stray).unwrap_err();
    assert!(err.0.contains("lookup: index 4 is outside 1..3"), "{err}");
    eprintln!("ok: a stray bound position is refused ({err})");
}
