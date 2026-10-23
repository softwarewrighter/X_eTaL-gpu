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
