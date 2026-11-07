//! Region contracts reject malformed state/captures and keep zero repeats inert.
use std::collections::HashMap;
use xetal_gpu_xir::{check, interp, text, Array, Data, Op, RegionKind, Shape, Value, ValueId};

fn wrapper(body: &str, kind: RegionKind) -> xetal_gpu_xir::Program {
    let mut p = text::parse("%x = const i64 [2] 3 4\noutput %x\n").unwrap();
    p.values.push(Value {
        name: "loop".into(),
        ty: None,
        op: Op::Region {
            kind,
            args: vec![ValueId(0)],
            body: Box::new(text::parse(body).unwrap()),
        },
    });
    p.outputs = vec![ValueId(1)];
    p
}

#[test]
fn validates_fixed_state_capture_and_scalar_map_contracts() {
    let changed = wrapper(
        "%s = input i64 [2]\n%r = ravel %s\n%one = take 1 %r\noutput %one\n",
        RegionKind::Repeat(3),
    );
    assert!(check(changed).unwrap_err().0.contains("repeated state must keep type"));
    let missing = wrapper("%s = input i64 [2]\n%c = input i64 []\noutput %s\n", RegionKind::Repeat(3));
    assert!(check(missing).unwrap_err().0.contains("inputs must match arguments"));
    let nonscalar = wrapper("%s = input i64 []\n%r = reshape [2] %s\noutput %r\n", RegionKind::Each);
    assert!(check(nonscalar).unwrap_err().0.contains("must return a scalar"));
    let mut invalid = wrapper("%s = input i64 [2]\noutput %s\n", RegionKind::Repeat(1));
    if let Op::Region { body, .. } = &mut invalid.values[1].op {
        body.outputs[0] = ValueId(99);
    }
    assert_eq!(check(invalid).unwrap_err().0, "program output is not a defined value");
}

#[test]
fn zero_repeat_never_executes_body_guard() {
    let p = wrapper(
        "%s = input i64 [2]\n%bad = const bool [] 1\n%g = guard %bad x6e65766572\noutput %s\n",
        RegionKind::Repeat(0),
    );
    assert_eq!(text::parse(&text::print(&p)).unwrap(), p);
    let c = check(p).unwrap();
    assert_eq!(
        interp::run(&c, &HashMap::new()).unwrap(),
        vec![Array::ints(Shape(vec![2]), vec![3, 4])]
    );
    let Op::Region { body, .. } = &c.program.values[1].op else {
        panic!()
    };
    assert_eq!(
        interp::run(
            &check((**body).clone()).unwrap(),
            &HashMap::from([("s".into(), Array::ints(Shape(vec![2]), vec![3, 4]))])
        )
        .unwrap_err()
        .0,
        "never"
    );
    assert!(matches!(interp::run(&c, &HashMap::new()).unwrap()[0].data, Data::Int(_)));
}
