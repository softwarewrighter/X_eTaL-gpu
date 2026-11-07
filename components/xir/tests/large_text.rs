//! Large generated graphs retain SSA references without repeated name scans.
use xetal_gpu_xir::{check, text, Op, ValueId};

#[test]
fn large_graph_references_and_duplicate_refusals() {
    let mut source = "%x0 = input f64 []\n".to_string();
    for i in 1..20000 {
        source += &format!("%x{i} = map add %x{} %x0\n", i - 1);
    }
    source += "output %x19999\n";
    let program = text::parse(&source).unwrap();
    assert_eq!(program.outputs, vec![ValueId(19999)]);
    for (i, value) in program.values.iter().enumerate().skip(1) {
        let Op::Map { args, .. } = &value.op else { panic!("map") };
        assert_eq!(args, &[ValueId(i - 1), ValueId(0)]);
    }
    check(program).unwrap();
    assert!(text::parse(&(source + "%x0 = input f64 []\n"))
        .unwrap_err()
        .0
        .contains("defined twice"));
    assert!(text::parse("%a = map neg %later\n%later = input f64 []\noutput %a\n")
        .unwrap_err()
        .0
        .contains("not defined"));
}
