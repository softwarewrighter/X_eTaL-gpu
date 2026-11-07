//! Lifetimes include same-launch readers, outputs, guards and resident data.
use xetal_gpu_opencl::{memory::layout, plan, GuardCheck, Schedule};
use xetal_gpu_xir::{check, text, Checked};

fn pipeline(extra_output: bool) -> Checked {
    let mut source =
        "%x = input f64 [32]\n%a = map neg %x\n%b = reduce add %a\n%c = map add %x %b\n%d = reduce add %c\n%o = map add %x %d\noutput %o\n"
            .to_string();
    if extra_output {
        source.push_str("output %c\n");
    }
    check(text::parse(&source).unwrap()).unwrap()
}

#[test]
fn reuse_follows_last_reader_and_preserves_outputs_and_inputs() {
    for extra in [false, true] {
        let checked = pipeline(extra);
        let p = plan(&checked, &Schedule::default());
        let m = layout(&p);
        let slot = |name| {
            let id = checked.program.find(name).unwrap();
            m.assignments[p.buffers.iter().position(|b| b.value == Some(id)).unwrap()]
        };
        assert_eq!(slot("a"), slot("c"));
        assert_eq!(slot("b"), slot("d"));
        assert_ne!(slot("x"), slot("a"));
        assert_ne!(slot("a"), slot("b"), "same-launch reader and writer cannot alias");
        if extra {
            assert_ne!(slot("c"), slot("o"));
        } else {
            assert_eq!(slot("c"), slot("o"));
        }
        assert!(m.physical_bytes < m.logical_bytes);
        for (i, physical) in m.assignments.iter().enumerate() {
            let representative = &p.buffers[m.representatives[*physical].0];
            assert_eq!(representative.elem, p.buffers[i].elem);
            assert!(representative.len.max(1) >= p.buffers[i].len.max(1));
        }
    }
}

#[test]
fn deferred_guard_reads_keep_the_condition_alive() {
    use xetal_gpu_opencl::{Arg, Buffer, BufferId, Elem, Init, Launch, Plan};
    let a = BufferId(0);
    let d = BufferId(1);
    let launch = |name: &str, id| Launch {
        kernel: name.into(),
        args: vec![Arg::Buffer(id)],
        global: 1,
        local: 1,
        note: String::new(),
    };
    let mut p = Plan {
        schedule: Schedule::default(),
        source: String::new(),
        kernels: vec![],
        outputs: vec![],
        position_checks: vec![],
        guard_checks: vec![],
        buffers: [a, d]
            .into_iter()
            .map(|id| Buffer {
                id,
                value: None,
                name: format!("b{}", id.0),
                elem: Elem::I32,
                len: 1,
                init: Init::Device,
            })
            .collect(),
        launches: vec![launch("write_a", a), launch("read_a", a), launch("write_d", d)],
    };
    assert_eq!(layout(&p).assignments[a.0], layout(&p).assignments[d.0]);
    p.guard_checks.push(GuardCheck {
        before_launch: p.launches.len(),
        buffer: a,
        message: "late check".into(),
        codes: vec![],
    });
    let m = layout(&p);
    assert_ne!(m.assignments[a.0], m.assignments[d.0]);
}
