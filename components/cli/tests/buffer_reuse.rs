//! Reused physical storage must survive repeated resident GPU runs and rebinding.
use std::collections::HashMap;
use xetal_gpu_opencl::{memory::layout, plan, Schedule};
use xetal_gpu_xir::{check, interp, text, Array, Shape};

#[test]
fn transient_storage_reuse_preserves_outputs_across_resident_runs() {
    let checked = check(text::parse("%x = input f64 [32]\n%a = map neg %x\n%b = reduce add %a\n%c = map add %x %b\n%d = reduce add %c\n%o = map add %x %d\noutput %o\noutput %c\n").unwrap()).unwrap();
    let p = plan(&checked, &Schedule::default());
    let m = layout(&p);
    assert!(m.representatives.len() < p.buffers.len());
    let inputs = |value| HashMap::from([("x".into(), Array::floats(Shape(vec![32]), vec![value; 32]))]);
    let devices = xetal_gpu_runtime::devices().unwrap();
    if devices.is_empty() {
        eprintln!("skipped: transient storage reuse on OpenCL (no device)");
    }
    for d in devices {
        let mut device = xetal_gpu_runtime::Prepared::new(&checked, &p, &inputs(1.), d.index).unwrap().0;
        for value in [1., 2., -3., 1.] {
            let bound = inputs(value);
            device.rebind("x", &bound["x"]).unwrap();
            let expected = interp::run(&checked, &bound).unwrap();
            for _ in 0..2 {
                assert_eq!(device.run_timed().unwrap().0, expected);
            }
        }
    }
}
