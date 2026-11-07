//! Resource inspection builds a plan without requiring an OpenCL device.
use std::{collections::HashMap, process::Command};

#[test]
fn reports_physical_payload_reuse_without_running_the_program() {
    let path = std::env::temp_dir().join(format!("xetal-memory-{}.xir", std::process::id()));
    std::fs::write(
        &path,
        "%x = input f64 [32]\n%a = map neg %x\n%b = reduce add %a\n%c = map add %x %b\n%d = reduce add %c\n%o = map add %x %d\noutput %o\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xetal-gpu"))
        .arg("memory")
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let output = String::from_utf8(result.stdout).unwrap();
    let metrics: HashMap<_, _> = output
        .lines()
        .map(|line| {
            let (name, value) = line.split_once(' ').unwrap();
            (name, value.parse::<usize>().unwrap())
        })
        .collect();
    assert!(metrics["physical_buffers"] < metrics["logical_buffers"]);
    assert!(metrics["physical_bytes"] < metrics["logical_bytes"]);
    assert!(metrics["launches"] > 0 && metrics["kernel_source_bytes"] > 0);
    assert_eq!(metrics["guards"], 0);
}
