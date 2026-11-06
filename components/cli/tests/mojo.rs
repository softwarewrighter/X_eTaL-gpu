//! The CLI exposes Mojo discovery and device timing without changing stdout.
use std::process::Command;

#[test]
fn mojo_discovery_and_repeated_timing() {
    if xetal_gpu_mojo::host::mojo_path().is_none() {
        eprintln!("skipped: Mojo CLI device test (no toolchain)");
        return;
    }
    let binary = env!("CARGO_BIN_EXE_xetal-gpu");
    let found = Command::new(binary).arg("devices").output().unwrap();
    assert!(found.status.success(), "{}", String::from_utf8_lossy(&found.stderr));
    let devices = String::from_utf8(found.stdout).unwrap();
    assert!(devices.lines().any(|l| l.starts_with("mojo  api=")), "{devices}");
    let path = std::env::temp_dir().join(format!("xetal-mojo-cli-{}.xir", std::process::id()));
    std::fs::write(&path, "%a = input i64 [3]\n%k = const i64 [] 2\n%b = map mul %a %k\noutput %b\n").unwrap();
    let out = Command::new(binary)
        .args([
            "run",
            path.to_str().unwrap(),
            "--device",
            "mojo",
            "--bind",
            "a=1,2,3",
            "--repeat",
            "2",
            "--time",
        ])
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "2 4 6\n");
    let timing = String::from_utf8(out.stderr).unwrap();
    let lines: Vec<_> = timing.lines().filter(|l| l.starts_with("repeat ")).collect();
    assert_eq!(lines.len(), 2, "{timing}");
    for line in &lines {
        assert!(line.contains("device intervals"), "{line}");
        let kernel_ms: f64 = line
            .split("kernels ")
            .nth(1)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!(kernel_ms.is_finite() && kernel_ms > 0.0, "{line}");
    }
    assert!(lines[1].contains("build 0.000 ms"), "{timing}");
}
