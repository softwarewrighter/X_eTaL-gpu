//! The CLI must preserve outputs and advance bound inputs without rebuilding a session.
use std::process::Command;
#[test]
fn resident_repeats_advance_and_preserve_each_output() {
    let dir = std::env::temp_dir().join(format!("xetal-batches-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scalar.xir"), "%base = input i64 []\noutput %base\n").unwrap();
    let mut devices = vec!["cpu".to_string()];
    devices.extend(
        xetal_gpu_runtime::devices()
            .unwrap()
            .into_iter()
            .map(|d| format!("opencl:{}", d.index)),
    );
    for device in devices {
        let saved = dir.join(device.replace(':', "-"));
        let out = Command::new(env!("CARGO_BIN_EXE_xetal-gpu"))
            .args([
                "run",
                dir.join("scalar.xir").to_str().unwrap(),
                "--device",
                &device,
                "--bind",
                "base=3",
                "--repeat",
                "3",
                "--advance",
                "base=7",
                "--save-repeats",
                saved.to_str().unwrap(),
                "--time",
            ])
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(String::from_utf8(out.stdout).unwrap(), "17\n");
        for (i, value) in [3, 10, 17].iter().enumerate() {
            assert_eq!(
                std::fs::read_to_string(saved.join(format!("batch-{i:04}.out"))).unwrap(),
                format!("{value}\n")
            );
        }
        if device.starts_with("opencl:") {
            let times = String::from_utf8(out.stderr).unwrap();
            assert_eq!(times.lines().filter(|l| l.contains("setup 0.000 ms")).count(), 2);
        }
    }
    let bad = Command::new(env!("CARGO_BIN_EXE_xetal-gpu"))
        .args([
            "run",
            dir.join("scalar.xir").to_str().unwrap(),
            "--bind",
            "base=9223372036854775807",
            "--repeat",
            "2",
            "--advance",
            "base=1",
        ])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("overflow"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn lowering_a_function_preserves_a_runtime_frame_parameter() {
    let dir = std::env::temp_dir().join(format!("xetal-function-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("frame.xtl"), "{ start -> (r_ange 2) + start }\n").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_xetal-gpu"))
        .args([
            "lower-function",
            dir.join("frame.xtl").to_str().unwrap(),
            "--param",
            "start:i64",
            "--regions",
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("input i64 []"));
    std::fs::write(dir.join("frame.xir"), out.stdout).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_xetal-gpu"))
        .args(["run", dir.join("frame.xir").to_str().unwrap(), "--bind", "start=5"])
        .output()
        .unwrap();
    assert!(run.status.success());
    assert_eq!(String::from_utf8(run.stdout).unwrap(), "6 7\n");
    std::fs::remove_dir_all(dir).unwrap();
}
