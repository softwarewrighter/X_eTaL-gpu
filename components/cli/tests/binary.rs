//! File conversion and binding through the actual command line.
use std::process::Command;

#[test]
fn convert_bind_and_refuse_existing_output_and_wrong_shape() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("xetal-pack-{}-{stamp}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    std::fs::write(dir.join("weights.txt"), "9007199254740993 -7\n").unwrap();
    std::fs::write(dir.join("model.xir"), "%a = input i64 [2]\noutput %a\n").unwrap();
    let call = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_xetal-gpu"))
            .current_dir(&dir)
            .args(args)
            .output()
            .unwrap()
    };
    assert!(call(&["pack", "i64", "2", "weights.txt", "weights.xta"]).status.success());
    let before = std::fs::read(dir.join("weights.xta")).unwrap();
    assert!(!call(&["pack", "i64", "2", "weights.txt", "weights.xta"]).status.success());
    assert_eq!(std::fs::read(dir.join("weights.xta")).unwrap(), before);
    let out = call(&["run", "model.xir", "--bind", "a=@weights.xta"]);
    assert!(out.status.success());
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "9007199254740993 -7\n");
    std::fs::write(dir.join("model.xir"), "%a = input i64 [1 2]\noutput %a\n").unwrap();
    let out = call(&["run", "model.xir", "--bind", "a=@weights.xta"]);
    assert!(!out.status.success());
    assert!(String::from_utf8(out.stderr).unwrap().contains("input requires i64 [1 2]"));
    std::fs::write(dir.join("weights.xta"), b"XTLARR01").unwrap();
    assert!(!call(&["run", "model.xir", "--bind", "a=@weights.xta"]).status.success());
    std::fs::remove_dir_all(dir).unwrap();
}
