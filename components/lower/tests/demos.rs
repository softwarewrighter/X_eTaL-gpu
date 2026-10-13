//! Every demo program lowers, and its lowered program, run by the
//! reference interpreter, prints what the evaluator printed (the
//! demo's reg-rs baseline): the hand twins' oracle, now the
//! translator's.

use std::fs;
use std::path::Path;

use xetal_gpu_lower::Printed;
use xetal_gpu_xir::{check, format, interp};

#[test]
fn every_demo_lowers_to_the_evaluators_output() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let libs: Vec<String> = fs::read_dir(root.join("libs"))
        .unwrap()
        .flatten()
        .map(|e| e.path().join("src").to_string_lossy().into_owned())
        .collect();
    std::env::set_var("XETAL_PATH", libs.join(":"));
    let mut failures = Vec::new();
    let mut n = 0;
    for lib in fs::read_dir(root.join("libs")).unwrap().flatten() {
        let demos = lib.path().join("demos");
        let Ok(entries) = fs::read_dir(&demos) else { continue };
        let mut paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "xtl"))
            .collect();
        paths.sort();
        for xtl in paths {
            n += 1;
            let stem = xtl.file_stem().unwrap().to_string_lossy().into_owned();
            let want_path = lib.path().join("tests").join(format!("demo-{stem}.out"));
            let want = fs::read_to_string(&want_path).unwrap_or_default();
            let got = (|| -> Result<String, String> {
                let lowered = xetal_gpu_lower::lower_file_full(&xtl).map_err(|e| e.0)?;
                let checked = check(lowered.program).map_err(|e| e.0)?;
                let outputs = interp::run(&checked, &lowered.inputs).map_err(|e| e.0)?;
                let mut device = outputs.iter();
                let mut text = String::new();
                for item in &lowered.printed {
                    match item {
                        Printed::Device(_) => text += &(format::show(device.next().unwrap()) + "\n"),
                        Printed::Host(t) => text += t,
                    }
                }
                Ok(text)
            })();
            match got {
                Ok(got) if got == want => {}
                Ok(got) => failures.push(format!(
                    "{stem}: differs from the evaluator\n--- evaluator\n{want}--- lowered\n{got}"
                )),
                Err(e) => failures.push(format!("{stem}: {e}")),
            }
        }
    }
    assert!(n >= 12, "demos found: {n}");
    assert!(
        failures.is_empty(),
        "{} of {n} demos fail:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}
