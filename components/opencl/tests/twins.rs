//! Every twin in libs/*/demos/*.xir has its generated kernel source
//! and its plan pinned in tests/expected/<name>.cl and .plan.
//! XETAL_BLESS=1 rewrites them (review the diff).

use std::fs;
use std::path::{Path, PathBuf};

use xetal_gpu_opencl::{explain, plan, Schedule};
use xetal_gpu_xir::{check, text};

fn twins() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../libs");
    let mut found = Vec::new();
    for lib in fs::read_dir(&root).expect("libs/").flatten() {
        let demos = lib.path().join("demos");
        if let Ok(entries) = fs::read_dir(&demos) {
            for e in entries.flatten() {
                if e.path().extension().is_some_and(|x| x == "xir") {
                    found.push(e.path());
                }
            }
        }
    }
    found.sort();
    found
}

fn pin(path: &Path, got: &str) -> Result<(), String> {
    let bless = std::env::var("XETAL_BLESS").is_ok_and(|v| v == "1");
    match fs::read_to_string(path) {
        Ok(want) if want == got => Ok(()),
        Ok(_) | Err(_) if bless => {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, got).unwrap();
            Ok(())
        }
        Ok(want) => Err(format!(
            "{} differs from the generated text (XETAL_BLESS=1 to accept):\n--- expected\n{want}\n--- got\n{got}",
            path.display()
        )),
        Err(_) => Err(format!("{} is missing (XETAL_BLESS=1 to create)", path.display())),
    }
}

#[test]
fn every_twin_has_its_kernels_and_plan_pinned() {
    let expected = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/expected");
    let twins = twins();
    assert!(!twins.is_empty(), "no twins found");
    let mut failures = Vec::new();
    for twin in &twins {
        let name = twin.file_stem().unwrap().to_str().unwrap();
        let src = fs::read_to_string(twin).unwrap();
        let checked = check(text::parse(&src).unwrap()).unwrap();
        let p = plan(&checked, &Schedule::default());
        if let Err(e) = pin(&expected.join(format!("{name}.cl")), &p.source) {
            failures.push(e);
        }
        if let Err(e) = pin(&expected.join(format!("{name}.plan")), &explain(&checked, &p)) {
            failures.push(e);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
