//! Every demo program in libs/*/demos/*.xtl, lowered from its X_eTaL
//! source, has its generated kernel source and its plan pinned in
//! tests/expected/<name>.cl and .plan. XETAL_BLESS=1 rewrites them
//! (review the diff). The lowering needs the pinned X_eTaL (just
//! xetal) and finds the libraries on XETAL_PATH, set here.

use std::fs;
use std::path::{Path, PathBuf};

use xetal_gpu_opencl::{explain, plan, Schedule};
use xetal_gpu_xir::check;

fn demos(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for lib in fs::read_dir(root.join("libs")).expect("libs/").flatten() {
        if let Ok(entries) = fs::read_dir(lib.path().join("demos")) {
            for e in entries.flatten() {
                if e.path().extension().is_some_and(|x| x == "xtl") {
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
fn every_demo_has_its_kernels_and_plan_pinned() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let libs: Vec<String> = fs::read_dir(root.join("libs"))
        .unwrap()
        .flatten()
        .map(|e| e.path().join("src").to_string_lossy().into_owned())
        .collect();
    std::env::set_var("XETAL_PATH", libs.join(":"));
    let expected = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/expected");
    let demos = demos(&root);
    assert!(!demos.is_empty(), "no demos found");
    let mut failures = Vec::new();
    for demo in &demos {
        let name = demo.file_stem().unwrap().to_str().unwrap();
        let program = match xetal_gpu_lower::lower_file(demo) {
            Ok(p) => p,
            Err(e) => {
                failures.push(format!("{name}: {}", e.0));
                continue;
            }
        };
        let checked = check(program).unwrap();
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
