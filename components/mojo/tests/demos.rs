//! Every demo in libs/*/demos/*.xtl, lowered from its X_eTaL source,
//! either has its generated Mojo source pinned in tests/expected/<name>.mojo
//! or is refused with a reason pinned in tests/expected/refused.txt
//! (XETAL_BLESS=1 rewrites them; review the diff). With the Mojo
//! toolchain present (scripts/mojo.sh, bin/mojo, or XETAL_GPU_MOJO) the
//! taken demos are built and run on this machine's GPU and must give the
//! evaluator's baseline (libs/<Name>/tests/demo-<name>.out) within 1e-5;
//! without it that part is skipped, loudly.

use std::fs;
use std::path::{Path, PathBuf};

use xetal_gpu_mojo::{emit, host};
use xetal_gpu_opencl::Schedule;
use xetal_gpu_xir::{check, format};

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

/// The outputs as xetal prints them against the baseline's text: Ints and
/// words exactly, Floats within a relative 1e-5 (the device computes in f32).
fn agrees(want: &str, got: &str) -> Result<(), String> {
    let w: Vec<&str> = want.lines().collect();
    let g: Vec<&str> = got.lines().collect();
    if w.len() != g.len() {
        return Err(format!("{} lines expected, {} given", w.len(), g.len()));
    }
    for (k, (a, b)) in w.iter().zip(&g).enumerate() {
        let (xs, ys): (Vec<&str>, Vec<&str>) = (a.split_whitespace().collect(), b.split_whitespace().collect());
        if xs.len() != ys.len() {
            return Err(format!("line {}: `{a}` against `{b}`", k + 1));
        }
        for (x, y) in xs.iter().zip(&ys) {
            if x == y {
                continue;
            }
            if x.parse::<i64>().is_ok() && y.parse::<i64>().is_ok() {
                return Err(format!("line {}: integer `{x}` differs from `{y}`", k + 1));
            }
            match (x.parse::<f64>(), y.parse::<f64>()) {
                (Ok(p), Ok(q)) if (p - q).abs() <= 1e-5 * p.abs().max(1.0) => {}
                _ => return Err(format!("line {}: `{a}` against `{b}`", k + 1)),
            }
        }
    }
    Ok(())
}

#[test]
fn every_demo_is_pinned_as_mojo_or_refused_by_name() {
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
    let mojo = host::mojo_path();
    match &mojo {
        Some(m) => eprintln!("mojo: {} (the taken demos run on this machine's GPU)", m.display()),
        None => eprintln!("skipped: the Mojo device runs (no toolchain: scripts/mojo.sh installs it)"),
    }
    let mut failures = Vec::new();
    let mut refused = String::new();
    let mut taken = 0;
    for demo in &demos {
        let name = demo.file_stem().unwrap().to_str().unwrap();
        let lowered = match xetal_gpu_lower::lower_file_full(demo) {
            Ok(l) => l,
            Err(e) => {
                failures.push(format!("{name}: lowering: {e}"));
                continue;
            }
        };
        let checked = check::check(lowered.program.clone()).unwrap();
        let (source, abi) = match emit(&checked, &Schedule::default(), &name.replace('-', "_")) {
            Ok(x) => x,
            Err(e) => {
                refused += &format!("{name}: {e}\n");
                continue;
            }
        };
        taken += 1;
        if let Err(e) = pin(&expected.join(format!("{name}.mojo")), &source) {
            failures.push(e);
        }
        let Some(mojo) = &mojo else { continue };
        let run = (|| -> Result<(), String> {
            let (lib, _) = host::build(mojo, &source, &name.replace('-', "_")).map_err(|e| e.0)?;
            let library = host::Library::load(&lib).map_err(|e| e.0)?;
            if library.init().map_err(|e| e.0)? == 0 {
                return Err("the build targets no GPU".into());
            }
            let (outputs, _) = library.run(&checked, &abi, &lowered.inputs).map_err(|e| e.0)?;
            let mut device = outputs.iter();
            let got: String = lowered
                .printed
                .iter()
                .map(|item| match item {
                    xetal_gpu_lower::Printed::Device(_) => format::show(device.next().expect("device output")) + "\n",
                    xetal_gpu_lower::Printed::Host(text) => text.clone(),
                })
                .collect();
            let lib_dir = demo.parent().unwrap().parent().unwrap();
            let want = fs::read_to_string(lib_dir.join(format!("tests/demo-{name}.out"))).map_err(|e| format!("baseline: {e}"))?;
            agrees(&want, &got)
        })();
        match run {
            Ok(()) => eprintln!("ok: {name} on the GPU through Mojo"),
            Err(e) => failures.push(format!("{name} through Mojo: {e}")),
        }
    }
    if let Err(e) = pin(&expected.join("refused.txt"), &refused) {
        failures.push(e);
    }
    assert!(taken > 0, "no demo was taken by the Mojo backend");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
