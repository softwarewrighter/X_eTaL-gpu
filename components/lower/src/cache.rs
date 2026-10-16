//! A cache of lowered programs: lowering a program asks the evaluator
//! for every statement the device cannot take, which costs an
//! evaluator run each, so a program is lowered once per content and
//! read back after. The key hashes everything the lowering read (the
//! program and its libraries as one text, the files `[]N_GET` read)
//! and the tool itself (its executable's size and modification time,
//! so a rebuild invalidates). The cache lives in `work/lower-cache/`
//! of the repository that holds the tool (or `XETAL_GPU_CACHE`);
//! `XETAL_GPU_NO_CACHE=1` turns it off.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use xetal_gpu_xir::ir::ValueId;
use xetal_gpu_xir::{binary, text};

use crate::{Lowered, Printed};

/// The cache directory, if caching is on and a place for it is known.
pub fn dir() -> Option<PathBuf> {
    if std::env::var_os("XETAL_GPU_NO_CACHE").is_some() {
        return None;
    }
    if let Some(d) = std::env::var_os("XETAL_GPU_CACHE") {
        return Some(PathBuf::from(d));
    }
    let exe = std::env::current_exe().ok()?;
    let mut d: Option<&Path> = exe.parent();
    while let Some(p) = d {
        if p.join("XETAL_COMMIT").exists() {
            return Some(p.join("work/lower-cache"));
        }
        d = p.parent();
    }
    None
}

/// The key of a lowering: what it read and the tool that read it.
pub fn key(combined_sources: &str, files_read: &[(PathBuf, String)]) -> String {
    let mut h = DefaultHasher::new();
    combined_sources.hash(&mut h);
    for (p, content) in files_read {
        p.hash(&mut h);
        content.hash(&mut h);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Ok(m) = std::fs::metadata(&exe) {
            m.len().hash(&mut h);
            if let Ok(t) = m.modified() {
                t.hash(&mut h);
            }
        }
    }
    format!("{:016x}", h.finish())
}

pub fn read(dir: &Path, key: &str) -> Option<Lowered> {
    let d = dir.join(key);
    let program = text::parse(&std::fs::read_to_string(d.join("program.xir")).ok()?).ok()?;
    let mut inputs = HashMap::new();
    for line in std::fs::read_to_string(d.join("inputs.txt")).ok()?.lines() {
        if line.is_empty() {
            continue;
        }
        inputs.insert(line.to_string(), binary::read(d.join(format!("{line}.xta"))).ok()?);
    }
    let mut printed = Vec::new();
    for line in std::fs::read_to_string(d.join("printed.txt")).ok()?.lines() {
        match line.split_once(' ') {
            Some(("D", id)) => printed.push(Printed::Device(ValueId(id.parse().ok()?))),
            Some(("H", t)) => printed.push(Printed::Host(unescape(t))),
            _ => return None,
        }
    }
    let mut lines = HashMap::new();
    for line in std::fs::read_to_string(d.join("lines.txt")).ok()?.lines() {
        if let Some((id, t)) = line.split_once(' ') {
            lines.insert(ValueId(id.parse().ok()?), unescape(t));
        }
    }
    let mut reasons = Vec::new();
    for line in std::fs::read_to_string(d.join("reasons.txt")).ok()?.lines() {
        if let Some((w, r)) = line.split_once('\t') {
            reasons.push((unescape(w), unescape(r)));
        }
    }
    Some(Lowered {
        program,
        inputs,
        printed,
        lines,
        reasons,
    })
}

pub fn write(dir: &Path, key: &str, lowered: &Lowered) {
    let d = dir.join(key);
    let _ = std::fs::create_dir_all(&d);
    let _ = std::fs::write(d.join("program.xir"), text::print(&lowered.program));
    let mut names = String::new();
    for (name, a) in &lowered.inputs {
        if let Ok(bytes) = binary::encode(a) {
            if std::fs::write(d.join(format!("{name}.xta")), bytes).is_ok() {
                names += name;
                names += "\n";
            }
        }
    }
    let _ = std::fs::write(d.join("inputs.txt"), names);
    let printed: String = lowered
        .printed
        .iter()
        .map(|p| match p {
            Printed::Device(id) => format!("D {}\n", id.0),
            Printed::Host(t) => format!("H {}\n", escape(t)),
        })
        .collect();
    let _ = std::fs::write(d.join("printed.txt"), printed);
    let lines: String = lowered.lines.iter().map(|(id, t)| format!("{} {}\n", id.0, escape(t))).collect();
    let _ = std::fs::write(d.join("lines.txt"), lines);
    let reasons: String = lowered
        .reasons
        .iter()
        .map(|(w, r)| format!("{}\t{}\n", escape(w), escape(r)))
        .collect();
    let _ = std::fs::write(d.join("reasons.txt"), reasons);
}

fn escape(t: &str) -> String {
    t.replace('\\', "\\\\").replace('\n', "\\n").replace('\t', "\\t")
}

fn unescape(t: &str) -> String {
    let mut out = String::new();
    let mut chars = t.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(o) => out.push(o),
                None => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// A read file's content, for the key: unreadable counts as empty.
pub fn file_entry(path: &Path) -> (PathBuf, String) {
    (path.to_path_buf(), std::fs::read_to_string(path).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use xetal_gpu_xir::{Array, Shape};

    #[test]
    fn round_trips_a_lowering() {
        let dir = std::env::temp_dir().join(format!("xetal-gpu-cache-test-{}", std::process::id()));
        let program = text::parse("%x = input f64 [2]\n%y = map neg %x\noutput %y\n").unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), Array::floats(Shape(vec![2]), vec![1.5, -2.0]));
        let lowered = Lowered {
            program,
            inputs,
            printed: vec![Printed::Host("a\nb\n".into()), Printed::Device(ValueId(1))],
            lines: [(ValueId(1), "2: y := n_eg x".to_string())].into_iter().collect(),
            reasons: vec![("x".into(), "a reason\twith a tab".into())],
        };
        write(&dir, "k", &lowered);
        let back = read(&dir, "k").unwrap();
        assert_eq!(text::print(&back.program), text::print(&lowered.program));
        assert_eq!(back.inputs["x"], lowered.inputs["x"]);
        assert_eq!(back.printed, lowered.printed);
        assert_eq!(back.lines, lowered.lines);
        assert_eq!(back.reasons, lowered.reasons);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
