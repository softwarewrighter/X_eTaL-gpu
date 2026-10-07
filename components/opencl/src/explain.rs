//! `xetal-gpu explain`: the plan in words, which values became which
//! kernels, with how many work-items, and what moves between host
//! and device.

use xetal_gpu_xir::Checked;

use crate::{Arg, Init, Plan};

/// The plan as text.
pub fn explain(checked: &Checked, plan: &Plan) -> String {
    let program = &checked.program;
    let name = |id| format!("%{}", program.value(id).name);
    let mut out = String::new();
    out.push_str(&format!("schedule: {}\n", plan.schedule));
    out.push_str("buffers:\n");
    for b in &plan.buffers {
        let what = match &b.init {
            Init::Input(n) => format!("input {n}, uploaded by the host"),
            Init::Const(_) => "constant, uploaded by the host".to_string(),
            Init::Device => "written on the device".to_string(),
        };
        let holds = match b.value {
            Some(v) => format!("{} {}", name(v), checked.ty(v)),
            None => "partials".to_string(),
        };
        out.push_str(&format!(
            "  {}: {} {} item{}, {}; {}\n",
            b.name,
            b.elem,
            b.len,
            if b.len == 1 { "" } else { "s" },
            holds,
            what
        ));
    }
    out.push_str("kernels:\n");
    for k in &plan.kernels {
        if k.elements > 0 {
            out.push_str(&format!(
                "  {}: {} element{}, one work-item each; computes {}; writes {}\n",
                k.name,
                k.elements,
                if k.elements == 1 { "" } else { "s" },
                k.computes.iter().map(|v| name(*v)).collect::<Vec<_>>().join(" "),
                if k.writes.is_empty() {
                    "nothing".to_string()
                } else {
                    k.writes.iter().map(|v| name(*v)).collect::<Vec<_>>().join(" ")
                }
            ));
        } else {
            out.push_str(&format!(
                "  {}: {}; computes {}\n",
                k.name,
                k.about,
                k.computes.iter().map(|v| name(*v)).collect::<Vec<_>>().join(" ")
            ));
        }
    }
    out.push_str("launches:\n");
    for (i, l) in plan.launches.iter().enumerate() {
        let args: Vec<String> = l
            .args
            .iter()
            .map(|a| match a {
                Arg::Buffer(b) => plan.buffer(*b).name.clone(),
                Arg::Long(n) => n.to_string(),
                Arg::Local(bytes) => format!("local {bytes} bytes"),
            })
            .collect();
        out.push_str(&format!(
            "  {}: {}({}) global {} local {}: {}\n",
            i + 1,
            l.kernel,
            args.join(", "),
            l.global,
            l.local,
            l.note
        ));
    }
    out.push_str("outputs:\n");
    for (v, b) in &plan.outputs {
        out.push_str(&format!("  {} from {}, read back by the host\n", name(*v), plan.buffer(*b).name));
    }
    out
}
