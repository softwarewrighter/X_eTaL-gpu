//! Host-side progression of bound scalar inputs across resident executions.
use std::{collections::HashMap, path::Path};
use xetal_gpu_xir::{format, Array, Data, Error, Result};

pub fn parse(value: &str) -> Result<(String, i64)> {
    let (name, step) = value.split_once('=').ok_or_else(|| Error("--advance requires NAME=STEP".into()))?;
    let step = step.parse().map_err(|_| Error("--advance step must be an Int".into()))?;
    Ok((name.into(), step))
}
pub fn validate(inputs: &HashMap<String, Array>, advances: &[(String, i64)]) -> Result<()> {
    let mut names = std::collections::HashSet::new();
    for (name, _) in advances {
        let a = inputs.get(name).ok_or_else(|| Error(format!("--advance {name}: no bound input")))?;
        if !names.insert(name) || !a.shape.is_scalar() || !matches!(a.data, Data::Int(_)) {
            return Err(Error("--advance requires distinct scalar Int inputs".into()));
        }
    }
    Ok(())
}
pub fn advance(inputs: &mut HashMap<String, Array>, advances: &[(String, i64)]) -> Result<()> {
    for (name, step) in advances {
        let a = inputs.get_mut(name).expect("validated batch input");
        let Data::Int(v) = &mut a.data else { unreachable!() };
        v[0] = v[0].checked_add(*step).ok_or_else(|| Error("--advance Int overflow".into()))?;
        if a.scalar == xetal_gpu_xir::Scalar::I32 && i32::try_from(v[0]).is_err() {
            return Err(Error("--advance i32 overflow".into()));
        }
    }
    Ok(())
}
pub fn save(dir: Option<&String>, repeat: usize, outputs: &[Array]) -> Result<()> {
    if let Some(dir) = dir {
        let path = Path::new(dir);
        std::fs::create_dir_all(path).map_err(|e| Error(e.to_string()))?;
        let text = outputs.iter().map(format::show).collect::<Vec<_>>().join("\n") + "\n";
        std::fs::write(path.join(format!("batch-{:04}.out", repeat - 1)), text).map_err(|e| Error(e.to_string()))?;
    }
    Ok(())
}
