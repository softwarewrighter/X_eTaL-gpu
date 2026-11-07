//! Physical buffer reuse for ordered, write-all kernel plans.
use crate::{Arg, BufferId, Elem, Init, Plan};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

/// Logical buffers mapped to physical allocations. Inputs/constants stay resident;
/// outputs remain live through readback. Guard reads participate in lifetimes.
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    pub assignments: Vec<usize>,
    pub representatives: Vec<BufferId>,
    pub logical_bytes: usize,
    pub physical_bytes: usize,
}

/// Reuse equal-type buffers only after all preceding readers have finished.
/// Launches execute on an in-order queue; a guard precedes its indexed launch.
pub fn layout(plan: &Plan) -> MemoryLayout {
    let n = plan.buffers.len();
    let mut first = vec![usize::MAX; n];
    let mut last = vec![0; n];
    let mut touch = |id: BufferId, phase: usize| {
        first[id.0] = first[id.0].min(phase);
        last[id.0] = last[id.0].max(phase);
    };
    for (i, launch) in plan.launches.iter().enumerate() {
        for arg in &launch.args {
            if let Arg::Buffer(id) = arg {
                touch(*id, 2 * i + 1);
            }
        }
    }
    for guard in &plan.guard_checks {
        touch(guard.buffer, 2 * guard.before_launch);
    }
    for (_, id) in &plan.outputs {
        last[id.0] = usize::MAX;
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by_key(|i| (first[*i], *i));
    let mut assignments = vec![0; n];
    let mut representatives: Vec<BufferId> = Vec::new();
    let mut active = BinaryHeap::<Reverse<(usize, usize)>>::new();
    let mut free = HashMap::<Elem, BTreeMap<usize, Vec<usize>>>::new();
    for i in order {
        let b = &plan.buffers[i];
        while active.peek().is_some_and(|Reverse((end, _))| *end < first[i]) {
            let Reverse((_, slot)) = active.pop().unwrap();
            let prior = &plan.buffers[representatives[slot].0];
            free.entry(prior.elem).or_default().entry(prior.len.max(1)).or_default().push(slot);
        }
        let reusable = matches!(b.init, Init::Device) && first[i] != usize::MAX;
        let candidate = if reusable {
            let sizes = free.entry(b.elem).or_default();
            let size = sizes.range(b.len.max(1)..).next().map(|(size, _)| *size);
            size.map(|size| {
                let slots = sizes.get_mut(&size).unwrap();
                let slot = slots.pop().unwrap();
                if slots.is_empty() {
                    sizes.remove(&size);
                }
                slot
            })
        } else {
            None
        };
        let slot = candidate.unwrap_or_else(|| {
            representatives.push(b.id);
            representatives.len() - 1
        });
        assignments[i] = slot;
        if reusable {
            active.push(Reverse((last[i], slot)));
        }
    }
    let bytes = |id: BufferId| plan.buffers[id.0].len.max(1).saturating_mul(plan.buffers[id.0].elem.bytes());
    let logical_bytes = plan.buffers.iter().map(|b| bytes(b.id)).fold(0usize, usize::saturating_add);
    let physical_bytes = representatives.iter().map(|id| bytes(*id)).fold(0usize, usize::saturating_add);
    MemoryLayout {
        assignments,
        representatives,
        logical_bytes,
        physical_bytes,
    }
}
