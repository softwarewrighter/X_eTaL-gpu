//! Portable array files: versioned header, shape and little-endian items.
//! This is host storage, independent of the device's schedule.

use crate::{Array, Data, Error, Result, Scalar, Shape, Ty};

pub const MAGIC: &[u8; 8] = b"XTLARR01";

/// Encode f32, f64 or i64, preserving the shape and row order.
pub fn encode(a: &Array) -> Result<Vec<u8>> {
    let tag = match a.scalar {
        Scalar::F32 => 1,
        Scalar::F64 => 2,
        Scalar::I64 => 3,
        _ => return Err(Error("binary arrays support f32, f64 and i64".into())),
    };
    if a.shape.rank() > 9 || a.shape.0.iter().try_fold(1usize, |n, d| n.checked_mul(*d)).is_none() {
        return Err(Error("binary array shape exceeds rank or size limits".into()));
    }
    // Validate public Array fields before encoding them.
    Array::new(a.scalar, a.shape.clone(), a.data.clone())?;
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&[tag, a.shape.rank() as u8, 0, 0, 0, 0, 0, 0]);
    for d in &a.shape.0 {
        bytes.extend_from_slice(&(*d as u64).to_le_bytes());
    }
    match &a.data {
        Data::Float(v) if tag == 1 => v.iter().for_each(|x| bytes.extend_from_slice(&(*x as f32).to_le_bytes())),
        Data::Float(v) => v.iter().for_each(|x| bytes.extend_from_slice(&x.to_le_bytes())),
        Data::Int(v) => v.iter().for_each(|x| bytes.extend_from_slice(&x.to_le_bytes())),
        Data::Bool(_) => unreachable!("validated scalar"),
    }
    Ok(bytes)
}

/// Reject malformed headers, overflowing shapes and any payload length mismatch
/// before allocating the decoded items. Scalars have rank zero and one item.
pub fn decode(bytes: &[u8]) -> Result<Array> {
    let bad = |s: &str| Error(format!("binary array: {s}"));
    if bytes.len() < 16 || !bytes.starts_with(MAGIC) {
        return Err(bad("missing XTLARR01 header"));
    }
    let (scalar, width) = match bytes[8] {
        1 => (Scalar::F32, 4usize),
        2 => (Scalar::F64, 8),
        3 => (Scalar::I64, 8),
        _ => return Err(bad("unknown scalar tag")),
    };
    let rank = usize::from(bytes[9]);
    if rank > 9 || bytes[10..16].iter().any(|b| *b != 0) {
        return Err(bad("invalid rank or reserved bytes"));
    }
    let header = 16 + rank * 8;
    if bytes.len() < header {
        return Err(bad("truncated shape"));
    }
    let mut shape = Vec::with_capacity(rank);
    for chunk in bytes[16..header].as_chunks::<8>().0 {
        let d = u64::from_le_bytes(*chunk);
        shape.push(usize::try_from(d).map_err(|_| bad("dimension exceeds host size"))?);
    }
    let count = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| bad("shape overflows"))?;
    let size = count
        .checked_mul(width)
        .and_then(|n| n.checked_add(header))
        .ok_or_else(|| bad("payload size overflows"))?;
    if bytes.len() != size {
        return Err(bad("payload length does not match shape and scalar width"));
    }
    let payload = &bytes[header..];
    let data = match scalar {
        Scalar::F32 => Data::Float(
            payload
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f64::from(f32::from_le_bytes(*c)))
                .collect(),
        ),
        Scalar::F64 => Data::Float(payload.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect()),
        Scalar::I64 => Data::Int(payload.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect()),
        _ => unreachable!(),
    };
    Array::new(scalar, Shape(shape), data)
}

/// Bind a binary file's array to a declared input. f32 storage can widen to
/// f64; all other types and every dimension must match exactly.
pub fn for_input(a: Array, ty: &Ty) -> Result<Array> {
    if a.shape != ty.shape || (a.scalar != ty.scalar && !(a.scalar == Scalar::F32 && ty.scalar == Scalar::F64)) {
        return Err(Error(format!("binary array is {}, input requires {ty}", a.ty())));
    }
    Array::new(ty.scalar, a.shape, a.data)
}

pub fn read(path: impl AsRef<std::path::Path>) -> Result<Array> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    decode(&bytes).map_err(|e| Error(format!("{}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_round_trips_and_widening() {
        for scalar in [Scalar::F32, Scalar::F64, Scalar::I64] {
            for shape in [Shape(vec![]), Shape(vec![2, 3]), Shape(vec![0, 3])] {
                let n = shape.len();
                let data = if scalar == Scalar::I64 {
                    Data::Int(vec![9_007_199_254_740_993; n])
                } else {
                    Data::Float(vec![-0.25; n])
                };
                let a = Array::new(scalar, shape, data).unwrap();
                assert_eq!(decode(&encode(&a).unwrap()).unwrap(), a);
            }
        }
        let a = Array::new(Scalar::F32, Shape(vec![2]), Data::Float(vec![0.1, -0.0])).unwrap();
        let bytes = encode(&a).unwrap();
        assert_eq!(&bytes[16..24], &2u64.to_le_bytes());
        assert_eq!(&bytes[24..28], &0.1f32.to_le_bytes());
        let ty = Ty {
            scalar: Scalar::F64,
            shape: Shape(vec![2]),
        };
        assert_eq!(for_input(decode(&bytes).unwrap(), &ty).unwrap().data, a.data);
        assert!(for_input(
            a,
            &Ty {
                scalar: Scalar::I64,
                shape: ty.shape
            }
        )
        .is_err());
    }

    #[test]
    fn corrupt_files_and_wrong_shapes_are_refused() {
        let good = encode(&Array::ints(Shape(vec![2]), vec![1, 2])).unwrap();
        for len in 0..good.len() {
            assert!(decode(&good[..len]).is_err());
        }
        let mut extra = good.clone();
        extra.push(0);
        assert!(decode(&extra).is_err());
        for (index, value) in [(8, 99), (9, 10), (10, 1)] {
            let mut bad = good.clone();
            bad[index] = value;
            assert!(decode(&bad).is_err());
        }
        let mut huge = good.clone();
        huge[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(decode(&huge).is_err());
        let ty = Ty {
            scalar: Scalar::I64,
            shape: Shape(vec![1, 2]),
        };
        assert!(for_input(decode(&good).unwrap(), &ty).is_err());
    }
}
