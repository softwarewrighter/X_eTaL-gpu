//! Printing arrays as X_eTaL prints them, so a run of the IR can be
//! compared with the evaluator's output line for line.
//!
//! - A single value is its number; a Bool is 1 or 0.
//! - A vector is its items separated by one space.
//! - A matrix is one line per row, the columns right-aligned to the
//!   widest item in each (`1 20 300` over `4  5   6`).
//! - A higher rank is printed plane by plane: one blank line between
//!   the planes of a rank-3 array, two between the rank-3 blocks of a
//!   rank-4 one, and so on (as many as the rank of the block that
//!   ends there, less two).
//! - A Float prints in full decimal with the shortest digits that
//!   read back as the same number, always with a point: `3.0`,
//!   `0.00000015`, `1000000000000000000000.0`, `-0.0`; `inf`.

use crate::interp::{Array, Data};

/// A Float as X_eTaL prints it.
pub fn float(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    let s = format!("{x}");
    if s.contains('.') {
        s
    } else {
        s + ".0"
    }
}

fn item(a: &Array, i: usize) -> String {
    match &a.data {
        Data::Int(v) => v[i].to_string(),
        Data::Float(v) => float(v[i]),
        Data::Bool(v) => if v[i] { "1" } else { "0" }.to_string(),
    }
}

/// The array as X_eTaL prints it (no trailing newline).
pub fn show(a: &Array) -> String {
    let n = a.data.len();
    match a.shape.rank() {
        0 => item(a, 0),
        1 => (0..n).map(|i| item(a, i)).collect::<Vec<_>>().join(" "),
        _ => {
            let cols = *a.shape.0.last().unwrap();
            let rows = n.checked_div(cols).unwrap_or(0);
            let rows_per_plane = a.shape.0[a.shape.rank() - 2];
            let leading = &a.shape.0[..a.shape.rank() - 2];
            let cells: Vec<String> = (0..n).map(|i| item(a, i)).collect();
            let widths: Vec<usize> = (0..cols)
                .map(|c| (0..rows).map(|r| cells[r * cols + c].len()).max().unwrap_or(0))
                .collect();
            let mut lines = Vec::new();
            for r in 0..rows {
                if r > 0 && rows_per_plane > 0 && r % rows_per_plane == 0 {
                    // The outermost leading axis that changes at this
                    // plane decides how many blank lines: one for the
                    // innermost, one more for each axis outside it.
                    let p = r / rows_per_plane;
                    let (prev, this) = (digits(p - 1, leading), digits(p, leading));
                    let j = (0..leading.len()).find(|&j| prev[j] != this[j]).unwrap_or(0);
                    for _ in j..leading.len() {
                        lines.push(String::new());
                    }
                }
                let line: Vec<String> = (0..cols).map(|c| format!("{:>w$}", cells[r * cols + c], w = widths[c])).collect();
                lines.push(line.join(" "));
            }
            lines.join("\n")
        }
    }
}

/// The multi-index of plane p over the leading axes (row order).
fn digits(p: usize, dims: &[usize]) -> Vec<usize> {
    let mut d = vec![0; dims.len()];
    let mut r = p;
    for (i, &n) in dims.iter().enumerate().rev() {
        d[i] = r % n.max(1);
        r /= n.max(1);
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Shape;

    #[test]
    fn floats_as_xetal_prints_them() {
        assert_eq!(float(3.0), "3.0");
        assert_eq!(float(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(float(1e21), "1000000000000000000000.0");
        assert_eq!(float(1.5e-7), "0.00000015");
        assert_eq!(float(-0.0), "-0.0");
        assert_eq!(float(f64::INFINITY), "inf");
        assert_eq!(float(1.0 / 3.0), "0.3333333333333333");
    }

    #[test]
    fn vectors_and_scalars() {
        assert_eq!(show(&Array::ints(Shape(vec![3]), vec![1, 20, 300])), "1 20 300");
        assert_eq!(show(&Array::ints(Shape::scalar(), vec![7])), "7");
        assert_eq!(show(&Array::floats(Shape(vec![2]), vec![1.5, -2.0])), "1.5 -2.0");
    }

    #[test]
    fn matrices_align_columns() {
        assert_eq!(
            show(&Array::ints(Shape(vec![2, 3]), vec![1, 20, 300, 4, 5, 6])),
            "1 20 300\n4  5   6"
        );
        assert_eq!(
            show(&Array::floats(Shape(vec![2, 2]), vec![1.5, -2.0, 3.25, 4.0])),
            " 1.5 -2.0\n3.25  4.0"
        );
        assert_eq!(show(&Array::ints(Shape(vec![2, 1, 2]), vec![1, 2, 3, 4])), "1 2\n\n3 4");
    }

    #[test]
    fn higher_ranks_as_xetal_prints_them() {
        // 2 2 2 2 r_eshape r_ange 16, 2 1 2 2 r_eshape r_ange 8, 2 2 1 1 2 r_eshape r_ange 8: xetal's blank lines.
        assert_eq!(
            show(&Array::ints(Shape(vec![2, 2, 2, 2]), (1..=16).collect())),
            " 1  2\n 3  4\n\n 5  6\n 7  8\n\n\n 9 10\n11 12\n\n13 14\n15 16"
        );
        assert_eq!(
            show(&Array::ints(Shape(vec![2, 1, 2, 2]), (1..=8).collect())),
            "1 2\n3 4\n\n\n5 6\n7 8"
        );
        assert_eq!(
            show(&Array::ints(Shape(vec![2, 2, 1, 1, 2]), (1..=8).collect())),
            "1 2\n\n\n3 4\n\n\n\n5 6\n\n\n7 8"
        );
    }
}
