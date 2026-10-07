//! The provisional accelerator IR of X_eTaL-gpu (XIR).
//!
//! A program is a list of values in SSA form, each a typed array
//! (a scalar type and a static shape) computed by one operation:
//! an input, a constant, an elementwise map over one or more
//! arrays (a rank-0 operand extends to every item, as in X_eTaL), a
//! select by a Bool mask, a reduction of a whole array to one
//! number, or a cast between scalar types. The outputs are the
//! values the program prints.
//!
//! The IR knows nothing of OpenCL or of X_eTaL syntax. It is written
//! to `docs/research7.txt` and to ask G1 in `docs/xetal-asks.md`, as
//! a stand-in for the accelerator IR X_eTaL will own; the text form
//! ([`text`]) is what a hand-lowered twin of an `.xtl` program is
//! written in, the checker ([`check`]) gives every value its type
//! and shape, and the interpreter ([`interp`]) is the reference
//! the GPU backends are compared with, computing as X_eTaL's
//! evaluator does.

pub mod check;
pub mod format;
pub mod interp;
pub mod ir;
pub mod text;

pub use check::{check, Checked};
pub use interp::{run, Array, Data};
pub use ir::{CastOp, Const, MapOp, Op, Program, ReduceOp, Scalar, Shape, Ty, Value, ValueId};

/// An error from any phase: a message, meant to be printed as is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}
