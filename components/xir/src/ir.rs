//! The IR itself: scalar types, shapes, operations, values, programs.

use std::fmt;

/// The scalar type of an array's items. X_eTaL's Int is `I64`, its
/// Float `F64`, its Bool `Bool`; `I32` and `F32` are the narrower
/// types a device schedule may choose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scalar {
    I32,
    I64,
    F32,
    F64,
    Bool,
}

impl Scalar {
    pub fn parse(s: &str) -> Option<Scalar> {
        Some(match s {
            "i32" => Scalar::I32,
            "i64" => Scalar::I64,
            "f32" => Scalar::F32,
            "f64" => Scalar::F64,
            "bool" => Scalar::Bool,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Scalar::I32 => "i32",
            Scalar::I64 => "i64",
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
            Scalar::Bool => "bool",
        }
    }

    pub fn is_int(self) -> bool {
        matches!(self, Scalar::I32 | Scalar::I64)
    }

    pub fn is_float(self) -> bool {
        matches!(self, Scalar::F32 | Scalar::F64)
    }

    pub fn is_num(self) -> bool {
        self.is_int() || self.is_float()
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The length of each axis; `[]` is a single value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn scalar() -> Shape {
        Shape(Vec::new())
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// How many items an array of this shape holds.
    pub fn len(&self) -> usize {
        self.0.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, n) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{n}")?;
        }
        write!(f, "]")
    }
}

/// An array's type: its scalar type and its shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ty {
    pub scalar: Scalar,
    pub shape: Shape,
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.scalar, self.shape)
    }
}

/// An elementwise operation; `arity` says how many arrays it takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MapOp {
    Add,
    Sub,
    Mul,
    Div,
    IDiv,
    Mod,
    Min,
    Max,
    Neg,
    Abs,
    Exp,
    Log,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Not,
}

impl MapOp {
    pub const ALL: [MapOp; 21] = [
        MapOp::Add,
        MapOp::Sub,
        MapOp::Mul,
        MapOp::Div,
        MapOp::IDiv,
        MapOp::Mod,
        MapOp::Min,
        MapOp::Max,
        MapOp::Neg,
        MapOp::Abs,
        MapOp::Exp,
        MapOp::Log,
        MapOp::Eq,
        MapOp::Ne,
        MapOp::Lt,
        MapOp::Le,
        MapOp::Gt,
        MapOp::Ge,
        MapOp::And,
        MapOp::Or,
        MapOp::Not,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MapOp::Add => "add",
            MapOp::Sub => "sub",
            MapOp::Mul => "mul",
            MapOp::Div => "div",
            MapOp::IDiv => "idiv",
            MapOp::Mod => "mod",
            MapOp::Min => "min",
            MapOp::Max => "max",
            MapOp::Neg => "neg",
            MapOp::Abs => "abs",
            MapOp::Exp => "exp",
            MapOp::Log => "log",
            MapOp::Eq => "eq",
            MapOp::Ne => "ne",
            MapOp::Lt => "lt",
            MapOp::Le => "le",
            MapOp::Gt => "gt",
            MapOp::Ge => "ge",
            MapOp::And => "and",
            MapOp::Or => "or",
            MapOp::Not => "not",
        }
    }

    pub fn parse(s: &str) -> Option<MapOp> {
        MapOp::ALL.iter().copied().find(|op| op.name() == s)
    }

    pub fn arity(self) -> usize {
        match self {
            MapOp::Neg | MapOp::Abs | MapOp::Exp | MapOp::Log | MapOp::Not => 1,
            _ => 2,
        }
    }

    /// What the operation accepts and gives, for the checker.
    pub fn kind(self) -> MapKind {
        match self {
            MapOp::Add | MapOp::Sub | MapOp::Mul | MapOp::Min | MapOp::Max | MapOp::Neg | MapOp::Abs => MapKind::Num,
            MapOp::Div | MapOp::Exp | MapOp::Log => MapKind::Float,
            MapOp::IDiv | MapOp::Mod => MapKind::Int,
            MapOp::Eq | MapOp::Ne => MapKind::EqCompare,
            MapOp::Lt | MapOp::Le | MapOp::Gt | MapOp::Ge => MapKind::OrdCompare,
            MapOp::And | MapOp::Or | MapOp::Not => MapKind::Logic,
        }
    }
}

impl fmt::Display for MapOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The typing rule of a map: what scalars it takes and what it gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapKind {
    /// Any numeric scalar, all arguments alike; gives the same.
    Num,
    /// Float scalars only (X_eTaL's `/`, `e_xp`, `l_og` give Floats); gives the same.
    Float,
    /// Int scalars only (`d_iv`, `m_od`); gives the same.
    Int,
    /// Any scalar, both alike; gives Bool.
    EqCompare,
    /// Numeric scalars, both alike; gives Bool.
    OrdCompare,
    /// Bool only; gives Bool.
    Logic,
}

/// A reduction of a whole array to one number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Add,
    Mul,
    Min,
    Max,
}

impl ReduceOp {
    pub const ALL: [ReduceOp; 4] = [ReduceOp::Add, ReduceOp::Mul, ReduceOp::Min, ReduceOp::Max];

    pub fn name(self) -> &'static str {
        match self {
            ReduceOp::Add => "add",
            ReduceOp::Mul => "mul",
            ReduceOp::Min => "min",
            ReduceOp::Max => "max",
        }
    }

    pub fn parse(s: &str) -> Option<ReduceOp> {
        ReduceOp::ALL.iter().copied().find(|op| op.name() == s)
    }
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A cast between scalar types, keeping the shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CastOp {
    pub to: Scalar,
}

/// A constant's items, as written: Ints, Floats or Bools.
#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Bool(Vec<bool>),
}

impl Const {
    pub fn len(&self) -> usize {
        match self {
            Const::Int(v) => v.len(),
            Const::Float(v) => v.len(),
            Const::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A value's position in the program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub usize);

/// How a value is computed.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Bound by the host when the program runs.
    Input,
    /// Items written in the program.
    Const(Const),
    /// An elementwise operation over the arguments, a rank-0
    /// argument extending to every item.
    Map { op: MapOp, args: Vec<ValueId> },
    /// `cond ? a : b`, item by item (each may be rank 0).
    Select { cond: ValueId, a: ValueId, b: ValueId },
    /// A whole array folded to one number (`axis: None`), or folded
    /// along one axis (1 is the first), which that axis leaves.
    Reduce { op: ReduceOp, arg: ValueId, axis: Option<usize> },
    /// The items converted to another scalar type.
    Cast { op: CastOp, arg: ValueId },
    /// The inner product `a '+ '* i_nner b`: a's last axis with b's
    /// first, each sum folded from the right.
    Matmul { a: ValueId, b: ValueId },
}

impl Op {
    /// The values this operation reads.
    pub fn args(&self) -> Vec<ValueId> {
        match self {
            Op::Input | Op::Const(_) => Vec::new(),
            Op::Map { args, .. } => args.clone(),
            Op::Select { cond, a, b } => vec![*cond, *a, *b],
            Op::Reduce { arg, .. } | Op::Cast { arg, .. } => vec![*arg],
            Op::Matmul { a, b } => vec![*a, *b],
        }
    }
}

/// One value of a program: its name (as written, without the `%`),
/// the type it declares (inputs, constants and casts declare one;
/// the checker fills in the rest) and its operation.
#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    pub name: String,
    pub ty: Option<Ty>,
    pub op: Op,
}

/// A program: values in definition order (an operation's arguments
/// come before it) and the values it outputs, in order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Program {
    pub values: Vec<Value>,
    pub outputs: Vec<ValueId>,
}

impl Program {
    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id.0]
    }

    pub fn find(&self, name: &str) -> Option<ValueId> {
        self.values.iter().position(|v| v.name == name).map(ValueId)
    }

    /// The inputs, in order.
    pub fn inputs(&self) -> Vec<ValueId> {
        (0..self.values.len())
            .map(ValueId)
            .filter(|id| matches!(self.value(*id).op, Op::Input))
            .collect()
    }
}
