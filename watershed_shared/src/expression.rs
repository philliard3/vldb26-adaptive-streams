use crate::{HabString, HabValue, Tuple};
use log::error;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

pub enum ComputationExpression {
    Field(HabString),
    Literal(ComputationLiteral),
    UnaryOp {
        op: UnaryOp,
        expr: Box<ComputationExpression>,
    },
    BinaryOp {
        op: BinaryOp,
        left: Box<ComputationExpression>,
        right: Box<ComputationExpression>,
    },
    TernaryOp {
        cond: Box<ComputationExpression>,
        left: Box<ComputationExpression>,
        right: Box<ComputationExpression>,
    },
    UserDefinedFunction {
        name: HabString,
        args: Vec<ComputationExpression>,
        action: Box<dyn Send + Sync + Fn(Vec<&HabValue>) -> HabValue>,
    },
}

impl std::fmt::Debug for ComputationExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputationExpression::Field(name) => write!(f, "Field({})", name),
            ComputationExpression::Literal(lit) => write!(f, "Literal({:?})", lit),
            ComputationExpression::BinaryOp { op, left, right } => {
                write!(f, "BinaryOp({:?}, {:?}, {:?})", op, left, right)
            }
            ComputationExpression::UnaryOp { op, expr } => {
                write!(f, "UnaryOp({:?}, {:?})", op, expr)
            }
            ComputationExpression::TernaryOp { cond, left, right } => {
                write!(f, "TernaryOp({:?}, {:?}, {:?})", cond, left, right)
            }
            ComputationExpression::UserDefinedFunction {
                name,
                args,
                action: _,
            } => {
                write!(
                    f,
                    "UserDefinedFunction({:?}, {:?}, <user defined function>)",
                    name, args
                )
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", content = "value")]
pub enum ComputationLiteral {
    #[serde(rename = "int")]
    Int(i32),
    #[serde(rename = "float")]
    Float(f64),
    #[serde(rename = "string")]
    String(HabString),
    #[serde(rename = "bool")]
    Bool(bool),
    #[serde(rename = "null")]
    Null,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum BinaryOp {
    #[serde(rename = "+")]
    Add,
    #[serde(rename = "-")]
    Sub,
    #[serde(rename = "*")]
    Mul,
    #[serde(rename = "/")]
    Div,
    #[serde(rename = "%")]
    Mod,
    #[serde(rename = "=")]
    Eq,
    #[serde(rename = "!=")]
    Neq,
    #[serde(rename = ">")]
    Gt,
    #[serde(rename = ">=")]
    Gte,
    #[serde(rename = "<")]
    Lt,
    #[serde(rename = "<=")]
    Lte,
    #[serde(rename = "&")]
    And,
    #[serde(rename = "|")]
    Or,
    #[serde(rename = "^")]
    Xor,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum UnaryOp {
    #[serde(rename = "-")]
    Neg,
    #[serde(rename = "!")]
    Not,
}

pub fn evaluate_computation_expression(tuple: &Tuple, expr: &ComputationExpression) -> HabValue {
    match expr {
        ComputationExpression::Field(name) => tuple
            .get(&**name)
            .unwrap_or_else(|| {
                error!("Field {:?} not found in tuple {:?}", name, tuple);
                panic!("Field {:?} not found in tuple {:?}", name, tuple);
            })
            .clone(),
        ComputationExpression::Literal(lit) => match lit {
            ComputationLiteral::Int(i) => HabValue::Integer(*i),
            ComputationLiteral::Float(f) => HabValue::Float(OrderedFloat(*f)),
            ComputationLiteral::String(s) => HabValue::String(s.clone()),
            ComputationLiteral::Bool(b) => HabValue::Bool(*b),
            ComputationLiteral::Null => HabValue::Null,
        },
        ComputationExpression::BinaryOp { op, left, right } => {
            let left = evaluate_computation_expression(tuple, left);
            // TODO: this currently short circuits when the first argument produces a boolean, but will not catch the case where the second argument produces a different type
            // special case short circuiting for boolean operations
            match op {
                BinaryOp::And => match left {
                    HabValue::Bool(false) => {
                        return HabValue::Bool(false);
                    }
                    HabValue::Bool(_) => {}
                    _ => {
                        error!("Type Error in expression {:?}\nBinaryOp And not supported for lefthand type {:?}", expr, left.get_type());
                        panic!("Type Error in expression {:?}\nBinaryOp And not supported for lefthand type {:?}", expr, left.get_type());
                    }
                },
                BinaryOp::Or => match left {
                    HabValue::Bool(true) => {
                        return HabValue::Bool(true);
                    }
                    HabValue::Bool(_) => {}
                    _ => {
                        error!("Type Error in expression {:?}\nBinaryOp Or not supported for lefthand type {:?}", expr, left.get_type());
                        panic!("Type Error in expression {:?}\nBinaryOp Or not supported for lefthand type {:?}", expr, left.get_type());
                    }
                },
                _ => {} // do nothing for non-short circuiting operations
            }
            let right = evaluate_computation_expression(tuple, right);
            match (left, right) {
                (HabValue::Integer(l), HabValue::Integer(r)) => match op {
                    BinaryOp::Add => HabValue::Integer(l + r),
                    BinaryOp::Sub => HabValue::Integer(l - r),
                    BinaryOp::Mul => HabValue::Integer(l * r),
                    BinaryOp::Div => HabValue::Integer(l / r),
                    BinaryOp::Mod => HabValue::Integer(l % r),
                    BinaryOp::Eq => HabValue::Bool(l == r),
                    BinaryOp::Neq => HabValue::Bool(l != r),
                    BinaryOp::Gt => HabValue::Bool(l > r),
                    BinaryOp::Gte => HabValue::Bool(l >= r),
                    BinaryOp::Lt => HabValue::Bool(l < r),
                    BinaryOp::Lte => HabValue::Bool(l <= r),
                    _ => unimplemented!("BinaryOp {:?} not implemented for integer", op),
                },
                (HabValue::Float(l), HabValue::Float(r)) => match op {
                    BinaryOp::Add => HabValue::Float(l + r),
                    BinaryOp::Sub => HabValue::Float(l - r),
                    BinaryOp::Mul => HabValue::Float(l * r),
                    BinaryOp::Div => HabValue::Float(l / r),
                    BinaryOp::Mod => HabValue::Float(l % r),
                    BinaryOp::Eq => HabValue::Bool(l == r),
                    BinaryOp::Neq => HabValue::Bool(l != r),
                    BinaryOp::Gt => HabValue::Bool(l > r),
                    BinaryOp::Gte => HabValue::Bool(l >= r),
                    BinaryOp::Lt => HabValue::Bool(l < r),
                    BinaryOp::Lte => HabValue::Bool(l <= r),
                    BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => {
                        unimplemented!("BinaryOp {:?} not implemented for float", op)
                    }
                },
                (HabValue::Bool(l), HabValue::Bool(r)) => match op {
                    BinaryOp::Eq => HabValue::Bool(l == r),
                    BinaryOp::Neq => HabValue::Bool(l != r),
                    BinaryOp::And => HabValue::Bool(l && r),
                    BinaryOp::Or => HabValue::Bool(l || r),
                    BinaryOp::Xor => HabValue::Bool(l ^ r),
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Mod
                    | BinaryOp::Gt
                    | BinaryOp::Gte
                    | BinaryOp::Lt
                    | BinaryOp::Lte => unimplemented!("BinaryOp {:?} not implemented for bool", op),
                },
                (left, right) => unimplemented!(
                    "Type Error in expression {:?}\nBinaryOp not supported for input types {:?}",
                    expr,
                    (left.get_type(), right.get_type())
                ),
            }
        }
        ComputationExpression::UnaryOp { op, expr } => {
            let expr = evaluate_computation_expression(tuple, expr);
            match (op, expr) {
                (UnaryOp::Neg, HabValue::Integer(i)) => HabValue::Integer(-i),
                (UnaryOp::Neg, HabValue::Float(f)) => HabValue::Float(-f),
                (UnaryOp::Not, HabValue::Bool(b)) => HabValue::Bool(!b),
                (op, expr) => unimplemented!(
                    "Type Error in expression {:?}\nUnaryOp {:?} not supported for input type {:?}",
                    expr,
                    op,
                    expr.get_type()
                ),
            }
        }
        ComputationExpression::TernaryOp { cond, left, right } => {
            let cond = evaluate_computation_expression(tuple, cond);
            match cond {
                HabValue::Bool(b) => {
                    if b {
                        evaluate_computation_expression(tuple, left)
                    } else {
                        evaluate_computation_expression(tuple, right)
                    }
                }
                _ => unreachable!(
                    "Type Error in expression {:?}\nTernaryOp predicate must be a boolean",
                    expr
                ),
            }
        }
        ComputationExpression::UserDefinedFunction {
            name: _,
            args,
            action,
        } => {
            let args: Vec<_> = args
                .iter()
                .map(|arg| evaluate_computation_expression(tuple, arg))
                .collect();
            let args = args.iter().collect();
            action(args)
        }
    }
}
