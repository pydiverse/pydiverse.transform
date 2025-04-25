from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import (
    NUMERIC,
    Bool,
    Const,
    Decimal,
    Float,
    Int,
)

pow = Operator(
    "__pow__",
    Signature(Int(), Int(), return_type=Float()),
    Signature(Float(), Float(), return_type=Float()),
    Signature(Decimal(), Decimal(), return_type=Decimal()),
    doc="""
Computes the power x ** y.

Note
----
Polars throws on negative exponents in the integer case. A polars error like
`failed to convert X to u32` may be due to negative inputs to this function.
""",
)

neg = Operator(
    "__neg__",
    *(Signature(t, return_type=t) for t in NUMERIC),
    doc="The unary - (negation) operator (__neg__)",
)

pos = Operator(
    "__pos__",
    *(Signature(t, return_type=t) for t in NUMERIC),
    doc="The unary + operator (__pos__)",
)

abs = Operator(
    "abs",
    *(Signature(t, return_type=t) for t in NUMERIC),
    doc="Computes the absolute value.",
)

round = Operator(
    "round",
    *(Signature(t, Const(Int()), return_type=t) for t in NUMERIC),
    param_names=["self", "decimals"],
    default_values=[..., 0],
    doc="""
Rounds to a given number of decimals.

:param decimals:
    The number of decimals to round by.
""",
)

floor = Operator(
    "floor",
    Signature(Float(), return_type=Float()),
    Signature(Decimal(), return_type=Decimal()),
    doc="Returns the largest integer less than or equal to the input.",
)

ceil = Operator(
    "ceil",
    Signature(Float(), return_type=Float()),
    Signature(Decimal(), return_type=Decimal()),
    doc="Returns the smallest integer greater than or equal to the input.",
)

log = Operator(
    "log",
    Signature(Float(), return_type=Float()),
    doc="Computes the natural logarithm.",
)

exp = Operator(
    "exp",
    Signature(Float(), return_type=Float()),
    doc="Computes the exponential function.",
)

is_inf = Operator(
    "is_inf",
    Signature(Float(), return_type=Bool()),
    doc="""
Whether the number is infinite.

Note
----
This is currently only useful for backends supporting IEEE 754-floats. On
other backends it always returns False.
""",
)

is_not_inf = Operator("is_not_inf", Signature(Float(), return_type=Bool()))

is_nan = Operator("is_nan", Signature(Float(), return_type=Bool()))

is_not_nan = Operator("is_not_nan", Signature(Float(), return_type=Bool()))
