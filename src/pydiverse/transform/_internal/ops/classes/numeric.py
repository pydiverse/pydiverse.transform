from __future__ import annotations

from pydiverse.transform._internal.ops.classes.logical import Logical
from pydiverse.transform._internal.ops.operator import Binary, ElementWise, Unary
from pydiverse.transform._internal.ops.signature import Signature, const
from pydiverse.transform._internal.tree.dtypes import Bool, Decimal, Float64, Int64

__all__ = [
    "Add",
    "Sub",
    "Mul",
    "TrueDiv",
    "FloorDiv",
    "Pow",
    "Mod",
    "Neg",
    "Pos",
    "Abs",
    "Round",
    "Floor",
    "Ceil",
    "Exp",
    "Log",
    "IsInf",
    "IsNotInf",
    "IsNan",
    "IsNotNan",
]


class Add(ElementWise, Binary):
    name = "__add__"
    signatures = [
        Signature(Int64, Int64, returns=Int64),
        Signature(Float64, Float64, returns=Float64),
        Signature(Decimal, Decimal, returns=Decimal),
    ]
    has_rversion = True


class Sub(ElementWise, Binary):
    name = "__sub__"
    signatures = [
        Signature(Int64, Int64, returns=Int64),
        Signature(Float64, Float64, returns=Float64),
        Signature(Decimal, Decimal, returns=Decimal),
    ]
    has_rversion = True


class Mul(ElementWise, Binary):
    name = "__mul__"
    signatures = [
        Signature(Int64, Int64, returns=Int64),
        Signature(Float64, Float64, returns=Float64),
        Signature(Decimal, Decimal, returns=Decimal),
    ]
    has_rversion = True


class TrueDiv(ElementWise, Binary):
    name = "__truediv__"
    signatures = [
        Signature(Int64, Int64, returns=Float64),
        Signature(Float64, Float64, returns=Float64),
        Signature(Decimal, Decimal, returns=Decimal),
    ]
    has_rversion = True


class FloorDiv(ElementWise, Binary):
    name = "__floordiv__"
    signatures = [Signature(Int64, Int64, returns=Int64)]
    has_rversion = True


class Mod(ElementWise, Binary):
    name = "__mod__"
    signatures = [Signature(Int64, Int64, returns=Int64)]
    has_rversion = True


class Pow(ElementWise, Binary):
    name = "__pow__"
    signatures = [
        Signature(Int64, Int64, returns=Float64),
        Signature(Float64, Float64, returns=Float64),
        Signature(Decimal, Decimal, returns=Decimal),
    ]
    arg_names = ["self", "exponent"]
    has_rversion = True


class Neg(ElementWise, Unary):
    name = "__neg__"
    signatures = [
        Signature(Int64, returns=Int64),
        Signature(Float64, returns=Float64),
        Signature(Decimal, returns=Decimal),
    ]


class Pos(Neg):
    name = "__pos__"


class Abs(Pos):
    name = "__abs__"


class Round(ElementWise):
    name = "__round__"
    signatures = [
        Signature(Int64, const(Int64), returns=Int64),
        Signature(Float64, const(Int64), returns=Float64),
        Signature(Decimal, const(Int64), returns=Decimal),
    ]
    defaults = [..., 0]


class Floor(ElementWise):
    name = "floor"
    signatures = [
        Signature(Float64, returns=Float64),
        Signature(Decimal, returns=Decimal),
    ]


class Ceil(Floor):
    name = "ceil"


class Log(ElementWise):
    name = "log"
    signatures = [Signature(Float64, returns=Float64)]


class Exp(Log):
    name = "exp"


class IsInf(ElementWise, Logical, Unary):
    name = "is_inf"
    signatures = [Signature(Float64, returns=Bool)]


class IsNotInf(IsInf):
    name = "is_not_inf"


class IsNan(IsInf):
    name = "is_nan"


class IsNotNan(IsNan):
    name = "is_not_nan"
