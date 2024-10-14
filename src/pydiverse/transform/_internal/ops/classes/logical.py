from __future__ import annotations

from pydiverse.transform._internal.ops.operator import (
    Binary,
    ElementWise,
    Operator,
    Unary,
)
from pydiverse.transform._internal.ops.signature import Signature, T
from pydiverse.transform._internal.tree.dtypes import (
    Bool,
    Date,
    Datetime,
    Float64,
    Int64,
    String,
)

__all__ = [
    "Equal",
    "NotEqual",
    "Less",
    "LessEqual",
    "Greater",
    "GreaterEqual",
    "IsNull",
    "IsNotNull",
    "FillNull",
    "IsIn",
    "And",
    "Or",
    "Xor",
    "Invert",
]


# operator that returns a real boolean (relevant for mssql)
class Logical(Operator): ...


class IsNull(ElementWise, Logical, Unary):
    name = "is_null"
    signatures = [Signature(T, returns=Bool)]


class IsNotNull(IsNull):
    name = "is_not_null"


class FillNull(ElementWise):
    name = "fill_null"
    signatures = [Signature(T, T, returns=T)]


class IsIn(ElementWise, Logical):
    name = "isin"
    signatures = [Signature(T, ..., returns=Bool)]


class And(ElementWise, Logical, Binary):
    name = "__and__"
    signatures = [Signature(Bool, Bool, returns=Bool)]
    has_rversion = True


class Or(And):
    name = "__or__"


class Xor(And):
    name = "__xor__"


class Invert(ElementWise, Logical):
    name = "__invert__"
    signatures = ["bool -> bool"]


class Equal(ElementWise, Logical, Binary):
    signatures = [Signature(T, T, returns=Bool)]
    name = "__eq__"


class NotEqual(Equal):
    name = "__ne__"


class Less(ElementWise, Logical, Binary):
    name = "__lt__"
    signatures = [
        Signature(Int64, Int64, returns=Bool),
        Signature(Float64, Float64, returns=Bool),
        Signature(String, String, returns=Bool),
        Signature(Datetime, Datetime, returns=Bool),
        Signature(Datetime, Date, returns=Bool),
        Signature(Date, Datetime, returns=Bool),
        Signature(Date, Date, returns=Bool),
    ]


class LessEqual(Less):
    name = "__le__"


class Greater(Less):
    name = "__gt__"


class GreaterEqual(Less):
    name = "__ge__"
