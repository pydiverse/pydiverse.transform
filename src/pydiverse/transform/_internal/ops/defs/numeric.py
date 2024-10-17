from __future__ import annotations

from pydiverse.transform._internal.ops.op import Binary, Operator, Unary

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


class Add(Operator, Binary):
    name = "__add__"
    signatures = [
        "int, int -> int",
        "float, float -> float",
        "decimal, decimal -> decimal",
    ]


class Sub(Operator, Binary):
    name = "__sub__"
    signatures = [
        "int, int -> int",
        "float, float -> float",
        "decimal, decimal -> decimal",
    ]


class Mul(Operator, Binary):
    name = "__mul__"
    signatures = [
        "int, int -> int",
        "float, float -> float",
        "decimal, decimal -> decimal",
    ]


class TrueDiv(Operator, Binary):
    name = "__truediv__"
    signatures = [
        "int, int -> float",
        "float, float -> float",
        "decimal, decimal -> decimal",
    ]


class FloorDiv(Operator, Binary):
    name = "__floordiv__"
    signatures = [
        "int, int -> int",
    ]


class Pow(Operator, Binary):
    name = "__pow__"
    signatures = [
        "int, int -> float",
        "float, float -> float",
        "decimal, decimal -> decimal",
    ]


class Mod(Operator, Binary):
    name = "__mod__"
    signatures = [
        "int, int -> int",
    ]


class Neg(Operator, Unary):
    name = "__neg__"
    signatures = [
        "int -> int",
        "float -> float",
        "decimal -> decimal",
    ]


class Pos(Operator, Unary):
    name = "__pos__"
    signatures = [
        "int -> int",
        "float -> float",
        "decimal -> decimal",
    ]


class Abs(Operator, Unary):
    name = "__abs__"
    signatures = [
        "int -> int",
        "float -> float",
        "decimal -> decimal",
    ]


class Round(Operator):
    name = "__round__"
    signatures = [
        "float, const int -> float",
        "decimal, const int -> decimal",
        "int, const int -> int",
    ]
    arg_names = ["self", "decimals"]
    defaults = [..., 0]


class Floor(Operator, Unary):
    name = "floor"
    signatures = [
        "float -> float",
        "decimal -> decimal",
    ]


class Ceil(Floor):
    name = "ceil"


class Log(Operator, Unary):
    name = "log"
    signatures = ["float -> float"]


class Exp(Log):
    name = "exp"
    signatures = ["float -> float"]


class IsInf(Operator, Unary):
    name = "is_inf"
    signatures = ["float -> bool"]


class IsNotInf(IsInf):
    name = "is_not_inf"


class IsNan(Operator, Unary):
    name = "is_nan"
    signatures = ["float -> bool"]


class IsNotNan(IsNan):
    name = "is_not_nan"
