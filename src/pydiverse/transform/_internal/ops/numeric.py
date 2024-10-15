from __future__ import annotations

from pydiverse.transform._internal.ops.core import Binary, ElementWise, Unary

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
        "int64, int64 -> int64",
        "float64, float64 -> float64",
        "decimal, decimal -> decimal",
    ]


class Sub(ElementWise, Binary):
    name = "__sub__"
    signatures = [
        "int64, int64 -> int64",
        "float64, float64 -> float64",
        "decimal, decimal -> decimal",
    ]


class Mul(ElementWise, Binary):
    name = "__mul__"
    signatures = [
        "int64, int64 -> int64",
        "float64, float64 -> float64",
        "decimal, decimal -> decimal",
    ]


class TrueDiv(ElementWise, Binary):
    name = "__truediv__"
    signatures = [
        "int64, int64 -> float64",
        "float64, float64 -> float64",
        "decimal, decimal -> decimal",
    ]


class FloorDiv(ElementWise, Binary):
    name = "__floordiv__"
    signatures = [
        "int64, int64 -> int64",
    ]


class Pow(ElementWise, Binary):
    name = "__pow__"
    signatures = [
        "int64, int64 -> float64",
        "float64, float64 -> float64",
        "decimal, decimal -> decimal",
    ]


class Mod(ElementWise, Binary):
    name = "__mod__"
    signatures = [
        "int64, int64 -> int64",
    ]


class Neg(ElementWise, Unary):
    name = "__neg__"
    signatures = [
        "int64 -> int64",
        "float64 -> float64",
        "decimal -> decimal",
    ]


class Pos(ElementWise, Unary):
    name = "__pos__"
    signatures = [
        "int64 -> int64",
        "float64 -> float64",
        "decimal -> decimal",
    ]


class Abs(ElementWise, Unary):
    name = "__abs__"
    signatures = [
        "int64 -> int64",
        "float64 -> float64",
        "decimal -> decimal",
    ]


class Round(ElementWise):
    name = "__round__"
    signatures = [
        "float64, const int64 -> float64",
        "decimal, const int64 -> decimal",
        "int64, const int64 -> int64",
    ]
    arg_names = ["self", "decimals"]
    defaults = [..., 0]


class Floor(ElementWise, Unary):
    name = "floor"
    signatures = [
        "float64 -> float64",
        "decimal -> decimal",
    ]


class Ceil(Floor):
    name = "ceil"


class Log(ElementWise, Unary):
    name = "log"
    signatures = ["float64 -> float64"]


class Exp(Log):
    name = "exp"
    signatures = ["float64 -> float64"]


class IsInf(ElementWise, Unary):
    name = "is_inf"
    signatures = ["float64 -> bool"]


class IsNotInf(IsInf):
    name = "is_not_inf"


class IsNan(ElementWise, Unary):
    name = "is_nan"
    signatures = ["float64 -> bool"]


class IsNotNan(IsNan):
    name = "is_not_nan"
