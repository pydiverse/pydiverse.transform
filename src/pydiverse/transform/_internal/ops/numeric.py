from __future__ import annotations

from pydiverse.transform._internal.ops.core import Binary, ElementWise, Unary

__all__ = [
    "Add",
    "RAdd",
    "Sub",
    "RSub",
    "Mul",
    "RMul",
    "TrueDiv",
    "RTrueDiv",
    "FloorDiv",
    "RFloorDiv",
    "Pow",
    "RPow",
    "Mod",
    "RMod",
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


class RAdd(Add):
    name = "__radd__"


class Sub(ElementWise, Binary):
    name = "__sub__"
    signatures = [
        "int64, int64 -> int64",
        "float64, float64 -> float64",
        "decimal, decimal -> decimal",
    ]


class RSub(Sub):
    name = "__rsub__"


class Mul(ElementWise, Binary):
    name = "__mul__"
    signatures = [
        "int64, int64 -> int64",
        "float64, float64 -> float64",
        "decimal, decimal -> decimal",
    ]


class RMul(Mul):
    name = "__rmul__"


class TrueDiv(ElementWise, Binary):
    name = "__truediv__"
    signatures = [
        "int64, int64 -> float64",
        "float64, float64 -> float64",
        "decimal, decimal -> decimal",
    ]


class RTrueDiv(TrueDiv):
    name = "__rtruediv__"


class FloorDiv(ElementWise, Binary):
    name = "__floordiv__"
    signatures = [
        "int64, int64 -> int64",
    ]


class RFloorDiv(FloorDiv):
    name = "__rfloordiv__"


class Pow(ElementWise, Binary):
    name = "__pow__"
    signatures = [
        "int64, int64 -> float64",
        "float64, float64 -> float64",
        "decimal, decimal -> decimal",
    ]


class RPow(Pow):
    name = "__rpow__"


class Mod(ElementWise, Binary):
    name = "__mod__"
    signatures = [
        "int64, int64 -> int64",
    ]


class RMod(Mod):
    name = "__rmod__"


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
        "int64 -> int64",
        "int64, const int64 -> int64",
        "float64 -> float64",
        "float64, const int64 -> float64",
        "decimal -> decimal",
        "decimal, const int64 -> decimal",
    ]


class Floor(ElementWise):
    name = "floor"
    signatures = [
        "float64 -> float64",
        "decimal -> decimal",
    ]


class Ceil(Floor):
    name = "ceil"


class Log(ElementWise):
    name = "log"
    signatures = ["float64 -> float64"]


class Exp(Log):
    name = "exp"
    signatures = ["float64 -> float64"]


class IsInf(ElementWise):
    name = "is_inf"
    signatures = ["float64 -> bool"]


class IsNotInf(IsInf):
    name = "is_not_inf"


class IsNan(ElementWise):
    name = "is_nan"
    signatures = ["float64 -> bool"]


class IsNotNan(IsNan):
    name = "is_not_nan"
