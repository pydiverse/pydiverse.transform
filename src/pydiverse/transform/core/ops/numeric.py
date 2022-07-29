from __future__ import annotations

from . import Binary, ElementWise, Unary

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
]


class Add(ElementWise, Binary):
    name = "__add__"
    signatures = [
        "int, int -> int",
        "int, float -> float",
        "float, int -> float",
        "float, float -> float",
    ]


class RAdd(Add):
    name = "__radd__"


class Sub(ElementWise, Binary):
    name = "__sub__"
    signatures = [
        "int, int -> int",
        "int, float -> float",
        "float, int -> float",
        "float, float -> float",
    ]


class RSub(Sub):
    name = "__rsub__"


class Mul(ElementWise, Binary):
    name = "__mul__"
    signatures = [
        "int, int -> int",
        "int, float -> float",
        "float, int -> float",
        "float, float -> float",
    ]


class RMul(Mul):
    name = "__rmul__"


class TrueDiv(ElementWise, Binary):
    name = "__truediv__"
    signatures = [
        "int, int -> float",
        "int, float -> float",
        "float, int -> float",
        "float, float -> float",
    ]


class RTrueDiv(TrueDiv):
    name = "__rtruediv__"


class FloorDiv(ElementWise, Binary):
    name = "__floordiv__"
    signatures = [
        "int, int -> int",
    ]


class RFloorDiv(FloorDiv):
    name = "__rfloordiv__"


class Pow(ElementWise, Binary):
    name = "__pow__"
    signatures = [
        "int, int -> int",
    ]


class RPow(Pow):
    name = "__rpow__"


class Mod(ElementWise, Binary):
    name = "__mod__"
    signatures = [
        "int, int -> int",
    ]


class RMod(Mod):
    name = "__rmod__"


class Neg(ElementWise, Unary):
    name = "__neg__"
    signatures = [
        "int -> int",
        "float -> float",
    ]


class Pos(ElementWise, Unary):
    name = "__pos__"
    signatures = [
        "int -> int",
        "float -> float",
    ]


class Abs(ElementWise, Unary):
    name = "__abs__"
    signatures = [
        "int -> int",
        "float -> float",
    ]


class Round(ElementWise):
    name = "__round__"
    signatures = [
        "int -> int",
        "int, int -> int",
        "float -> int",
        "float, int -> float",
    ]
