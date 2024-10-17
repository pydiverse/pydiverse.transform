from __future__ import annotations

from pydiverse.transform._internal.ops.op import NoExprMethod, Operator

__all__ = [
    "HMax",
    "HMin",
]


class Horizontal(Operator, NoExprMethod):
    arg_names = ["args"]


class HMax(Horizontal):
    name = "hmax"
    signatures = ["T... -> T"]


class HMin(Horizontal):
    name = "hmin"
    signatures = ["T... -> T"]
