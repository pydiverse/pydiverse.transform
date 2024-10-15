from __future__ import annotations

from pydiverse.transform._internal.ops.core import ElementWise, NoExprMethod

__all__ = [
    "HMax",
    "HMin",
]


class Horizontal(ElementWise, NoExprMethod):
    arg_names = ["args"]


class HMax(Horizontal):
    name = "hmax"
    signatures = ["T... -> T"]


class HMin(Horizontal):
    name = "hmin"
    signatures = ["T... -> T"]
