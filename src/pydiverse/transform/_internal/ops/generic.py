from __future__ import annotations

from pydiverse.transform._internal.ops.core import ElementWise, NoExprMethod

__all__ = [
    "HMax",
    "HMin",
]


class HMax(ElementWise, NoExprMethod):
    name = "hmax"
    signatures = ["T... -> T"]


class HMin(ElementWise, NoExprMethod):
    name = "hmin"
    signatures = ["T... -> T"]
