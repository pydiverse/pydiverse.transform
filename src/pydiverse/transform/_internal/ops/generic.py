from __future__ import annotations

from pydiverse.transform._internal.ops.core import ElementWise, NoExprMethod

__all__ = [
    "Greatest",
    "Least",
]


class Greatest(ElementWise, NoExprMethod):
    name = "__greatest"
    signatures = ["T... -> T"]


class Least(ElementWise, NoExprMethod):
    name = "__least"
    signatures = ["T... -> T"]
