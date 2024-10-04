from __future__ import annotations

from pydiverse.transform._internal.ops.core import ElementWise

__all__ = [
    "Greatest",
    "Least",
]


class Greatest(ElementWise):
    name = "__greatest"
    signatures = ["T... -> T"]


class Least(ElementWise):
    name = "__least"
    signatures = ["T... -> T"]
