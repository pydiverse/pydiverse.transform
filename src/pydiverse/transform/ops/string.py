from __future__ import annotations

from pydiverse.transform.ops.core import ElementWise, OperatorExtension, Unary
from pydiverse.transform.ops.numeric import Add, RAdd

__all__ = [
    "StringAdd",
    "StringRAdd",
    "Strip",
]


class StringAdd(OperatorExtension):
    operator = Add
    signatures = ["str, str -> str"]


class StringRAdd(OperatorExtension):
    operator = RAdd
    signatures = ["str, str -> str"]


####


class StringUnary(ElementWise, Unary):
    signatures = [
        "str -> str",
    ]


class Strip(StringUnary):
    name = "strip"
