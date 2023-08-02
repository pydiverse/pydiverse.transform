from __future__ import annotations

from pydiverse.transform.ops.core import ElementWise, OperatorExtension, Unary
from pydiverse.transform.ops.logical import Logical
from pydiverse.transform.ops.numeric import Add, RAdd

__all__ = [
    "StringAdd",
    "StringRAdd",
    "Strip",
    "StringLength",
    "Upper",
    "Lower",
    "Replace",
    "StartsWith",
    "EndsWith",
    "Contains",
]


class StringAdd(OperatorExtension):
    operator = Add
    signatures = [
        "str, str -> str",
    ]


class StringRAdd(OperatorExtension):
    operator = RAdd
    signatures = [
        "str, str -> str",
    ]


####


class StringUnary(ElementWise, Unary):
    signatures = [
        "str -> str",
    ]


class Strip(StringUnary):
    name = "strip"


class StringLength(StringUnary):
    name = "len"


class Upper(StringUnary):
    name = "upper"


class Lower(StringUnary):
    name = "lower"


class Replace(ElementWise):
    name = "replace"
    signatures = [
        "str, const str, const str -> str",
    ]


class StartsWith(ElementWise, Logical):
    name = "startswith"
    signatures = [
        "str, const str -> bool",
    ]


class EndsWith(ElementWise, Logical):
    name = "endswith"
    signatures = [
        "str, const str -> bool",
    ]


class Contains(ElementWise, Logical):
    name = "contains"
    signatures = [
        "str, const str -> bool",
    ]
