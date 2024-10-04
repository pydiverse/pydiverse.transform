from __future__ import annotations

from pydiverse.transform._internal.ops.core import ElementWise, OperatorExtension, Unary
from pydiverse.transform._internal.ops.logical import Logical
from pydiverse.transform._internal.ops.numeric import Add, RAdd

__all__ = [
    "StrAdd",
    "StrRAdd",
    "StrStrip",
    "StrLen",
    "StrToUpper",
    "StrToLower",
    "StrReplaceAll",
    "StrStartsWith",
    "StrEndsWith",
    "StrContains",
    "StrSlice",
    "StrToDateTime",
    "StrToDate",
]


class StrAdd(OperatorExtension):
    operator = Add
    signatures = [
        "str, str -> str",
    ]


class StrRAdd(OperatorExtension):
    operator = RAdd
    signatures = [
        "str, str -> str",
    ]


####


class StrUnary(ElementWise, Unary):
    signatures = [
        "str -> str",
    ]


class StrStrip(StrUnary):
    name = "str.strip"


class StrLen(StrUnary):
    name = "str.len"
    signatures = [
        "str -> int64",
    ]


class StrToUpper(StrUnary):
    name = "str.to_upper"


class StrToLower(StrUnary):
    name = "str.to_lower"


class StrReplaceAll(ElementWise):
    name = "str.replace_all"
    signatures = [
        "str, const str, const str -> str",
    ]


class StrStartsWith(ElementWise, Logical):
    name = "str.starts_with"
    signatures = [
        "str, const str -> bool",
    ]


class StrEndsWith(ElementWise, Logical):
    name = "str.ends_with"
    signatures = [
        "str, const str -> bool",
    ]


class StrContains(ElementWise, Logical):
    name = "str.contains"
    signatures = [
        "str, const str -> bool",
    ]


class StrSlice(ElementWise):
    name = "str.slice"
    signatures = ["str, int64, int64 -> str"]


class StrToDateTime(ElementWise):
    name = "str.to_datetime"
    signatures = ["str -> datetime"]


class StrToDate(ElementWise):
    name = "str.to_date"
    signatures = ["str -> date"]
