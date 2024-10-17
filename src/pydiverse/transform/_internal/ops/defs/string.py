from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator, OperatorExtension, Unary
from pydiverse.transform._internal.ops.ops.logical import Logical
from pydiverse.transform._internal.ops.ops.numeric import Add

__all__ = [
    "StrAdd",
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


class StrUnary(Operator, Unary):
    signatures = [
        "str -> str",
    ]


class StrStrip(StrUnary):
    name = "str.strip"


class StrLen(StrUnary):
    name = "str.len"
    signatures = [
        "str -> int",
    ]


class StrToUpper(StrUnary):
    name = "str.to_upper"


class StrToLower(StrUnary):
    name = "str.to_lower"


class StrReplaceAll(Operator):
    name = "str.replace_all"
    signatures = [
        "str, const str, const str -> str",
    ]
    arg_names = ["self", "substr", "replacement"]


class StrStartsWith(Operator, Logical):
    name = "str.starts_with"
    signatures = [
        "str, const str -> bool",
    ]
    arg_names = ["self", "prefix"]


class StrEndsWith(Operator, Logical):
    name = "str.ends_with"
    signatures = [
        "str, const str -> bool",
    ]
    arg_names = ["self", "suffix"]


class StrContains(Operator, Logical):
    name = "str.contains"
    signatures = [
        "str, const str -> bool",
    ]
    arg_names = ["self", "substr"]


class StrSlice(Operator):
    name = "str.slice"
    signatures = ["str, int, int -> str"]
    arg_names = ["self", "offset", "n"]


class StrToDateTime(Operator, Unary):
    name = "str.to_datetime"
    signatures = ["str -> datetime"]


class StrToDate(Operator, Unary):
    name = "str.to_date"
    signatures = ["str -> date"]
