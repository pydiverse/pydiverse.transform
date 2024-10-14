from __future__ import annotations

from pydiverse.transform._internal.ops.classes.logical import Logical
from pydiverse.transform._internal.ops.operator import ElementWise, Unary
from pydiverse.transform._internal.ops.signature import Signature, const
from pydiverse.transform._internal.tree.dtypes import Date, Datetime, Int64, String

__all__ = [
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


class StrUnary(ElementWise, Unary):
    signatures = [Signature(String, returns=String)]


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
    signatures = [Signature(String, const(String), const(String), returns=String)]


class StrStartsWith(ElementWise, Logical):
    name = "str.starts_with"
    signatures = [Signature(String, const(String), returns=String)]


class StrEndsWith(StrStartsWith):
    name = "str.ends_with"


class StrContains(StrStartsWith):
    name = "str.contains"


class StrSlice(ElementWise):
    name = "str.slice"
    signatures = [Signature(String, Int64, Int64, returns=String)]


class StrToDateTime(ElementWise):
    name = "str.to_datetime"
    signatures = [Signature(String, returns=Datetime)]


class StrToDate(ElementWise):
    name = "str.to_date"
    signatures = [Signature(String, returns=Date)]
