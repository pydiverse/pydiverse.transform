from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Param, Signature
from pydiverse.transform._internal.tree.types import Bool, Date, Datetime, Int, String


class StrUnary(Operator):
    signatures = [Signature(String(), return_type=String())]


str_strip = StrUnary("str.strip")
str_upper = StrUnary("str.upper")
str_lower = StrUnary("str.lower")

str_len = Operator("str.len", [Signature(String(), return_type=Int())])

str_replace_all = Operator(
    "str.replace_all",
    [
        Signature(
            String(),
            Param(String(const=True), "substr"),
            Param(String(const=True), "replacement"),
            return_type=String(),
        )
    ],
)

str_starts_with = Operator(
    "str.starts_with",
    [
        Signature(
            String(),
            Param(String(const=True), "prefix"),
            return_type=Bool(),
        )
    ],
)

str_ends_with = Operator(
    "str.ends_with",
    [
        Signature(
            String(),
            Param(String(const=True), "suffix"),
            return_type=Bool(),
        )
    ],
)


str_contains = Operator(
    "str.contains",
    [
        Signature(
            String(),
            Param(String(const=True), "substr"),
            return_type=Bool(),
        )
    ],
)

str_slice = Operator(
    "str.slice",
    [
        Signature(
            String(), Param(Int(), "offset"), Param(Int(), "n"), return_type=String()
        )
    ],
)

str_to_datetime = Operator(
    "str.to_datetime", [Signature(String(), return_type=Datetime())]
)

str_to_date = Operator("str.to_date", [Signature(String(), return_type=Date())])
