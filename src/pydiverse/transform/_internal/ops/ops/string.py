from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Param, Signature
from pydiverse.transform._internal.tree.types import Bool, Date, Datetime, Int, String


class StrUnary(Operator):
    signatures = [Signature(String(), return_type=String())]


strip = StrUnary("str.strip")
upper = StrUnary("str.upper")
lower = StrUnary("str.lower")

len = Operator("str.len", [Signature(String(), return_type=Int())])

replace_all = Operator(
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

starts_with = Operator(
    "str.starts_with",
    [
        Signature(
            String(),
            Param(String(const=True), "prefix"),
            return_type=Bool(),
        )
    ],
)

ends_with = Operator(
    "str.ends_with",
    [
        Signature(
            String(),
            Param(String(const=True), "suffix"),
            return_type=Bool(),
        )
    ],
)


contains = Operator(
    "str.contains",
    [
        Signature(
            String(),
            Param(String(const=True), "substr"),
            return_type=Bool(),
        )
    ],
)

slice = Operator(
    "str.slice",
    [
        Signature(
            String(), Param(Int(), "offset"), Param(Int(), "n"), return_type=String()
        )
    ],
)

to_datetime = Operator("str.to_datetime", [Signature(String(), return_type=Datetime())])

to_date = Operator("str.to_date", [Signature(String(), return_type=Date())])
