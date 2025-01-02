from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import Bool, Date, Datetime, Int, String


class StrUnary(Operator):
    def __init__(self, name: str, doc: str = ""):
        super().__init__(name, Signature(String(), return_type=String()), doc=doc)


str_strip = StrUnary("str.strip")
str_upper = StrUnary("str.upper")
str_lower = StrUnary("str.lower")

str_len = Operator("str.len", Signature(String(), return_type=Int()))

str_replace_all = Operator(
    "str.replace_all",
    Signature(String(), String(const=True), String(const=True), return_type=String()),
    param_names=["self", "substr", "replacement"],
)

str_starts_with = Operator(
    "str.starts_with",
    Signature(String(), String(const=True), return_type=Bool()),
    param_names=["self", "prefix"],
)

str_ends_with = Operator(
    "str.ends_with",
    Signature(String(), String(const=True), return_type=Bool()),
    param_names=["self", "suffix"],
)


str_contains = Operator(
    "str.contains",
    Signature(String(), String(const=True), return_type=Bool()),
    param_names=["self", "substr"],
)

str_slice = Operator(
    "str.slice",
    Signature(String(), Int(), Int(), return_type=String()),
    param_names=["self", "offset", "n"],
)

str_to_datetime = Operator(
    "str.to_datetime", Signature(String(), return_type=Datetime())
)

str_to_date = Operator("str.to_date", Signature(String(), return_type=Date()))
