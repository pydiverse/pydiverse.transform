from __future__ import annotations

from pydiverse.transform._internal.ops.operator import ElementWise
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.dtypes import (
    Date,
    Datetime,
    Float64,
    Int64,
    String,
)

__all__ = [
    "HorizontalMax",
    "HorizontalMin",
]


class HorizontalMax(ElementWise):
    name = "horizontal_max"
    signatures = [
        Signature(Int64, ..., returns=Int64),
        Signature(Float64, ..., returns=Float64),
        Signature(String, ..., returns=String),
        Signature(Datetime, ..., returns=Datetime),
        Signature(Date, ..., returns=Date),
    ]


class HorizontalMin(HorizontalMax):
    name = "horizontal_min"
