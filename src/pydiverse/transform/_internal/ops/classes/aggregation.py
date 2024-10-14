from __future__ import annotations

from pydiverse.transform._internal.ops.operator import Aggregate
from pydiverse.transform._internal.ops.signature import Signature, T
from pydiverse.transform._internal.tree.dtypes import (
    Bool,
    Date,
    Datetime,
    Float64,
    Int64,
    String,
)

__all__ = [
    "Min",
    "Max",
    "Mean",
    "Sum",
    "Any",
    "All",
    "Count",
]


class Min(Aggregate):
    name = "min"

    signatures = [
        Signature(Int64, returns=Int64),
        Signature(Float64, returns=Float64),
        Signature(String, returns=String),
        Signature(Datetime, returns=Datetime),
        Signature(Date, returns=Date),
    ]


class Max(Min):
    name = "max"


class Mean(Aggregate):
    name = "mean"
    signatures = [
        Signature(Int64, returns=Int64),
        Signature(Float64, returns=Float64),
    ]


class Sum(Mean):
    name = "sum"


class Any(Aggregate):
    name = "any"
    signatures = [Signature(Bool, returns=Bool)]


class All(Any):
    name = "all"


class Count(Aggregate):
    name = "count"
    signatures = [
        Signature(returns=Int64),
        Signature(T, returns=Int64),
    ]
