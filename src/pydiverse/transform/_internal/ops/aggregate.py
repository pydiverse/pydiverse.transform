from __future__ import annotations

from pydiverse.transform._internal.ops.core import Aggregate, Nullary, Unary

__all__ = [
    "Min",
    "Max",
    "Mean",
    "Sum",
    "Any",
    "All",
    "Count",
    "Len",
]


class Min(Aggregate, Unary):
    name = "min"
    signatures = [
        "int64 -> int64",
        "float64 -> float64",
        "str -> str",
        "datetime -> datetime",
        "date -> date",
    ]


class Max(Aggregate, Unary):
    name = "max"
    signatures = [
        "int64 -> int64",
        "float64 -> float64",
        "str -> str",
        "datetime -> datetime",
        "date -> date",
    ]


class Mean(Aggregate, Unary):
    name = "mean"
    signatures = [
        "int64 -> float64",
        "float64 -> float64",
    ]


class Sum(Aggregate, Unary):
    name = "sum"
    signatures = [
        "int64 -> int64",
        "float64 -> float64",
    ]


class Any(Aggregate, Unary):
    name = "any"
    signatures = [
        "bool -> bool",
    ]


class All(Aggregate, Unary):
    name = "all"
    signatures = [
        "bool -> bool",
    ]


class Count(Aggregate, Unary):
    name = "count"
    signatures = [
        "T -> int64",
    ]
    arg_names = ["self"]


class Len(Aggregate, Nullary):
    name = "len"
    signatures = ["-> int64"]
