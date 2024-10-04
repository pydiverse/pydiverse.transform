from __future__ import annotations

from pydiverse.transform._internal.ops.core import Aggregate, Unary

__all__ = [
    "Min",
    "Max",
    "Mean",
    "Sum",
    "Any",
    "All",
    "Count",
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


class Count(Aggregate):
    name = "count"
    signatures = [
        "-> int64",
        "T -> int64",
    ]
