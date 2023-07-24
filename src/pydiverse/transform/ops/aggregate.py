from __future__ import annotations

from pydiverse.transform.ops.core import Aggregate, Unary

__all__ = [
    "Min",
    "Max",
    "Mean",
    "Sum",
    "Any",
    "All",
    "StringJoin",
    "Count",
]


class Min(Aggregate, Unary):
    name = "min"
    signatures = [
        "int -> int",
        "float -> float",
        "str -> str",
    ]


class Max(Aggregate, Unary):
    name = "max"
    signatures = [
        "int -> int",
        "float -> float",
        "str -> str",
    ]


class Mean(Aggregate, Unary):
    name = "mean"
    signatures = [
        "int -> float",
        "float -> float",
    ]


class Sum(Aggregate, Unary):
    name = "sum"
    signatures = [
        "int -> int",
        "float -> float",
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


class StringJoin(Aggregate):
    name = "join"
    signatures = [
        "str, const str -> str",
    ]


class Count(Aggregate):
    name = "count"
    signatures = [
        "-> int",
        "T -> int",
    ]
