from __future__ import annotations

from .core import Aggregate, Unary

__all__ = [
    "Min",
    "Max",
    "Mean",
    "Sum",
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


class StringJoin(Aggregate):
    name = "join"
    signatures = [
        "str, str -> str",
    ]


class Count(Aggregate):
    name = "count"
    signatures = [
        "-> int",
        "T -> int",
    ]
