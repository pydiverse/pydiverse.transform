from __future__ import annotations

from .core import Window

__all__ = [
    "Shift",
    "RowNumber",
    "Rank",
]


class Shift(Window):
    name = "shift"
    signatures = [
        "T, const-int -> T",
        "T, const-int, T -> T",
    ]


class RowNumber(Window):
    name = "row_number"
    signatures = [
        "-> int",  # uses arrange argument
        "T -> int",
    ]


class Rank(Window):
    name = "rank"
    signatures = [
        "T -> int",
    ]
