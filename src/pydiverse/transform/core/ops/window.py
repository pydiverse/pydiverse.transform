from __future__ import annotations

from .core import Nullary, Window

__all__ = [
    "Shift",
    "RowNumber",
    "Rank",
]


class Shift(Window):
    name = "shift"
    signatures = [
        "T, int -> T",
        "T, int, T -> T",
    ]


class RowNumber(Window, Nullary):
    name = "row_number"
    signatures = [
        "-> int",
    ]


class Rank(Window, Nullary):
    name = "rank"
    signatures = [
        "-> int",
    ]
