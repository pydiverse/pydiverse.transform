from __future__ import annotations

from .core import Nullary, Window

__all__ = [
    "Shift",
    "RowNumber",
]


class Shift(Window):
    name = "shift"
    signatures = [
        "T, const-int -> T",
        "T, const-int, T -> T",
    ]


class RowNumber(Window, Nullary):
    name = "row_number"
    signatures = [
        "-> int",
    ]
