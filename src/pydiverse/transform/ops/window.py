from __future__ import annotations

from pydiverse.transform.ops.core import Nullary, Window

__all__ = [
    "Shift",
    "RowNumber",
    "Rank",
    "DenseRank",
]


class Shift(Window):
    name = "shift"
    signatures = [
        "T, const int -> T",
        "T, const int, const T -> T",
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


class DenseRank(Window, Nullary):
    name = "dense_rank"
    signatures = [
        "-> int",
    ]
