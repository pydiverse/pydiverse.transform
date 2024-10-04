from __future__ import annotations

from pydiverse.transform._internal.ops.core import Nullary, Window

__all__ = [
    "Shift",
    "RowNumber",
    "Rank",
    "DenseRank",
]


class Shift(Window):
    name = "shift"
    signatures = [
        "T, const int64 -> T",
        "T, const int64, const T -> T",
    ]


class RowNumber(Window, Nullary):
    name = "row_number"
    signatures = [
        "-> int64",
    ]


class Rank(Window, Nullary):
    name = "rank"
    signatures = [
        "-> int64",
    ]


class DenseRank(Window, Nullary):
    name = "dense_rank"
    signatures = [
        "-> int64",
    ]
