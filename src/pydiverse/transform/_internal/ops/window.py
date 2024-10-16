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
        "T, const int, const T -> T",
    ]
    arg_names = ["self", "n", "fill_value"]
    defaults = [..., ..., None]


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
