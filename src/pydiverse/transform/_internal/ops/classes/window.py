from __future__ import annotations

from pydiverse.transform._internal.ops.operator import Window
from pydiverse.transform._internal.ops.signature import Signature, T, const
from pydiverse.transform._internal.tree.dtypes import Int64

__all__ = [
    "Shift",
    "RowNumber",
    "Rank",
    "DenseRank",
]


class Shift(Window):
    name = "shift"
    signatures = [Signature(T, const(Int64), const(T), returns=T)]
    arg_names = ["self", "n", "default"]
    defaults = [..., ..., None]


class RowNumber(Window):
    name = "row_number"
    signatures = [Signature(returns=Int64)]
    is_expression_method = False


class Rank(RowNumber):
    name = "rank"


class DenseRank(Rank):
    name = "dense_rank"
