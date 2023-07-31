from __future__ import annotations

from pydiverse.transform.ops.core import Nullary, Unary, Window

__all__ = [
    "Shift",
    "RowNumber",
    "Rank",
    "DenseRank",
]


class WindowImplicitArrange(Window):
    """
    Like a window function, except that the expression on which this op
    gets called, is used for arranging.

    Converts a call like this ``tbl.col1.nulls_first().rank()``, into a call like this
    ``(tbl.col1).rank(arrange=[tbl.col1.nulls_first()])
    """

    def mutate_args(self, args, kwargs):
        if len(args) == 0:
            return args, kwargs

        from pydiverse.transform.core.util.util import ordering_peeler

        arrange = args[0]
        peeled_first_arg, _, _ = ordering_peeler(arrange)
        args = (peeled_first_arg, *args[1:])

        assert "arrange" not in kwargs
        kwargs = {**kwargs, "arrange": [arrange]}
        return args, kwargs


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


class Rank(WindowImplicitArrange, Unary):
    name = "rank"
    signatures = [
        "T -> int",
    ]


class DenseRank(WindowImplicitArrange, Unary):
    name = "dense_rank"
    signatures = [
        "T -> int",
    ]
