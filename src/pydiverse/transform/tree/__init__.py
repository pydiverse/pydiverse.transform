from __future__ import annotations

from pydiverse.transform.util.map2d import Map2d

from . import preprocessing
from .table_expr import TableExpr

__all__ = ["preprocess", "TableExpr"]


def preprocess(expr: TableExpr) -> TableExpr:
    preprocessing.rename_overwritten_cols(expr)
    preprocessing.propagate_names(expr, Map2d())
    preprocessing.propagate_types(expr)
    preprocessing.update_partition_by_kwarg(expr)
