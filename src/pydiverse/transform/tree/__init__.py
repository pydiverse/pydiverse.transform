from __future__ import annotations

from . import preprocessing
from .table_expr import TableExpr

__all__ = ["preprocess", "TableExpr"]


def preprocess(expr: TableExpr) -> TableExpr:
    preprocessing.rename_overwritten_cols(expr)
    preprocessing.propagate_names(expr, set())
    preprocessing.propagate_types(expr)
    preprocessing.update_partition_by_kwarg(expr)
