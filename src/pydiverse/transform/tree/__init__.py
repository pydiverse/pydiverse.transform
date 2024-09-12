from __future__ import annotations

from . import preprocessing
from .table_expr import TableExpr

__all__ = ["preprocess", "TableExpr"]


def preprocess(expr: TableExpr) -> TableExpr:
    preprocessing.update_partition_by_kwarg(expr)
    preprocessing.propagate_needed_cols(expr)
