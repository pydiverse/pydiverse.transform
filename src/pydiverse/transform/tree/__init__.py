from __future__ import annotations

from pydiverse.transform.util.map2d import Map2d

from . import verbs
from .table_expr import TableExpr

__all__ = ["preprocess", "TableExpr"]


def preprocess(expr: TableExpr) -> TableExpr:
    verbs.rename_overwritten_cols(expr)
    verbs.propagate_names(expr, Map2d())
    verbs.propagate_types(expr)
    verbs.update_partition_by_kwarg(expr)
