from __future__ import annotations

from . import verbs
from .col_expr import Map2d
from .table_expr import TableExpr
from .verbs import recursive_copy

__all__ = ["propagate_names", "propagate_types", "TableExpr"]


def propagate_names(expr: TableExpr):
    verbs.propagate_names(expr, Map2d())


def propagate_types(expr: TableExpr):
    verbs.propagate_types(expr)
