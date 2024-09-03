from __future__ import annotations

from . import verbs
from .table_expr import TableExpr

__all__ = ["propagate_names", "propagate_types", "TableExpr"]


def propagate_names(expr: TableExpr):
    verbs.propagate_names(expr, set())


def propagate_types(expr: TableExpr):
    verbs.propagate_types(expr)
