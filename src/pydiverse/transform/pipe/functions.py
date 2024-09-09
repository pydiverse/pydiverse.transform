from __future__ import annotations

from pydiverse.transform.tree.col_expr import (
    ColExpr,
    ColFn,
    Order,
    WhenClause,
)

__all__ = [
    "count",
    "row_number",
]


def when(condition: ColExpr) -> WhenClause:
    return WhenClause([], condition)


def count(expr: ColExpr | None = None):
    if expr is None:
        return ColFn("count")
    else:
        return ColFn("count", expr)


def row_number(*, arrange: list[ColExpr], partition_by: list[ColExpr] | None = None):
    return ColFn(
        "row_number",
        arrange=[Order.from_col_expr(ord) for ord in arrange],
        partition_by=partition_by,
    )


def rank(*, arrange: list[ColExpr], partition_by: list[ColExpr] | None = None):
    return ColFn(
        "rank",
        arrange=[Order.from_col_expr(ord) for ord in arrange],
        partition_by=partition_by,
    )


def dense_rank(*, arrange: list[ColExpr], partition_by: list[ColExpr] | None = None):
    return ColFn(
        "dense_rank",
        arrange=[Order.from_col_expr(ord) for ord in arrange],
        partition_by=partition_by,
    )


def min(first: ColExpr, *expr: ColExpr):
    return ColFn("__least", first, *expr)


def max(first: ColExpr, *expr: ColExpr):
    return ColFn("__greatest", first, *expr)
