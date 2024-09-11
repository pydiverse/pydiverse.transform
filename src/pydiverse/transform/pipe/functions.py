from __future__ import annotations

from pydiverse.transform.tree.col_expr import (
    ColExpr,
    ColFn,
    WhenClause,
    wrap_literal,
)

__all__ = [
    "count",
    "row_number",
]


def clean_kwargs(**kwargs) -> dict[str, list[ColExpr]]:
    return {key: wrap_literal(val) for key, val in kwargs.items() if val is not None}


def when(condition: ColExpr) -> WhenClause:
    return WhenClause([], wrap_literal(condition))


def count(expr: ColExpr | None = None):
    if expr is None:
        return ColFn("count")
    else:
        return ColFn("count", wrap_literal(expr))


def row_number(*, arrange: list[ColExpr], partition_by: list[ColExpr] | None = None):
    return ColFn(
        "row_number", **clean_kwargs(arrange=arrange, partition_by=partition_by)
    )


def rank(*, arrange: list[ColExpr], partition_by: list[ColExpr] | None = None):
    return ColFn("rank", **clean_kwargs(arrange=arrange, partition_by=partition_by))


def dense_rank(*, arrange: list[ColExpr], partition_by: list[ColExpr] | None = None):
    return ColFn(
        "dense_rank", **clean_kwargs(arrange=arrange, partition_by=partition_by)
    )


def min(first: ColExpr, *expr: ColExpr):
    return ColFn("__least", wrap_literal(first), *wrap_literal(expr))


def max(first: ColExpr, *expr: ColExpr):
    return ColFn("__greatest", wrap_literal(first), *wrap_literal(expr))
