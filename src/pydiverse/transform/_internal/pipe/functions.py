# ruff: noqa: A002

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydiverse.transform._internal.tree.col_expr import (
    Col,
    ColExpr,
    ColFn,
    ColName,
    LiteralCol,
    WhenClause,
    wrap_literal,
)
from pydiverse.transform._internal.tree.dtypes import Bool, Dtype, Int64

__all__ = ["count", "row_number", "rank", "when", "dense_rank", "min", "max"]


def when(condition: ColExpr) -> WhenClause:
    condition = wrap_literal(condition)
    if condition.dtype() is not None and condition.dtype() != Bool:
        raise TypeError(
            "argument for `when` must be of boolean type, but has type "
            f"`{condition.dtype()}`"
        )

    return WhenClause([], wrap_literal(condition))


def lit(val: Any, dtype: Dtype | None = None) -> LiteralCol:
    return LiteralCol(val, dtype)


def count(
    self: ColExpr = None,
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
) -> ColExpr[Int64]:
    return ColFn("count", self, partition_by=partition_by, filter=filter)


def dense_rank(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    arrange: ColExpr | Iterable[ColExpr] | None = None,
    filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
) -> ColExpr[Int64]:
    return ColFn(
        "dense_rank", partition_by=partition_by, arrange=arrange, filter=filter
    )


def max(*args: ColExpr) -> ColExpr:
    return ColFn("hmax", *args)


def min(*args: ColExpr) -> ColExpr:
    return ColFn("hmin", *args)


def rank(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    arrange: ColExpr | Iterable[ColExpr] | None = None,
    filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
) -> ColExpr[Int64]:
    return ColFn("rank", partition_by=partition_by, arrange=arrange, filter=filter)


def row_number(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    arrange: ColExpr | Iterable[ColExpr] | None = None,
    filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
) -> ColExpr[Int64]:
    return ColFn(
        "row_number", partition_by=partition_by, arrange=arrange, filter=filter
    )
