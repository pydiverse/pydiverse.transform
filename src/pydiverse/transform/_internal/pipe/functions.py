# ruff: noqa: A002

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydiverse.transform._internal.tree import types
from pydiverse.transform._internal.tree.col_expr import (
    Col,
    ColExpr,
    ColFn,
    ColName,
    LiteralCol,
    WhenClause,
    wrap_literal,
)
from pydiverse.transform._internal.tree.types import Bool, Dtype, Int

__all__ = ["len", "row_number", "rank", "when", "dense_rank", "min", "max"]


def when(condition: ColExpr) -> WhenClause:
    condition = wrap_literal(condition)
    if condition.dtype() is not None and condition.dtype() != Bool:
        raise TypeError(
            "argument for `when` must be of boolean type, but has type "
            f"`{condition.dtype()}`"
        )

    return WhenClause([], wrap_literal(condition))


def lit(val: Any, dtype: Dtype | None = None) -> LiteralCol:
    if types.is_concrete(dtype):
        return LiteralCol(val, dtype).cast(dtype)
    return LiteralCol(val, dtype)


def dense_rank(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    arrange: ColExpr | Iterable[ColExpr] | None = None,
    filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
) -> ColExpr[Int]:
    return ColFn(
        "dense_rank", partition_by=partition_by, arrange=arrange, filter=filter
    )


def max(*args: ColExpr) -> ColExpr:
    return ColFn("hmax", *args)


def min(*args: ColExpr) -> ColExpr:
    return ColFn("hmin", *args)


def len(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
) -> ColExpr[Int]:
    return ColFn("len", partition_by=partition_by, filter=filter)


def rank(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    arrange: ColExpr | Iterable[ColExpr] | None = None,
    filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
) -> ColExpr[Int]:
    return ColFn("rank", partition_by=partition_by, arrange=arrange, filter=filter)


def row_number(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    arrange: ColExpr | Iterable[ColExpr] | None = None,
    filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
) -> ColExpr[Int]:
    return ColFn(
        "row_number", partition_by=partition_by, arrange=arrange, filter=filter
    )
