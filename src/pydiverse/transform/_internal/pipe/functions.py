from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydiverse.transform._internal.tree import dtypes
from pydiverse.transform._internal.tree.col_expr import (
    ColExpr,
    ColFn,
    LiteralCol,
    WhenClause,
    wrap_literal,
)

__all__ = ["count", "row_number", "rank", "when", "dense_rank", "min", "max"]


def when(condition: ColExpr) -> WhenClause:
    condition = wrap_literal(condition)
    if condition.dtype() is not None and condition.dtype() != dtypes.Bool:
        raise TypeError(
            "argument for `when` must be of boolean type, but has type "
            f"`{condition.dtype()}`"
        )

    return WhenClause([], wrap_literal(condition))


def lit(val: Any, dtype: dtypes.Dtype | None = None) -> LiteralCol:
    return LiteralCol(val, dtype)


# TODO: also do code gen for these


def count(
    expr: ColExpr | None = None,
    *,
    filter: ColExpr | Iterable[ColExpr] | None = None,  # noqa: A002
):
    if expr is None:
        return ColFn("count", filter=filter)
    else:
        return ColFn("count", expr)


def row_number(
    *,
    arrange: ColExpr | Iterable[ColExpr],
    partition_by: ColExpr | list[ColExpr] | None = None,
):
    return ColFn("row_number", arrange=arrange, partition_by=partition_by)


def rank(
    *,
    arrange: ColExpr | Iterable[ColExpr],
    partition_by: ColExpr | Iterable[ColExpr] | None = None,
):
    return ColFn("rank", arrange=arrange, partition_by=partition_by)


def dense_rank(
    *,
    arrange: ColExpr | Iterable[ColExpr],
    partition_by: ColExpr | Iterable[ColExpr] | None = None,
):
    return ColFn("dense_rank", arrange=arrange, partition_by=partition_by)


def min(arg: ColExpr, *additional_args: ColExpr):
    return ColFn("__least", arg, *additional_args)


def max(arg: ColExpr, *additional_args: ColExpr):
    return ColFn("__greatest", arg, *additional_args)
