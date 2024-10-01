from __future__ import annotations

from collections.abc import Iterable

from pydiverse.transform.tree import dtypes
from pydiverse.transform.tree.col_expr import (
    ColExpr,
    ColFn,
    WhenClause,
    wrap_literal,
)

__all__ = ["count", "row_number", "rank", "when", "dense_rank", "min", "max"]


def clean_kwargs(**kwargs) -> dict[str, list[ColExpr]]:
    return {key: wrap_literal(val) for key, val in kwargs.items() if val is not None}


def when(condition: ColExpr) -> WhenClause:
    condition = wrap_literal(condition)
    if condition.dtype() is not None and condition.dtype() != dtypes.Bool:
        raise TypeError(
            "argument for `when` must be of boolean type, but has type "
            f"`{condition.dtype()}`"
        )

    return WhenClause([], wrap_literal(condition))


def count(
    expr: ColExpr | None = None,
    *,
    filter: ColExpr | Iterable[ColExpr] | None = None,  # noqa: A002
):
    if expr is None:
        return ColFn("count", **clean_kwargs(filter=filter))
    else:
        return ColFn("count", wrap_literal(expr))


def row_number(
    *,
    arrange: ColExpr | Iterable[ColExpr],
    partition_by: ColExpr | list[ColExpr] | None = None,
):
    return ColFn(
        "row_number", **clean_kwargs(arrange=arrange, partition_by=partition_by)
    )


def rank(
    *,
    arrange: ColExpr | Iterable[ColExpr],
    partition_by: ColExpr | Iterable[ColExpr] | None = None,
):
    return ColFn("rank", **clean_kwargs(arrange=arrange, partition_by=partition_by))


def dense_rank(
    *,
    arrange: ColExpr | Iterable[ColExpr],
    partition_by: ColExpr | Iterable[ColExpr] | None = None,
):
    return ColFn(
        "dense_rank", **clean_kwargs(arrange=arrange, partition_by=partition_by)
    )


def min(arg: ColExpr, *additional_args: ColExpr):
    return ColFn("__least", wrap_literal(arg), *wrap_literal(additional_args))


def max(arg: ColExpr, *additional_args: ColExpr):
    return ColFn("__greatest", wrap_literal(arg), *wrap_literal(additional_args))
