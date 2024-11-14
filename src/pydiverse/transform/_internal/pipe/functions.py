# ruff: noqa: A002

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, overload

from pydiverse.transform._internal.ops import ops
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
from pydiverse.transform._internal.tree.types import (
    Bool,
    Date,
    Datetime,
    Decimal,
    Dtype,
    Float,
    Int,
    String,
)

__all__ = ["len", "row_number", "rank", "when", "dense_rank", "min", "max"]


def when(condition: ColExpr) -> WhenClause:
    condition = wrap_literal(condition)
    if condition.dtype() is not None and not condition.dtype() <= Bool():
        raise TypeError(
            "argument for `when` must be of boolean type, but has type "
            f"`{condition.dtype()}`"
        )

    return WhenClause([], wrap_literal(condition))


def lit(val: Any, dtype: Dtype | None = None) -> LiteralCol:
    if dtype is not None and types.is_subtype(dtype):
        return LiteralCol(val, dtype).cast(dtype)
    return LiteralCol(val, dtype)


def dense_rank(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    arrange: ColExpr | Iterable[ColExpr] | None = None,
) -> ColExpr[Int]:
    return ColFn(ops.dense_rank, partition_by=partition_by, arrange=arrange)


@overload
def max(*args: ColExpr[Int]) -> ColExpr[Int]: ...


@overload
def max(*args: ColExpr[Float]) -> ColExpr[Float]: ...


@overload
def max(*args: ColExpr[Decimal]) -> ColExpr[Decimal]: ...


@overload
def max(*args: ColExpr[String]) -> ColExpr[String]: ...


@overload
def max(*args: ColExpr[Datetime]) -> ColExpr[Datetime]: ...


@overload
def max(*args: ColExpr[Date]) -> ColExpr[Date]: ...


def max(*args: ColExpr) -> ColExpr:
    return ColFn(ops.horizontal_max, *args)


@overload
def min(*args: ColExpr[Int]) -> ColExpr[Int]: ...


@overload
def min(*args: ColExpr[Float]) -> ColExpr[Float]: ...


@overload
def min(*args: ColExpr[Decimal]) -> ColExpr[Decimal]: ...


@overload
def min(*args: ColExpr[String]) -> ColExpr[String]: ...


@overload
def min(*args: ColExpr[Datetime]) -> ColExpr[Datetime]: ...


@overload
def min(*args: ColExpr[Date]) -> ColExpr[Date]: ...


def min(*args: ColExpr) -> ColExpr:
    return ColFn(ops.horizontal_min, *args)


def len(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
) -> ColExpr[Int]:
    return ColFn(ops.len, partition_by=partition_by, filter=filter)


def rank(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    arrange: ColExpr | Iterable[ColExpr] | None = None,
) -> ColExpr[Int]:
    return ColFn(ops.rank, partition_by=partition_by, arrange=arrange)


def row_number(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    arrange: ColExpr | Iterable[ColExpr] | None = None,
) -> ColExpr[Int]:
    return ColFn(ops.row_number, partition_by=partition_by, arrange=arrange)
