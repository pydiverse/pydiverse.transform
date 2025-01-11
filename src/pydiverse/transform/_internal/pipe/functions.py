# ruff: noqa: A002

from __future__ import annotations

import functools
import operator
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


def when(condition: ColExpr) -> WhenClause:
    condition = wrap_literal(condition)
    if condition.dtype() is not None and not condition.dtype() <= Bool():
        raise TypeError(
            "argument for `when` must be of boolean type, but has type "
            f"`{condition.dtype()}`"
        )

    return WhenClause([], wrap_literal(condition))


def lit(val: Any, dtype: Dtype | None = None) -> LiteralCol:
    """
    Creates a pydiverse.transform expression from a python builtin type.

    Usually, you can just use python builtins in expressions without wrapping them in
    ``lit``. The pydiverse.transform data type of the value is then inferred. However,
    ``lit`` allows to set the exact pydiverse.transform type, which may be useful
    sometimes.
    """
    if dtype is not None and types.is_subtype(dtype):
        return LiteralCol(val, dtype).cast(dtype)
    return LiteralCol(val, dtype)


def all(arg: ColExpr[Bool], *args: ColExpr[Bool]) -> ColExpr[Bool]:
    return functools.reduce(operator.and_, (arg, *args))


def any(arg: ColExpr[Bool], *args: ColExpr[Bool]) -> ColExpr[Bool]:
    return functools.reduce(operator.or_, (arg, *args))


@overload
def sum(arg: ColExpr[Int], *args: ColExpr[Int]) -> ColExpr[Int]: ...


@overload
def sum(arg: ColExpr[Float], *args: ColExpr[Float]) -> ColExpr[Float]: ...


@overload
def sum(arg: ColExpr[Decimal], *args: ColExpr[Decimal]) -> ColExpr[Decimal]: ...


@overload
def sum(arg: ColExpr[String], *args: ColExpr[String]) -> ColExpr[String]: ...


def sum(arg: ColExpr, *args: ColExpr) -> ColExpr:
    return functools.reduce(operator.add, (arg, *args))


# --- from here the code is generated, do not delete this comment ---


def coalesce(arg: ColExpr, *args: ColExpr) -> ColExpr:
    """"""

    return ColFn(ops.coalesce, arg, *args)


def count(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
) -> ColExpr[Int]:
    """
    Returns the number of rows of the current table, like :code:`COUNT(*)` in SQL.
    """

    return ColFn(ops.count_star, partition_by=partition_by, filter=filter)


def dense_rank(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    arrange: ColExpr | Iterable[ColExpr],
) -> ColExpr[Int]:
    """"""

    return ColFn(ops.dense_rank, partition_by=partition_by, arrange=arrange)


@overload
def max(arg: ColExpr[Int], *args: ColExpr[Int]) -> ColExpr[Int]: ...


@overload
def max(arg: ColExpr[Float], *args: ColExpr[Float]) -> ColExpr[Float]: ...


@overload
def max(arg: ColExpr[Decimal], *args: ColExpr[Decimal]) -> ColExpr[Decimal]: ...


@overload
def max(arg: ColExpr[String], *args: ColExpr[String]) -> ColExpr[String]: ...


@overload
def max(arg: ColExpr[Datetime], *args: ColExpr[Datetime]) -> ColExpr[Datetime]: ...


@overload
def max(arg: ColExpr[Date], *args: ColExpr[Date]) -> ColExpr[Date]: ...


def max(arg: ColExpr, *args: ColExpr) -> ColExpr:
    """"""

    return ColFn(ops.horizontal_max, arg, *args)


@overload
def min(arg: ColExpr[Int], *args: ColExpr[Int]) -> ColExpr[Int]: ...


@overload
def min(arg: ColExpr[Float], *args: ColExpr[Float]) -> ColExpr[Float]: ...


@overload
def min(arg: ColExpr[Decimal], *args: ColExpr[Decimal]) -> ColExpr[Decimal]: ...


@overload
def min(arg: ColExpr[String], *args: ColExpr[String]) -> ColExpr[String]: ...


@overload
def min(arg: ColExpr[Datetime], *args: ColExpr[Datetime]) -> ColExpr[Datetime]: ...


@overload
def min(arg: ColExpr[Date], *args: ColExpr[Date]) -> ColExpr[Date]: ...


def min(arg: ColExpr, *args: ColExpr) -> ColExpr:
    """"""

    return ColFn(ops.horizontal_min, arg, *args)


def rank(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    arrange: ColExpr | Iterable[ColExpr],
) -> ColExpr[Int]:
    """
    The number of strictly smaller elements in the column plus one.

    This is the same as ``rank("min")`` in polars.

    Examples
    --------
    >>> t = pdt.Table({"a": [3, 1, 4, 1, 5, 9, 4]})
    >>> t >> mutate(b=pdt.rank(arrange=t.a)) >> export(Polars(lazy=False))
    shape: (7, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 3   ┆ 3   │
    │ 1   ┆ 1   │
    │ 4   ┆ 4   │
    │ 1   ┆ 1   │
    │ 5   ┆ 6   │
    │ 9   ┆ 7   │
    │ 4   ┆ 4   │
    └─────┴─────┘
    """

    return ColFn(ops.rank, partition_by=partition_by, arrange=arrange)


def row_number(
    *,
    partition_by: Col | ColName | Iterable[Col | ColName] | None = None,
    arrange: ColExpr | Iterable[ColExpr],
) -> ColExpr[Int]:
    """"""

    return ColFn(ops.row_number, partition_by=partition_by, arrange=arrange)
