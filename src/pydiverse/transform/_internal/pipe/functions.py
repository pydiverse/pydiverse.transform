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
    Duration,
    Float,
    Int,
    String,
)


def when(condition: ColExpr) -> WhenClause:
    condition = wrap_literal(condition)
    if (
        condition.dtype() is not None
        and not types.without_const(condition.dtype()) == Bool()
    ):
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


# --- from here the code is generated, do not delete this comment ---


def coalesce(arg: ColExpr, *args: ColExpr) -> ColExpr:
    """
    Returns the first non-null value among the given.

    :param arg:
        The first value.

    :param args:
        Further values. All must have the same type.

    Examples
    --------
    >>> t = pdt.Table(
    ...     {
    ...         "a": [5, None, 435, -1, 8, None],
    ...         "b": [-45, None, 6, 23, 1, 0],
    ...         "c": [10, 2, None, None, None, None],
    ...     }
    ... )
    >>> (
    ...     t
    ...     >> mutate(
    ...         x=pdt.coalesce(t.a, t.b, t.c),
    ...         y=pdt.coalesce(t.c, t.b, t.a),
    ...     )
    ...     >> show()
    ... )
    Table <unnamed>, backend: PolarsImpl
    shape: (6, 5)
    ┌──────┬──────┬──────┬─────┬─────┐
    │ a    ┆ b    ┆ c    ┆ x   ┆ y   │
    │ ---  ┆ ---  ┆ ---  ┆ --- ┆ --- │
    │ i64  ┆ i64  ┆ i64  ┆ i64 ┆ i64 │
    ╞══════╪══════╪══════╪═════╪═════╡
    │ 5    ┆ -45  ┆ 10   ┆ 5   ┆ 10  │
    │ null ┆ null ┆ 2    ┆ 2   ┆ 2   │
    │ 435  ┆ 6    ┆ null ┆ 435 ┆ 6   │
    │ -1   ┆ 23   ┆ null ┆ -1  ┆ 23  │
    │ 8    ┆ 1    ┆ null ┆ 8   ┆ 1   │
    │ null ┆ 0    ┆ null ┆ 0   ┆ 0   │
    └──────┴──────┴──────┴─────┴─────┘
    """

    return ColFn(ops.coalesce, arg, *args)


def count(
    *,
    partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
    filter: ColExpr[Bool] | Iterable[ColExpr[Bool]] | None = None,
) -> ColExpr[Int]:
    """
    Returns the number of rows of the current table, like :code:`COUNT(*)` in SQL.
    """

    return ColFn(ops.count_star, partition_by=partition_by, filter=filter)


def dense_rank(
    *,
    partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
    arrange: ColExpr | Iterable[ColExpr],
) -> ColExpr[Int]:
    """
    The number of smaller or equal values in the column (not counting duplicates).

    This function has two syntax alternatives, as shown in the example below. The
    pdt. version is a bit more flexible, because it allows sorting by multiple
    expressions.

    Examples
    --------
    >>> t = pdt.Table({"a": [5, -1, 435, -1, 8, None, 8]})
    >>> (
    ...     t
    ...     >> mutate(
    ...         x=t.a.nulls_first().dense_rank(),
    ...         y=pdt.dense_rank(arrange=t.a.nulls_first()),
    ...     )
    ...     >> show()
    ... )
    Table <unnamed>, backend: PolarsImpl
    shape: (7, 3)
    ┌──────┬─────┬─────┐
    │ a    ┆ x   ┆ y   │
    │ ---  ┆ --- ┆ --- │
    │ i64  ┆ i64 ┆ i64 │
    ╞══════╪═════╪═════╡
    │ 5    ┆ 3   ┆ 3   │
    │ -1   ┆ 2   ┆ 2   │
    │ 435  ┆ 5   ┆ 5   │
    │ -1   ┆ 2   ┆ 2   │
    │ 8    ┆ 4   ┆ 4   │
    │ null ┆ 1   ┆ 1   │
    │ 8    ┆ 4   ┆ 4   │
    └──────┴─────┴─────┘
    """

    return ColFn(ops.dense_rank, partition_by=partition_by, arrange=arrange)


def all(arg: ColExpr[Bool], *args: ColExpr[Bool]) -> ColExpr[Bool]:
    """"""

    return ColFn(ops.horizontal_all, arg, *args)


def any(arg: ColExpr[Bool], *args: ColExpr[Bool]) -> ColExpr[Bool]:
    """"""

    return ColFn(ops.horizontal_any, arg, *args)


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


@overload
def max(arg: ColExpr[Bool], *args: ColExpr[Bool]) -> ColExpr[Bool]: ...


def max(arg: ColExpr, *args: ColExpr) -> ColExpr:
    """
    The maximum of the given columns.

    Examples
    --------
    >>> t = pdt.Table(
    ...     {
    ...         "a": [5, None, 435, -1, 8, None],
    ...         "b": [-45, None, 6, 23, -1, 0],
    ...         "c": [10, None, 2, None, -53, 3],
    ...     }
    ... )
    >>> t >> mutate(x=pdt.max(t.a, t.b, t.c)) >> show()
    Table <unnamed>, backend: PolarsImpl
    shape: (6, 4)
    ┌──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ c    ┆ x    │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  ┆ i64  │
    ╞══════╪══════╪══════╪══════╡
    │ 5    ┆ -45  ┆ 10   ┆ 10   │
    │ null ┆ null ┆ null ┆ null │
    │ 435  ┆ 6    ┆ 2    ┆ 435  │
    │ -1   ┆ 23   ┆ null ┆ 23   │
    │ 8    ┆ -1   ┆ -53  ┆ 8    │
    │ null ┆ 0    ┆ 3    ┆ 3    │
    └──────┴──────┴──────┴──────┘
    """

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


@overload
def min(arg: ColExpr[Bool], *args: ColExpr[Bool]) -> ColExpr[Bool]: ...


def min(arg: ColExpr, *args: ColExpr) -> ColExpr:
    """
    The minimum of the given columns.

    Examples
    --------
    >>> t = pdt.Table(
    ...     {
    ...         "a": [5, None, 435, -1, 8, None],
    ...         "b": [-45, None, 6, 23, -1, 0],
    ...         "c": [10, None, 2, None, -53, 3],
    ...     }
    ... )
    >>> t >> mutate(x=pdt.min(t.a, t.b, t.c)) >> show()
    Table <unnamed>, backend: PolarsImpl
    shape: (6, 4)
    ┌──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ c    ┆ x    │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  ┆ i64  │
    ╞══════╪══════╪══════╪══════╡
    │ 5    ┆ -45  ┆ 10   ┆ -45  │
    │ null ┆ null ┆ null ┆ null │
    │ 435  ┆ 6    ┆ 2    ┆ 2    │
    │ -1   ┆ 23   ┆ null ┆ -1   │
    │ 8    ┆ -1   ┆ -53  ┆ -53  │
    │ null ┆ 0    ┆ 3    ┆ 0    │
    └──────┴──────┴──────┴──────┘
    """

    return ColFn(ops.horizontal_min, arg, *args)


@overload
def sum(arg: ColExpr[Int], *args: ColExpr[Int]) -> ColExpr[Int]: ...


@overload
def sum(arg: ColExpr[Float], *args: ColExpr[Float]) -> ColExpr[Float]: ...


@overload
def sum(arg: ColExpr[Decimal], *args: ColExpr[Decimal]) -> ColExpr[Decimal]: ...


@overload
def sum(arg: ColExpr[String], *args: ColExpr[String]) -> ColExpr[String]: ...


@overload
def sum(arg: ColExpr[Duration], *args: ColExpr[Duration]) -> ColExpr[Duration]: ...


def sum(arg: ColExpr, *args: ColExpr) -> ColExpr:
    """"""

    return ColFn(ops.horizontal_sum, arg, *args)


def rank(
    *,
    partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
    arrange: ColExpr | Iterable[ColExpr],
) -> ColExpr[Int]:
    """
    The number of strictly smaller elements in the column plus one.

    This is the same as ``rank("min")`` in polars. This function has two syntax
    alternatives, as shown in the example below. The pdt. version is a bit more
    flexible, because it allows sorting by multiple expressions.


    Examples
    --------
    >>> t = pdt.Table({"a": [5, -1, 435, -1, 8, None, 8]})
    >>> (
    ...     t
    ...     >> mutate(
    ...         x=t.a.nulls_first().rank(),
    ...         y=pdt.rank(arrange=t.a.nulls_first()),
    ...     )
    ...     >> show()
    ... )
    Table <unnamed>, backend: PolarsImpl
    shape: (7, 3)
    ┌──────┬─────┬─────┐
    │ a    ┆ x   ┆ y   │
    │ ---  ┆ --- ┆ --- │
    │ i64  ┆ i64 ┆ i64 │
    ╞══════╪═════╪═════╡
    │ 5    ┆ 4   ┆ 4   │
    │ -1   ┆ 2   ┆ 2   │
    │ 435  ┆ 7   ┆ 7   │
    │ -1   ┆ 2   ┆ 2   │
    │ 8    ┆ 5   ┆ 5   │
    │ null ┆ 1   ┆ 1   │
    │ 8    ┆ 5   ┆ 5   │
    └──────┴─────┴─────┘
    """

    return ColFn(ops.rank, partition_by=partition_by, arrange=arrange)


def row_number(
    *,
    partition_by: Col | ColName | str | Iterable[Col | ColName | str] | None = None,
    arrange: ColExpr | Iterable[ColExpr] | None = None,
) -> ColExpr[Int]:
    """
    Computes the index of a row.

    Via the *arrange* argument, this can be done relative to a different order of
    the rows. But note that the result may not be unique if the argument of
    *arrange* contains duplicates.

    Examples
    --------
    >>> t = pdt.Table({"a": [5, -1, 435, -34, 8, None, 0]})
    >>> (
    ...     t
    ...     >> mutate(
    ...         x=pdt.row_number(),
    ...         y=pdt.row_number(arrange=t.a),
    ...     )
    ...     >> show()
    ... )
    Table <unnamed>, backend: PolarsImpl
    shape: (7, 3)
    ┌──────┬─────┬─────┐
    │ a    ┆ x   ┆ y   │
    │ ---  ┆ --- ┆ --- │
    │ i64  ┆ i64 ┆ i64 │
    ╞══════╪═════╪═════╡
    │ 5    ┆ 1   ┆ 5   │
    │ -1   ┆ 2   ┆ 3   │
    │ 435  ┆ 3   ┆ 7   │
    │ -34  ┆ 4   ┆ 2   │
    │ 8    ┆ 5   ┆ 6   │
    │ null ┆ 6   ┆ 1   │
    │ 0    ┆ 7   ┆ 4   │
    └──────┴─────┴─────┘
    """

    return ColFn(ops.row_number, partition_by=partition_by, arrange=arrange)
