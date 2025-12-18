# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import copy
import functools
import itertools
import operator
import uuid
from typing import Any, Literal, overload

import polars as pl

from pydiverse.common import Bool, Int64
from pydiverse.transform._internal import errors
from pydiverse.transform._internal.backend.table_impl import (
    TableImpl,
    split_join_cond,
)
from pydiverse.transform._internal.backend.targets import (
    Dict,
    DictOfLists,
    ListOfDicts,
    Polars,
    Scalar,
    Target,
)
from pydiverse.transform._internal.errors import (
    ColumnNotFoundError,
    DataTypeError,
    ErrorWithSource,
    FunctionTypeError,
)
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.ops.op import Ftype
from pydiverse.transform._internal.pipe.pipeable import (
    Pipeable,
    check_subquery,
    modify_ast,
    verb,
)
from pydiverse.transform._internal.pipe.table import Table
from pydiverse.transform._internal.tree import types
from pydiverse.transform._internal.tree.col_expr import (
    Col,
    ColExpr,
    ColFn,
    ColName,
    EvalAligned,
    LiteralCol,
    Order,
    Series,
    get_ast_path_str,
    wrap_literals,
)
from pydiverse.transform._internal.tree.verbs import (
    Alias,
    Arrange,
    Filter,
    GroupBy,
    Join,
    Mutate,
    Rename,
    Select,
    SliceHead,
    Summarize,
    Ungroup,
    Union,
)

__all__ = [
    "alias",
    "collect",
    "build_query",
    "show_query",
    "select",
    "drop",
    "rename",
    "mutate",
    "join",
    "left_join",
    "inner_join",
    "full_join",
    "union",
    "filter",
    "arrange",
    "group_by",
    "ungroup",
    "summarize",
    "slice_head",
    "export",
]


@overload
def alias(new_name: str | None = None, *, keep_col_refs: bool = False) -> Pipeable: ...


@verb
@modify_ast
def alias(table: Table, new_name: str | None = None, *, keep_col_refs: bool = False) -> Pipeable:
    """
    Changes the name of the current table and allows subqueries in SQL.

    :param new_name:
        The new name assigned to the table. If this is ``None``, the table
        retains its previous name.

    :param keep_col_refs:
        Whether column references are reset. If ``keep_col_refs=False``, the resulting
        table is not considered to be derived from the input table, i.e. one cannot use
        columns from the input table in subsequent operations on the result table.
        However, the reset of column references is necessary before performing a
        self-join.

    Examples
    --------

    A self join without applying ``alias`` before raises an exception:

    >>> t = pdt.Table({"a": [4, 2, 1, 4], "b": ["l", "g", "uu", "--   r"]})
    >>> t >> join(t, t.a == t.a, how="inner", suffix="_right")
    ValueError: table `<pydiverse.transform._internal.backend.polars.PolarsImpl object
    at 0x13f86d510>` occurs twice in the table tree.
    hint: To join two tables derived from a common table, apply `>> alias()` to one of
    them before the join.

    By applying ``alias`` to the right table and storing the result in a new variable,
    the self join succeeds.

    >>> (
    ...     t
    ...     >> join(s := t >> alias(), t.a == s.a, how="inner", suffix="_right")
    ...     >> show()
    ... )
    shape: (6, 4)
    ┌─────┬────────┬─────────┬─────────┐
    │ a   ┆ b      ┆ a_right ┆ b_right │
    │ --- ┆ ---    ┆ ---     ┆ ---     │
    │ i64 ┆ str    ┆ i64     ┆ str     │
    ╞═════╪════════╪═════════╪═════════╡
    │ 4   ┆ l      ┆ 4       ┆ l       │
    │ 4   ┆ --   r ┆ 4       ┆ l       │
    │ 2   ┆ g      ┆ 2       ┆ g       │
    │ 1   ┆ uu     ┆ 1       ┆ uu      │
    │ 4   ┆ l      ┆ 4       ┆ --   r  │
    │ 4   ┆ --   r ┆ 4       ┆ --   r  │
    └─────┴────────┴─────────┴─────────┘
    """

    if new_name is None:
        new_name = table._ast.name

    new = copy.copy(table)
    new._ast = Alias(
        new._ast,
        uuid_map={uid: uuid.uuid1() for uid in table._cache.cols.keys()} if not keep_col_refs else None,
    )
    new._ast.name = new_name

    return new


@overload
def collect(target: Target | None = None, *, keep_col_refs: bool = True) -> Pipeable: ...


@verb
def collect(
    table: Table,
    target: Target | None = None,
    *,
    keep_col_refs: bool = True,
) -> Pipeable:
    """
    Execute all accumulated operations and write the result to a new Table.

    This verb is only for polars-backed tables. All operations lazily stored in the
    table are executed and a new table containing the result is returned. The returned
    table always stored the data in a polars LazyFrame. One can choose whether the
    following operations on the table are executed via polars or DuckDB on the
    LazyFrame (see also :doc:`/examples/duckdb_polars_parquet`).

    :param target:
        The execution engine to be used from here on. Can be either ``Polars`` or
        ``DuckDb``.

    Examples
    --------
    Here, ``collect`` does not change anything in the result, but the ``mutate`` is
    executed on the DataFrame when ``collect`` is called, whereas the ``arrange`` is
    only executed when ``export`` is called. Without ``collect``, the ``mutate`` would
    only have been executed with the ``export``, too.

    >>> t = pdt.Table({"a": [4, 2, 1, 4], "b": ["l", "g", "uu", "--   r"]})
    >>> (
    ...     t
    ...     >> mutate(z=t.a + t.b.str.len())
    ...     >> collect()
    ...     >> arrange(C.z, t.a)
    ...     >> show()
    ... )
    shape: (4, 3)
    ┌─────┬────────┬─────┐
    │ a   ┆ b      ┆ z   │
    │ --- ┆ ---    ┆ --- │
    │ i64 ┆ str    ┆ i64 │
    ╞═════╪════════╪═════╡
    │ 1   ┆ uu     ┆ 3   │
    │ 2   ┆ g      ┆ 3   │
    │ 4   ┆ l      ┆ 5   │
    │ 4   ┆ --   r ┆ 10  │
    └─────┴────────┴─────┘
    """
    errors.check_arg_type(Target | type(Target) | type(None), "collect", "target", target)

    if target is None:
        target = Polars()

    if not isinstance(target, Target):
        assert issubclass(target, Target)
        target = target()

    df = table >> export(Polars(lazy=False))

    if not keep_col_refs:
        return Table(df)

    # TODO: keep_hidden_cols option

    assert len(table) == len(table._cache.name_to_uuid)

    new = Table(
        TableImpl.from_resource(
            df,
            target,
            name=table._ast.name,
            # preserve UUIDs -> this keeps column references across collect()
            uuids={name: uid for name, uid in table._cache.name_to_uuid.items()},
        )
    )
    new._cache.derived_from = table._cache.derived_from | {new._ast}
    new._cache.partition_by = [preprocess_arg(col, new) for col in table._cache.partition_by]

    return new


@overload
def export(target: Target, *, schema_overrides: dict[str, Any] | None = None) -> Pipeable: ...


@verb
def export(
    table: Table,
    target: Target | type[Target],
    *,
    schema_overrides: dict[str, Any] | None = None,
) -> Pipeable:
    """Convert a pydiverse.transform Table to a data frame.

    :param target:
        Can currently be either a ``Polars`` or ``Pandas`` object. For polars, one can
        specify whether a DataFrame or LazyFrame is returned via the ``lazy`` keyword
        parameter.
        If ``lazy=True``, no actual computations are performed, they just get stored in
        the LazyFrame.

    :param schema_overrides:
        A dictionary of column names to backend-specific data types. This controls which
        data types are used when writing to the backend. Because the data types are not
        constrained by pydiverse.transform's type system, this may sometimes be
        preferred over a cast.

    :return:
        A polars or pandas DataFrame / LazyFrame.

    Examples
    --------

    >>> t1 = pdt.Table(
    ...     {
    ...         "a": [3, 1, 4, 1, 5, 9],
    ...         "b": [2.465, 0.22, -4.477, 10.8, -81.2, 0.0],
    ...         "c": ["a,", "bcbc", "pydiverse", "transform", "'  '", "-22"],
    ...     }
    ... )
    >>> t1 >> export(Pandas())
       a      b          c
    0  3  2.465         a,
    1  1   0.22       bcbc
    2  4 -4.477  pydiverse
    3  1   10.8  transform
    4  5  -81.2       '  '
    5  9    0.0        -22
    >>> t1 >> show()
    shape: (6, 3)
    ┌─────┬────────┬───────────┐
    │ a   ┆ b      ┆ c         │
    │ --- ┆ ---    ┆ ---       │
    │ i64 ┆ f64    ┆ str       │
    ╞═════╪════════╪═══════════╡
    │ 3   ┆ 2.465  ┆ a,        │
    │ 1   ┆ 0.22   ┆ bcbc      │
    │ 4   ┆ -4.477 ┆ pydiverse │
    │ 1   ┆ 10.8   ┆ transform │
    │ 5   ┆ -81.2  ┆ '  '      │
    │ 9   ┆ 0.0    ┆ -22       │
    └─────┴────────┴───────────┘
    """

    errors.check_arg_type(Target | type(Target), "export", "target", target)

    if not isinstance(target, Target):
        assert issubclass(target, Target)
        target = target()

    if isinstance(target, Scalar):
        if len(table) != 1:
            raise TypeError(
                f"to export a table to a scalar, it must have exactly one column, but found {len(table)} columns"
            )
        df: pl.DataFrame = table >> export(Polars())
        if df.height != 1:
            raise TypeError(f"to export a table to a scalar, it must have exactly one row, but found {df.height} rows")
        return df.item()

    elif isinstance(target, Dict):
        df: pl.DataFrame = table >> export(Polars())
        if df.height != 1:
            raise TypeError(f"cannot export a table with more than one row to `Dict`, found {df.height} rows")
        return df.to_dicts()[0]

    elif isinstance(target, DictOfLists):
        return (table >> export(Polars())).to_dict(as_series=False)

    elif isinstance(target, ListOfDicts):
        return (table >> export(Polars())).to_dicts()

    # TODO: allow stuff like pdt.Int(): pl.UInt32() in schema_overrides and resolve that
    # to columns
    SourceBackend: type[TableImpl] = table._cache.backend
    if schema_overrides is None:
        schema_overrides = dict()
    return SourceBackend.export(
        table._ast.clone(),
        target,
        schema_overrides={table[col_name]._uuid: dtype for col_name, dtype in schema_overrides.items()},
    )


@overload
def build_query() -> Pipeable: ...


@verb
def build_query(table: Table) -> str | None:
    """
    Compiles the operations accumulated on the current table to a SQL query.

    :returns:
        The SQL query as a string.
    """

    return table._cache.backend.build_query(table._ast.clone())


@overload
def show_query(pipe: bool = False) -> Pipeable | None: ...


@verb
def show_query(table: Table, pipe: bool = False) -> Pipeable | None:
    """
    Prints the compiled SQL query to stdout.

    :param pipe:
        If set to `True`, the table is returned, else `None` is returned.

    Note
    ----
    During interactive development in the python shell, you usually want to keep
    `pipe = False`, else the table is printed, too. The main use for `pipe = True` is
    when you don't want to break a long sequence of verbs in a file.
    """

    if query := table >> build_query():
        print(query)
    else:
        print(f"No query to show for table {table._ast.short_name()}. (backend: {table._cache.backend.backend_name})")

    return table if pipe else None


@overload
def select(*cols: Col | ColName | str) -> Pipeable: ...


@verb
@modify_ast
def select(table: Table, *cols: Col | ColName | str) -> Pipeable:
    """
    Selects a subset of columns.

    :param cols:
        The columns to be included in the resulting table.

    Examples
    --------
    >>> t = pdt.Table({"a": [3, 2, 6, 4], "b": ["lll", "g", "u0", "__**_"]})
    >>> t >> select(t.a) >> show()
    shape: (4, 1)
    ┌─────┐
    │ a   │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 3   │
    │ 2   │
    │ 6   │
    │ 4   │
    └─────┘
    """

    errors.check_vararg_type(Col | ColName | str, "select", *cols)

    for col in cols:
        if isinstance(col, ColName | str) and col not in table:
            raise ColumnNotFoundError(f"column `{col.ast_repr()}` does not exist in table `{table._ast.short_name()}`")
        elif col not in table and col._uuid in table._cache.cols:
            raise ColumnNotFoundError(
                f"cannot select hidden column `{col.ast_repr()}` again\n"
                "hint: A column becomes hidden if you deselected it before or "
                "overwrite it in `mutate` or `summarize`."
            )

    cols = [ColName(col) if isinstance(col, str) else col for col in cols]

    new = copy.copy(table)
    new._ast = Select(table._ast, [preprocess_arg(col, table) for col in cols])

    return new


@overload
def drop(*cols: Col | ColName | str) -> Pipeable: ...


@verb
def drop(table: Table, *cols: Col | ColName | str) -> Pipeable:
    """
    Removes a subset of the columns.

    :param cols:
        The columns to be removed.

    Examples
    --------
    >>> t = pdt.Table({"a": [3, 2, 6, 4], "b": ["lll", "g", "u0", "__**_"]})
    >>> t >> drop(t.a) >> show()
    shape: (4, 1)
    ┌───────┐
    │ b     │
    │ ---   │
    │ str   │
    ╞═══════╡
    │ lll   │
    │ g     │
    │ u0    │
    │ __**_ │
    └───────┘
    """
    errors.check_vararg_type(Col | ColName | str, "drop", *cols)
    cols = [ColName(col) if isinstance(col, str) else col for col in cols]

    dropped_uuids = {preprocess_arg(col, table)._uuid for col in cols}
    return table >> select(
        *(name for name, uid in table._cache.name_to_uuid.items() if uid not in dropped_uuids),
    )


@overload
def rename(name_map: dict[str, str]) -> Pipeable: ...


@verb
@modify_ast
def rename(table: Table, name_map: dict[str | Col | ColName, str]) -> Pipeable:
    """
    Renames columns.

    :param name_map:
        A dictionary assigning some columns new names.

    Examples
    --------
    Renaming one column:

    >>> t = pdt.Table({"a": [3, 2, 6, 4], "b": ["lll", "g", "u0", "__**_"]})
    >>> t >> rename({"a": "h"}) >> show()
    shape: (4, 2)
    ┌─────┬───────┐
    │ h   ┆ b     │
    │ --- ┆ ---   │
    │ i64 ┆ str   │
    ╞═════╪═══════╡
    │ 3   ┆ lll   │
    │ 2   ┆ g     │
    │ 6   ┆ u0    │
    │ 4   ┆ __**_ │
    └─────┴───────┘

    Here is a more subtle example: As long as there are no two equal column names in the
    result table, one can give names to columns that already exist in the table. In the
    following example, the names of columns *a* and *b* are swapped.

    >>> s = t >> rename({"a": "b", "b": "a"}) >> show()
    Table <unnamed>, backend: PolarsImpl
    shape: (4, 2)
    ┌─────┬───────┐
    │ b   ┆ a     │
    │ --- ┆ ---   │
    │ i64 ┆ str   │
    ╞═════╪═══════╡
    │ 3   ┆ lll   │
    │ 2   ┆ g     │
    │ 6   ┆ u0    │
    │ 4   ┆ __**_ │
    └─────┴───────┘

    When using the column ``t.a`` in an expression derived from *s* now, it
    still refers to the same column, which now has the name *b*. The anonymous
    column ``C.a``, however, refers to the column with name *a* in the *current*
    table.

    >>> s >> mutate(u=t.a, v=C.a) >> show()
    shape: (4, 4)
    ┌─────┬───────┬─────┬───────┐
    │ b   ┆ a     ┆ u   ┆ v     │
    │ --- ┆ ---   ┆ --- ┆ ---   │
    │ i64 ┆ str   ┆ i64 ┆ str   │
    ╞═════╪═══════╪═════╪═══════╡
    │ 3   ┆ lll   ┆ 3   ┆ lll   │
    │ 2   ┆ g     ┆ 2   ┆ g     │
    │ 6   ┆ u0    ┆ 6   ┆ u0    │
    │ 4   ┆ __**_ ┆ 4   ┆ __**_ │
    └─────┴───────┴─────┴───────┘
    """
    errors.check_arg_type(dict, "rename", "name_map", name_map)

    for v in name_map.values():
        if not isinstance(v, str):
            raise TypeError(
                f"values in the `name_map` of `rename` must have type `str`, found `{v.__class__.__name__}` instead"
            )
    for k in name_map.keys():
        if not isinstance(k, str | Col | ColName):
            raise TypeError(
                "keys in the `name_map` of `rename` must have type `str | Col | "
                f"ColName`, found `{k.__class__.__name__}` instead"
            )

    name_map = {(preprocess_arg(k, table) if isinstance(k, ColName | Col) else k): v for k, v in name_map.items()}
    name_map = {(table._cache.uuid_to_name[k._uuid] if isinstance(k, Col) else k): v for k, v in name_map.items()}

    if d := set(name_map).difference(table._cache.name_to_uuid):
        raise ValueError(f"no column with name `{next(iter(d))}` in table `{table._ast.short_name()}`")

    if d := (set(table._cache.name_to_uuid).difference(name_map)) & set(name_map.values()):
        raise ValueError(f"rename would cause duplicate column name `{next(iter(d))}`")

    new = copy.copy(table)
    new._ast = Rename(table._ast, name_map)

    return new


@overload
def mutate(**kwargs: ColExpr) -> Pipeable: ...


@verb
@modify_ast
def mutate(table: Table, **kwargs: ColExpr) -> Pipeable:
    """
    Adds new columns to the table.

    :param kwargs:
        Each key is the name of a new column, and its value is the column
        expression defining the new column.

    Examples
    --------
    >>> t1 = pdt.Table(
    ...     dict(a=[3, 1, 4, 1, 5, 9], b=[2.465, 0.22, -4.477, 10.8, -81.2, 0.0])
    ... )
    >>> t1 >> mutate(u=t1.a * t1.b) >> show()
    shape: (6, 3)
    ┌─────┬────────┬─────────┐
    │ a   ┆ b      ┆ u       │
    │ --- ┆ ---    ┆ ---     │
    │ i64 ┆ f64    ┆ f64     │
    ╞═════╪════════╪═════════╡
    │ 3   ┆ 2.465  ┆ 7.395   │
    │ 1   ┆ 0.22   ┆ 0.22    │
    │ 4   ┆ -4.477 ┆ -17.908 │
    │ 1   ┆ 10.8   ┆ 10.8    │
    │ 5   ┆ -81.2  ┆ -406.0  │
    │ 9   ┆ 0.0    ┆ 0.0     │
    └─────┴────────┴─────────┘
    """

    names, values = list(kwargs.keys()), list(kwargs.values())
    uuids = [uuid.uuid1() for _ in names]
    new = copy.copy(table)
    preprocessed = []
    for name, val in zip(names, values, strict=True):
        try:
            preprocessed.append(preprocess_arg(val, table))
        except ErrorWithSource as e:
            raise type(e)(
                e.args[0] + f"\noccurred in `mutate` argument `{name}`\n"
                f"AST path to error source: {get_ast_path_str(val, e.source)}"
            ) from e
    new._ast = Mutate(table._ast, names, preprocessed, uuids)

    return new


@overload
def filter(*predicates: ColExpr[Bool]) -> Pipeable: ...


@verb
@modify_ast
def filter(table: Table, *predicates: ColExpr[Bool]) -> Pipeable:
    """
    Selects a subset of rows based on some condition.

    :param predicates:
        Column expressions of boolean type to filter by. Only rows where all expressions
        are true are included in the result.

    Examples
    --------
    >>> t = pdt.Table({"a": [3, 2, 6, 4], "b": ["lll", "g", "u0", "__**_"]})
    >>> t >> filter(t.a <= 4, ~t.b.str.contains("_")) >> show()
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 3   ┆ lll │
    │ 2   ┆ g   │
    └─────┴─────┘
    """
    new = copy.copy(table)
    preprocessed = []
    for i, pred in enumerate(predicates, 1):
        try:
            preprocessed.append(preprocess_arg(pred, table))
        except ErrorWithSource as e:
            raise type(e)(
                e.args[0] + f"\noccurred in {i}-{ordinal_suffix(i)} `filter` argument\n"
                f"AST path to error source: {get_ast_path_str(pred, e.source)}"
            ) from e

    new._ast = Filter(table._ast, preprocessed)

    for cond in new._ast.predicates:
        if not types.without_const(cond.dtype()) == Bool():
            raise DataTypeError(
                "predicates given to `filter` must be of boolean type.\n"
                f"hint: {cond} is of type {cond.dtype()} instead."
            )

        for fn in cond.iter_subtree_postorder():
            if isinstance(fn, ColFn) and fn.op.ftype in (
                Ftype.WINDOW,
                Ftype.AGGREGATE,
            ):
                raise FunctionTypeError(
                    f"forbidden window function `{fn.op.name}` in `filter`\nhint: If "
                    "you want to filter by an expression containing a window / "
                    "aggregation function, first add the expression as a column via "
                    "`mutate`."
                )

    return new


@overload
def arrange(*order_by: ColExpr) -> Pipeable: ...


@verb
@modify_ast
def arrange(table: Table, by: ColExpr, *more_by: ColExpr) -> Pipeable:
    """
    Sorts the rows of the table.

    :param order_by:
        Column expressions to sort by. The order of the expressions determines
        the priority.

    Examples
    --------
    >>> t = pdt.Table(
    ...     {
    ...         "r": [2, 7, 3, 2, 6, None, 4],
    ...         "s": ["l", "o", "a", "c", "s", "---", "3"],
    ...         "p": [0.655, -4.33, None, 143.6, 0.0, 1.0, 4.5],
    ...     }
    ... )
    >>> t >> arrange(t.r.nulls_first(), t.p) >> show()
    shape: (7, 3)
    ┌──────┬─────┬───────┐
    │ r    ┆ s   ┆ p     │
    │ ---  ┆ --- ┆ ---   │
    │ i64  ┆ str ┆ f64   │
    ╞══════╪═════╪═══════╡
    │ null ┆ --- ┆ 1.0   │
    │ 2    ┆ l   ┆ 0.655 │
    │ 2    ┆ c   ┆ 143.6 │
    │ 3    ┆ a   ┆ null  │
    │ 4    ┆ 3   ┆ 4.5   │
    │ 6    ┆ s   ┆ 0.0   │
    │ 7    ┆ o   ┆ -4.33 │
    └──────┴─────┴───────┘
    >>> t >> arrange(t.p.nulls_last().descending(), t.s) >> show()
    shape: (7, 3)
    ┌──────┬─────┬───────┐
    │ r    ┆ s   ┆ p     │
    │ ---  ┆ --- ┆ ---   │
    │ i64  ┆ str ┆ f64   │
    ╞══════╪═════╪═══════╡
    │ 2    ┆ c   ┆ 143.6 │
    │ 4    ┆ 3   ┆ 4.5   │
    │ null ┆ --- ┆ 1.0   │
    │ 2    ┆ l   ┆ 0.655 │
    │ 6    ┆ s   ┆ 0.0   │
    │ 7    ┆ o   ┆ -4.33 │
    │ 3    ┆ a   ┆ null  │
    └──────┴─────┴───────┘
    """

    order_by = [ColName(col) if isinstance(col, str) else col for col in (by, *more_by)]
    new = copy.copy(table)

    preprocessed = []
    for i, ord in enumerate(order_by, 1):
        try:
            preprocessed.append(preprocess_arg(Order.from_col_expr(ord), table))
        except ErrorWithSource as e:
            raise type(e)(
                e.args[0] + f"\noccurred in {i}-{ordinal_suffix(i)} `arrange` argument\n"
                f"AST path to error source: {get_ast_path_str(ord, e.source)}"
            ) from e

    new._ast = Arrange(table._ast, preprocessed)

    return new


@overload
def group_by(table: Table, *cols: Col | ColName | str, add=False) -> Pipeable: ...


@verb
@modify_ast
def group_by(table: Table, *cols: Col | ColName | str, add=False) -> Pipeable:
    """
    Add a grouping state to the table.

    :param cols:
        The columns to group by.

    :param add:
        If ``add=True``, the given columns are added to the set of columns the table is
        currently grouped by. If ``add=False``, the current grouping state is replaced
        by the given columns.

    This verb does not modify the table itself, but only adds a grouping in the
    background. The number of rows is only reduced when
    :doc:`summarize <pydiverse.transform.summarize>`
    is called. The
    :doc:`ungroup <pydiverse.transform.ungroup>`
    verb can be used to clear the grouping state.
    """
    errors.check_vararg_type(Col | ColName | str, "group_by", *cols)
    errors.check_arg_type(bool, "group_by", "add", add)
    cols = [ColName(col) if isinstance(col, str) else col for col in cols]

    for col in cols:
        if isinstance(col, Col) and col._uuid not in table._cache.uuid_to_name:
            raise ValueError(f"cannot group by non-selected column `{col.ast_repr()}`")

    new = copy.copy(table)
    new._ast = GroupBy(table._ast, [preprocess_arg(col, table) for col in cols], add)

    return new


@overload
def ungroup() -> Pipeable: ...


@verb
@modify_ast
def ungroup(table: Table) -> Pipeable:
    """
    Clear the grouping state of the table.

    Examples
    --------
    In the following example, ``group_by`` and ``ungroup`` are used to specify that each
    aggregation function between them uses the column ``t.c`` for grouping.

    >>> t = pdt.Table(
    ...     {
    ...         "a": [1.2, 5.077, -2.29, -0.0, 3.0, -7.7],
    ...         "b": ["a  ", "transform", "pipedag", "cdegh", "  -ade ", "  pq"],
    ...         "c": [True, False, None, None, True, True],
    ...         "d": [4, 4, 2, 0, 1, 0],
    ...     }
    ... )
    >>> (
    ...     t
    ...     >> group_by(t.c)
    ...     >> mutate(
    ...         u=t.b.str.len().max() + t.a.min(),
    ...         v=t.d.mean(filter=t.a >= 0),
    ...     )
    ...     >> ungroup()
    ...     >> show()
    ... )
    shape: (6, 6)
    ┌───────┬───────────┬───────┬─────┬────────┬─────┐
    │ a     ┆ b         ┆ c     ┆ d   ┆ u      ┆ v   │
    │ ---   ┆ ---       ┆ ---   ┆ --- ┆ ---    ┆ --- │
    │ f64   ┆ str       ┆ bool  ┆ i64 ┆ f64    ┆ f64 │
    ╞═══════╪═══════════╪═══════╪═════╪════════╪═════╡
    │ 1.2   ┆ a         ┆ true  ┆ 4   ┆ -0.7   ┆ 2.5 │
    │ 5.077 ┆ transform ┆ false ┆ 4   ┆ 14.077 ┆ 4.0 │
    │ -2.29 ┆ pipedag   ┆ null  ┆ 2   ┆ 4.71   ┆ 0.0 │
    │ -0.0  ┆ cdegh     ┆ null  ┆ 0   ┆ 4.71   ┆ 0.0 │
    │ 3.0   ┆   -ade    ┆ true  ┆ 1   ┆ -0.7   ┆ 2.5 │
    │ -7.7  ┆   pq      ┆ true  ┆ 0   ┆ -0.7   ┆ 2.5 │
    └───────┴───────────┴───────┴─────┴────────┴─────┘
    """
    new = copy.copy(table)
    new._ast = Ungroup(table._ast)

    return new


@overload
def summarize(**kwargs: ColExpr) -> Pipeable: ...


@verb
@modify_ast
def summarize(table: Table, **kwargs: ColExpr) -> Pipeable:
    """
    Computes aggregates over groups of rows.

    :param kwargs:
        Each key is the name of a new column, and its value is the column
        expression defining the new column. The column expression may not contain
        columns that are neither part of the grouping columns nor wrapped in an
        aggregation function.


    In contrast to :doc:`pydiverse.transform.mutate`, this verb in general reduces the
    number of rows and only keeps the grouping columns and the new columns defined in
    the kwargs. One row for each unique combination of values in the grouping columns
    is created.


    Examples
    --------
    >>> t = pdt.Table(
    ...     {
    ...         "a": [1.2, 5.077, -2.29, -0.0, 3.0, -7.7],
    ...         "b": ["a  ", "transform", "pipedag", "cdegh", "  -ade ", "  pq"],
    ...         "c": [True, False, None, None, True, True],
    ...     }
    ... )
    >>> (
    ...     t
    ...     >> group_by(t.c)
    ...     >> summarize(
    ...         u=t.b.str.len().mean(),
    ...         v=t.a.sum(filter=t.a >= 0),
    ...     )
    ...     >> show()
    ... )
    shape: (3, 3)
    ┌───────┬──────────┬───────┐
    │ c     ┆ u        ┆ v     │
    │ ---   ┆ ---      ┆ ---   │
    │ bool  ┆ f64      ┆ f64   │
    ╞═══════╪══════════╪═══════╡
    │ true  ┆ 4.666667 ┆ 4.2   │
    │ null  ┆ 6.0      ┆ -0.0  │
    │ false ┆ 9.0      ┆ 5.077 │
    └───────┴──────────┴───────┘
    """
    names, values = list(kwargs.keys()), list(kwargs.values())
    uuids = [uuid.uuid1() for _ in names]
    new = copy.copy(table)

    preprocessed = []
    for name, val in zip(names, values, strict=True):
        try:
            preprocessed.append(preprocess_arg(val, table, agg_is_window=False))
        except ErrorWithSource as e:
            raise type(e)(
                e.args[0] + f"\noccurred in `summarize` argument `{name}`\n"
                f"AST path to error source: {get_ast_path_str(val, e.source)}"
            ) from e
    new._ast = Summarize(table._ast, names, preprocessed, uuids)

    partition_by = set(table._cache.partition_by)

    if len(kwargs) == 0 and len(partition_by) == 0:
        raise ValueError("summarize without preceding group_by needs at least one column to summarize")

    def check_summarize_col_expr(expr: ColExpr, agg_fn_above: bool):
        if isinstance(expr, Col) and expr._uuid not in partition_by and not agg_fn_above:
            raise FunctionTypeError(
                f"column `{expr.ast_repr()}` is neither aggregated nor part of the grouping columns"
            )

        elif isinstance(expr, ColFn):
            if expr.op.ftype == Ftype.WINDOW:
                raise FunctionTypeError(f"forbidden window function `{expr.op.name}` in `summarize`")
            elif expr.op.ftype == Ftype.AGGREGATE and "partition_by" not in expr.context_kwargs:
                agg_fn_above = True

        for child in expr.iter_children():
            check_summarize_col_expr(child, agg_fn_above)

    for root in new._ast.values:
        check_summarize_col_expr(root, False)

    return new


@overload
def slice_head(n: int, *, offset: int = 0) -> Pipeable: ...


@verb
@modify_ast
def slice_head(table: Table, n: int, *, offset: int = 0) -> Pipeable:
    """
    Selects a subset of rows based on their index.

    :param n:
        The number of rows to select.

    :param offset:
        The index of the first row (0-based) that is included in the selection.


    Examples
    --------
    >>> t = pdt.Table(
    ...     {
    ...         "a": [65, 5, 312, -55, 0],
    ...         "b": ["l", "r", "srq", "---", " "],
    ...     }
    ... )
    >>> t >> slice_head(3, offset=1) >> show()
    shape: (3, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ str │
    ╞═════╪═════╡
    │ 5   ┆ r   │
    │ 312 ┆ srq │
    │ -55 ┆ --- │
    └─────┴─────┘
    """
    errors.check_arg_type(int, "slice_head", "n", n)
    errors.check_arg_type(int, "slice_head", "offset", offset)

    if table._cache.partition_by:
        raise ValueError("cannot apply `slice_head` to a grouped table")

    new = copy.copy(table)
    new._ast = SliceHead(table._ast, n, offset)

    return new


@overload
def join(
    right: Table,
    on: ColExpr[Bool] | str | list[ColExpr[Bool] | str],
    how: Literal["inner", "left", "full"],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable: ...


@verb
def join(
    left: Table,
    right: Table,
    on: ColExpr[Bool] | str | list[ColExpr[Bool] | str],
    how: Literal["inner", "left", "full"],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable:
    """
    Joins two tables on a boolean expression.

    The left table in the join comes through the pipe `>>` operator from the
    left.

    :param right:
        The right table to join with.

    :param on:
        The join condition. If this is a list, the elements are joined via AND. Strings
        are interpreted as an equality condition on the columns with that name. See the
        note below for more information on which expressions are allowed.

    :param how:
        The join type.

    :param validate:
        Only relevant for polars. When set to ``"m:m"``, this does nothing. If set to
        ``"1:m"``, it is checked whether each right row matches at most one left row. In
        case this does not hold, an error is raised. Symmetrically, if set to ``"m:1"``
        it is checked whether each left row matches at most one right row. If set to
        ``"1:1"`` both ``"1:m"`` and ``"m:1"`` are checked.

    :param suffix:
        A string that is appended to all column names from the right table. If no suffix
        is specified and there are no column name collisions, columns will retain their
        original name. If there are name collisions, the name of the right table is
        appended to all columns of the right table. If this still does not resolve all
        name collisions, additionally an integer is appended to the column names of the
        right table. If the join condition only contains equalities of columns of the
        same name, only the columns of the right table colliding with a column of the
        left table are renamed.

    Note
    ----
    Not all backends can handle arbitrary boolean expressions in ``on`` with every join
    type.

    :polars:
        For everything except conjunctions of equalities, it depends on whether polars
        ``join_asof`` can handle the join condition.
    :postgres:
        For full joins, the join condition must be hashable or mergeable. See the
        `postgres documentation <https://wiki.postgresql.org/wiki/PostgreSQL_vs_SQL_Standard#FULL_OUTER_JOIN_conditions>`_
        for more details.

    Tip
    ---
    Two tables cannot be joined if one is derived from the other. In particular, before
    a self-join, the ``alias`` verb has to be applied to one table.

    Examples
    --------
    >>> t1 = pdt.Table({"a": [3, 1, 4, 1, 5, 9, 4]}, name="t1")
    >>> t2 = pdt.Table({"a": [4, 4, 1, 7], "b": ["f", "g", "h", "i"]}, name="t2")
    >>> t1 >> join(t2, t1.a == t2.a, how="left") >> show()
    shape: (9, 3)
    ┌─────┬──────┬──────┐
    │ a   ┆ a_t2 ┆ b_t2 │
    │ --- ┆ ---  ┆ ---  │
    │ i64 ┆ i64  ┆ str  │
    ╞═════╪══════╪══════╡
    │ 3   ┆ null ┆ null │
    │ 1   ┆ 1    ┆ h    │
    │ 4   ┆ 4    ┆ f    │
    │ 4   ┆ 4    ┆ g    │
    │ 1   ┆ 1    ┆ h    │
    │ 5   ┆ null ┆ null │
    │ 9   ┆ null ┆ null │
    │ 4   ┆ 4    ┆ f    │
    │ 4   ┆ 4    ┆ g    │
    └─────┴──────┴──────┘
    """

    errors.check_arg_type(Table, "join", "right", right)
    errors.check_arg_type(ColExpr | list | str, "join", "on", on)
    errors.check_arg_type(str | None, "join", "suffix", suffix)
    errors.check_literal_type(["inner", "left", "full"], "join", "how", how)
    errors.check_literal_type(["1:1", "1:m", "m:1", "m:m"], "join", "validate", validate)

    if left._cache.backend != right._cache.backend:
        raise TypeError("cannot join two tables with different backends")

    if left._cache.partition_by:
        raise ValueError(f"cannot join grouped table `{left._ast.short_name()}`")
    elif right._cache.partition_by:
        raise ValueError(f"cannot join grouped table `{right._ast.short_name()}`")

    if intersection := left._cache.derived_from & right._cache.derived_from:
        raise ValueError(
            f"table `{list(intersection)[0].name}` occurs twice in the table "
            "tree.\nhint: To join two tables derived from a common table, "
            "apply `>> alias()` to one of them before the join."
        )

    user_suffix = suffix
    if suffix is None and right._ast.name is not None:
        suffix = f"_{right._ast.name}"
    if suffix is None:
        suffix = "_right"

    left_names = set(left._cache.uuid_to_name[col._uuid] for col in left)
    right_names = set(right._cache.uuid_to_name[col._uuid] for col in right)

    # Lambda column resolution and checks for existence of columns are done manually
    # here since we need to incorporate columns from the right.
    def _preprocess_on(expr: ColExpr):
        if isinstance(expr, ColName):
            if expr in left:
                if expr in right:
                    raise ValueError(
                        f"`{expr.ast_repr()}` is ambiguous here. Please use the source "
                        "table of the column to reference it."
                    )
                return left[expr.name]
            if expr not in right:
                raise ValueError(f"no column with name `{expr.name}` found in the left or right table")
            return right[expr.name]

        if isinstance(expr, Col) and expr._uuid not in left._cache.cols and expr._uuid not in right._cache.cols:
            raise ValueError(
                f"column `{expr.ast_repr()}` used in `on` neither exists in the table "
                f"`{left._ast.short_name()}` nor in the table "
                f"`{right._ast.short_name()}`. The source table "
                f"`{expr._ast.short_name()}` of the column must be an "
                "ancestor of one of the two input tables."
            )

        return expr

    # bring join condition in standard expression form
    if not isinstance(on, list):
        on = [on]
    on = [left[expr] == right[expr] if isinstance(expr, str) else expr for expr in on]

    on = [pred.map_subtree(_preprocess_on) for pred in on]
    for i, pred in enumerate(on, 1):
        try:
            dtype = pred.dtype()
            pred.ftype(agg_is_window=False)
        except ErrorWithSource as e:
            raise type(e)(
                e.args[0] + f"\noccurred in {i}-{ordinal_suffix(i)} element of `on`\n"
                f"AST path to error source: {get_ast_path_str(pred, e.source)}"
            ) from e

        if types.without_const(dtype) != Bool():
            raise DataTypeError(
                f"{i}-{ordinal_suffix(i)} element of `on` has type `{pred.dtype()}`, "
                "but join conditions must have boolean type"
            )

    # apply `rename` to the right table if necessary
    if user_suffix:
        for name in right._cache.name_to_uuid.keys():
            if name + suffix in left_names:
                raise ValueError(
                    f"column name `{name + suffix}` appears both in the left and right "
                    f"table using the user-provided suffix `{suffix}`\n"
                    "hint: Specify a different suffix to prevent name collisions or "
                    "no suffix for automatic name collision resolution."
                )

        # If the user actively specifies a suffix, we apply it to all right columns.
        right >>= rename({col: col.name + user_suffix for col in right})

    elif right_names & left_names:
        cnt = 0
        for name in right_names:
            suffixed = name + suffix + (f"_{cnt}" if cnt > 0 else "")
            while suffixed in left_names:
                cnt += 1
                suffixed = name + suffix + f"_{cnt}"

        if cnt > 0:
            suffix += f"_{cnt}"

        on_uuids = set(
            col._uuid for col in itertools.chain(*(pred.iter_subtree_preorder() for pred in on)) if isinstance(col, Col)
        )
        right_on_names = set(col.name for col in right if col._uuid in on_uuids)

        if not (right_names - right_on_names) & left_names:
            # If nothing except join columns clashes, we only rename the clashing
            # columns on the right.
            right >>= rename({col: col.name + suffix for col in right if col.name in left_names})

        else:
            right >>= rename({col: col.name + suffix for col in right})

    if len(on) == 0:
        on = LiteralCol(True)
    else:
        on = functools.reduce(operator.and_, on[1:], on[0])

    if how == "full" and not all(pred.op == ops.equal for pred in split_join_cond(on)):
        raise ValueError("in a `full` join, only equality predicates can be used")

    for fn in on.iter_subtree_postorder():
        if isinstance(fn, ColFn) and fn.op.ftype != Ftype.ELEMENT_WISE:
            raise FunctionTypeError(
                f"window function `{fn.op.name}` not allowed in `on`\n"
                "hint: First add the result of the window function to the table using "
                "`mutate`."
            )

    new = copy.copy(left)
    new._ast = Join(left._ast, right._ast, on, how, validate)

    new, left = check_subquery(new, left)
    new, right = check_subquery(new, right, is_right=True)

    new._cache = left._cache.update(new._ast, right_cache=right._cache)

    return new


# We define the join variations explicitly instead of via functools.partial since vscode
# gives functools.partial objects a different color than normal python functions which
# looks very confusing.
@overload
def inner_join(
    right: Table,
    on: ColExpr[Bool] | str | list[ColExpr[Bool] | str],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable: ...


@verb
def inner_join(
    left: Table,
    right: Table,
    on: ColExpr[Bool] | str | list[ColExpr[Bool] | str],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable:
    """
    Alias for the :doc:`pydiverse.transform.join` verb with ``how="inner"``.
    """

    return left >> join(right, on, "inner", validate=validate, suffix=suffix)


@overload
def left_join(
    right: Table,
    on: ColExpr[Bool] | str | list[ColExpr[Bool] | str],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable: ...


@verb
def left_join(
    left: Table,
    right: Table,
    on: ColExpr[Bool] | str | list[ColExpr[Bool] | str],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable:
    """
    Alias for the :doc:`pydiverse.transform.join` verb with ``how="left"``.
    """

    return left >> join(right, on, "left", validate=validate, suffix=suffix)


@overload
def full_join(
    right: Table,
    on: ColExpr[Bool] | str | list[ColExpr[Bool] | str],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable: ...


@verb
def full_join(
    left: Table,
    right: Table,
    on: ColExpr[Bool] | str | list[ColExpr[Bool] | str],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable:
    """
    Alias for the :doc:`pydiverse.transform.join` verb with ``how="full"``.
    """

    return left >> join(right, on, "full", validate=validate, suffix=suffix)


@overload
def cross_join(
    right: Table,
    *,
    suffix: str | None = None,
) -> Pipeable: ...


@verb
def cross_join(
    left: Table,
    right: Table,
    *,
    suffix: str | None = None,
) -> Pipeable:
    """
    Alias for the :doc:`pydiverse.transform.join` verb with an empty ``on`` clause.
    """

    return left >> join(right, how="inner", on=[], suffix=suffix)


@overload
def union(
    right: Table,
    *,
    distinct: bool = False,
) -> Pipeable: ...


@overload
def union(
    left: Table,
    right: Table,
    *,
    distinct: bool = False,
) -> Table: ...


def _union_impl(
    left: Table,
    right: Table,
    *,
    distinct: bool = False,
) -> Table:
    errors.check_arg_type(Table, "union", "right", right)

    if left._cache.backend != right._cache.backend:
        raise TypeError("cannot union two tables with different backends")

    if left._cache.partition_by:
        raise ValueError(f"cannot union grouped table `{left._ast.short_name()}`")
    elif right._cache.partition_by:
        raise ValueError(f"cannot union grouped table `{right._ast.short_name()}`")

    # Check that both tables have the same columns
    left_cols = set(left._cache.name_to_uuid.keys())
    right_cols = set(right._cache.name_to_uuid.keys())

    if left_cols != right_cols:
        missing_left = right_cols - left_cols
        missing_right = left_cols - right_cols
        error_msg = "tables must have the same columns for union"
        if missing_left:
            error_msg += f"\n  columns in right but not in left: {sorted(missing_left)}"
        if missing_right:
            error_msg += f"\n  columns in left but not in right: {sorted(missing_right)}"
        raise ValueError(error_msg)

    # Check column type compatibility using lca_type
    from pydiverse.transform._internal.tree.types import lca_type

    for col_name in left_cols:
        left_col = left._cache.cols[left._cache.name_to_uuid[col_name]]
        right_col = right._cache.cols[right._cache.name_to_uuid[col_name]]
        left_dtype = left_col.dtype()
        right_dtype = right_col.dtype()

        # Check if types are compatible by trying to find a common ancestor
        try:
            lca_type([left_dtype, right_dtype])
        except DataTypeError as e:
            raise TypeError(
                f"column '{col_name}' has incompatible types: left has {left_dtype}, right has {right_dtype}"
            ) from e

    new = copy.copy(left)
    new._ast = Union(left._ast, right._ast, distinct)

    new, left = check_subquery(new, left)
    new, right = check_subquery(new, right, is_right=True)

    new._cache = left._cache.update(new._ast, right_cache=right._cache)

    return new


def union(
    left_or_right: Table,
    right: Table | None = None,
    *,
    distinct: bool = False,
) -> Table | Pipeable:
    """
    Unions two tables by stacking rows vertically.

    The left table in the union may come through the pipe `>>` operator from the
    left.

    :param left_or_right:
        In case of tbl1 >> union(tbl2) this is the right table (tbl2).
        In case of union(tbl1, tbl2) this is the left table (tbl1).

    :param right:
        The right table to union with in case of union(tbl1, tbl2).

    :param distinct:
        If ``True``, performs UNION (removes duplicates). If ``False``,
        performs UNION ALL (keeps duplicates).

    Note
    ----
    Both tables must have the same number of columns with compatible types.
    Column names must match between the two tables.

    Examples
    --------
    >>> t1 = pdt.Table({"a": [1, 2, 3], "b": [4, 5, 6]}, name="t1")
    >>> t2 = pdt.Table({"a": [7, 8], "b": [9, 10]}, name="t2")
    >>> t1 >> union(t2) >> show()
    shape: (5, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 4   │
    │ 2   ┆ 5   │
    │ 3   ┆ 6   │
    │ 7   ┆ 9   │
    │ 8   ┆ 10  │
    └─────┴─────┘

    You can also call it in this form:
    >>> union(t1, t2) >> show()
    """
    # If called with two arguments directly (union(tbl1, tbl2)), return Table directly
    if right is not None:
        return _union_impl(left_or_right, right, distinct=distinct)  # returns Table

    # If called with one argument (union(tbl2)), behave like a verb for pipe syntax
    # Create a verb wrapper that will be called by the pipe operator
    # Note: left_or_right is the right table in this branch (since right is None)
    @verb
    def _union_verb(left_table: Table, right_table: Table, distinct: bool) -> Table:
        return _union_impl(left_table, right_table, distinct=distinct)

    return _union_verb(left_or_right, distinct=distinct)  # returns Pipeable


@overload
def show(pipe: bool = False) -> Pipeable | None: ...


@verb
def show(table: Table, pipe: bool = False) -> Pipeable | None:
    """
    Prints the table to stdout.

    :param pipe:
        If set to `True`, the table is returned, else `None` is returned.

    Note
    ----
    During interactive development in the python shell, you usually want to keep
    `pipe = False`, else the table is printed twice. The main use for `pipe = True` is
    when you don't want to break a long sequence of verbs in a file.
    """
    print(table)
    return table if pipe else None


@overload
def name() -> str | None: ...


@verb
def name(table: Table) -> str | None:
    """
    Returns the name of the table.
    """
    return table._ast.name


@overload
def columns() -> list[str]: ...


@verb
def columns(table: Table) -> list[str]:
    """
    Returns the column names of the table.
    """
    return list(table._cache.name_to_uuid.keys())


@overload
def ast_repr(verb_depth: int = 7, expr_depth: int = 2, pipe: bool = False) -> Pipeable | None: ...


@verb
def ast_repr(table: Table, verb_depth: int = 7, expr_depth: int = 2, *, pipe: bool = False) -> Pipeable | None:
    r"""
    Prints the AST of the table to stdout.

    :param pipe:
        If set to `True`, the table is returned, else `None` is returned.

    Note
    ----
    During interactive development in the python shell, you usually want to keep
    `pipe = False`, else the table is printed, too. The main use for `pipe = True` is
    when you don't want to break a long sequence of verbs in a file.

    Examples
    --------
    >>> tbl1 = Table(dict(a=[1, 2], b=["x", "y"]), name="tbl1")
    >>> tbl2 = Table(dict(a=[1, 2], c=["z", "zz"]), name="tbl2")
    >>> (
    ...     tbl1
    ...     >> mutate(u=C.a + 5)
    ...     >> select(tbl1.a)
    ...     >> left_join(
    ...         tbl2
    ...         >> arrange(tbl2.a)
    ...         >> filter(tbl2.c.str.len() <= 10)
    ...         >> full_join(s := tbl2 >> alias("s"), on="c"),
    ...         on="a",
    ...     )
    ...     >> slice_head(32)
    ...     >> ast_repr()
    ... )
    tbl2 = Table(df, Polars(), name="tbl2")
    tbl1 = Table(df, Polars(), name="tbl1")

    (
        tbl1
        >> mutate(u=tbl1.a + 5)
        >> select(tbl1.a)
        >> join(
            tbl2
            >> arrange(tbl2.a)
            >> filter(tbl2.c.str.len() <= 10)
            >> join(
                tbl2 >> alias("s") >> rename({"a": "a_s", "c": "c_s"}),
                how="full",
                on=tbl2.c == s.c,
                validate="m:m",
            )
            >> rename({"a": "a_tbl2"}),
            how="left",
            on=tbl1.a == tbl2.a,
            validate="m:m",
        )
        >> slice_head(n=32, offset=0)
    )
    """

    print(table._ast.ast_repr(verb_depth, expr_depth), end="")
    return table if pipe else None


def preprocess_arg(arg: ColExpr, table: Table, *, agg_is_window: bool = True) -> Any:
    arg = wrap_literals(arg)
    assert isinstance(arg, ColExpr | Order)

    def _preprocess_expr(expr: ColExpr, eval_aligned: bool = False):
        if isinstance(expr, Col) and expr._uuid not in table._cache.cols and not eval_aligned:
            raise ColumnNotFoundError(f"column `{expr.ast_repr()}` does not exist in table `{table._ast.name}`")

        if not eval_aligned and isinstance(expr, Series):
            raise TypeError(
                f"invalid use of series `{expr.ast_repr()}` in a column expression"
                "\nnote: polars / pandas series must be wrapped in `eval_aligned` for "
                "use in pydiverse.transform"
            )

        if (
            agg_is_window
            and isinstance(expr, ColFn)
            and "partition_by" not in expr.context_kwargs
            and (expr.op.ftype in (Ftype.WINDOW, Ftype.AGGREGATE))
        ):
            expr.context_kwargs["partition_by"] = [table._cache.cols[uid] for uid in table._cache.partition_by]

        if isinstance(expr, ColName):
            return table[expr.name]

        new = copy.copy(expr)
        new.map_children(
            functools.partial(
                _preprocess_expr,
                eval_aligned=eval_aligned | isinstance(expr, EvalAligned),
            )
        )

        # add casts for boolean add / sum
        # If we have more operations like these, which we want to map to other
        # operations on the AST, consider streamlining this in some special function
        if (
            isinstance(new, ColFn)
            and len(new.args) > 0
            and types.without_const(new.args[0].dtype()) == Bool()
            and new.op in (ops.add, ops.sum)
        ):
            new.args = [arg.cast(Int64) for arg in new.args]

        return new

    res = _preprocess_expr(arg)
    assert res.dtype() is not None
    assert res.ftype(agg_is_window=agg_is_window) is not None
    return res


def ordinal_suffix(n: int):
    return {1: "st", 2: "nd"}.get(n % 10, "th")
