from __future__ import annotations

import functools
from collections import ChainMap
from collections.abc import Iterable
from typing import Literal

from pydiverse.transform.core import dtypes
from pydiverse.transform.core.dispatchers import builtin_verb
from pydiverse.transform.core.expressions import (
    Column,
    LambdaColumn,
    SymbolicExpression,
)
from pydiverse.transform.core.expressions.util import iterate_over_expr
from pydiverse.transform.core.table_impl import AbstractTableImpl, ColumnMetaData
from pydiverse.transform.core.util import (
    bidict,
    ordered_set,
    sign_peeler,
    translate_ordering,
)
from pydiverse.transform.errors import ExpressionTypeError, FunctionTypeError
from pydiverse.transform.ops import OPType

__all__ = [
    "alias",
    "collect",
    "build_query",
    "show_query",
    "select",
    "rename",
    "mutate",
    "join",
    "left_join",
    "inner_join",
    "outer_join",
    "filter",
    "arrange",
    "group_by",
    "ungroup",
    "summarise",
    "slice_head",
    "export",
]


def check_cols_available(
    tables: AbstractTableImpl | Iterable[AbstractTableImpl],
    columns: set[Column],
    function_name: str,
):
    if isinstance(tables, AbstractTableImpl):
        tables = (tables,)
    available_columns = ChainMap(*(table.available_cols for table in tables))
    missing_columns = []
    for col in columns:
        if col.uuid not in available_columns:
            missing_columns.append(col)
    if missing_columns:
        missing_columns_str = ", ".join(map(lambda x: str(x), missing_columns))
        raise ValueError(
            f"Can't access column(s) {missing_columns_str} in {function_name}() because"
            " they aren't available in the input."
        )


def check_lambdas_valid(tbl: AbstractTableImpl, *expressions):
    lambdas = []
    for expression in expressions:
        lambdas.extend(
            lc for lc in iterate_over_expr(expression) if isinstance(lc, LambdaColumn)
        )
    missing_lambdas = {lc for lc in lambdas if lc.name not in tbl.named_cols.fwd}
    if missing_lambdas:
        missing_lambdas_str = ", ".join(map(lambda x: str(x), missing_lambdas))
        raise ValueError(f"Invalid lambda column(s) {missing_lambdas_str}.")


def cols_in_expression(expression) -> set[Column]:
    return {c for c in iterate_over_expr(expression) if isinstance(c, Column)}


def cols_in_expressions(expressions) -> set[Column]:
    if len(expressions) == 0:
        return set()
    return set.union(*(cols_in_expression(e) for e in expressions))


def validate_table_args(*tables):
    if len(tables) == 0:
        return

    for table in tables:
        if not isinstance(table, AbstractTableImpl):
            raise TypeError(f"Expected a TableImpl but got {type(table)} instead.")

    backend = type(tables[0])
    for table in tables:
        if type(table) is not backend:
            raise ValueError(
                f"Can't mix tables with different backends. Expected '{backend}' but"
                f" found '{type(table)}'."
            )


@builtin_verb()
def alias(tbl: AbstractTableImpl, name: str | None = None):
    """Creates a new table object with a different name and reassigns column UUIDs.
    Must be used before performing a self-join."""
    validate_table_args(tbl)
    return tbl.alias(name)


@builtin_verb()
def collect(tbl: AbstractTableImpl):
    validate_table_args(tbl)
    return tbl.collect()


@builtin_verb()
def export(tbl: AbstractTableImpl):
    validate_table_args(tbl)
    return tbl.export()


@builtin_verb()
def build_query(tbl: AbstractTableImpl):
    return tbl.build_query()


@builtin_verb()
def show_query(tbl: AbstractTableImpl):
    if query := tbl.build_query():
        print(query)
    else:
        print(f"No query to show for {type(tbl).__name__}")

    return tbl


@builtin_verb()
def select(tbl: AbstractTableImpl, *args: Column | LambdaColumn):
    if len(args) == 1 and args[0] is Ellipsis:
        # >> select(...)  ->  Select all columns
        args = [
            tbl.cols[uuid].as_column(name, tbl)
            for name, uuid in tbl.named_cols.fwd.items()
        ]

    # Validate input
    validate_table_args(tbl)
    check_cols_available(tbl, cols_in_expressions(args), "select")
    check_lambdas_valid(tbl, *args)

    cols = []
    positive_selection = None
    for col in args:
        col, is_pos = sign_peeler(col)
        if positive_selection is None:
            positive_selection = is_pos
        else:
            if is_pos is not positive_selection:
                raise ValueError(
                    "All columns in input must have the same sign."
                    " Can't mix selection with deselection."
                )

        if not isinstance(col, (Column, LambdaColumn)):
            raise TypeError(
                "Arguments to select verb must be of type 'Column' or 'LambdaColumn'"
                f" and not {type(col)}."
            )
        cols.append(col)

    selects = []
    for col in cols:
        if isinstance(col, Column):
            selects.append(tbl.named_cols.bwd[col.uuid])
        elif isinstance(col, LambdaColumn):
            selects.append(col.name)

    # Invert selection
    if positive_selection is False:
        exclude = set(selects)
        selects.clear()
        for name in tbl.selects:
            if name not in exclude:
                selects.append(name)

    new_tbl = tbl.copy()
    new_tbl.preverb_hook("select", *args)
    new_tbl.selects = ordered_set(selects)
    new_tbl.select(*args)
    return new_tbl


@builtin_verb()
def rename(tbl: AbstractTableImpl, name_map: dict[str, str]):
    # Type check
    for k, v in name_map.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError(
                f"Key and Value of `name_map` must both be strings: ({k!r}, {v!r})"
            )

    # Reference col that doesn't exist
    if missing_cols := name_map.keys() - tbl.named_cols.fwd.keys():
        raise KeyError("Table has no columns named: " + ", ".join(missing_cols))

    # Can't rename two cols to the same name
    _seen = set()
    if duplicate_names := {
        name for name in name_map.values() if name in _seen or _seen.add(name)
    }:
        raise ValueError(
            "Can't rename multiple columns to the same name: "
            + ", ".join(duplicate_names)
        )

    # Can't rename a column to one that already exists
    unmodified_cols = tbl.named_cols.fwd.keys() - name_map.keys()
    if duplicate_names := unmodified_cols & set(name_map.values()):
        raise ValueError(
            "Table already contains columns named: " + ", ".join(duplicate_names)
        )

    # Rename
    new_tbl = tbl.copy()
    new_tbl.selects = ordered_set(name_map.get(name, name) for name in new_tbl.selects)

    uuid_name_map = {new_tbl.named_cols.fwd[old]: new for old, new in name_map.items()}
    for uuid in uuid_name_map:
        del new_tbl.named_cols.bwd[uuid]
    for uuid, name in uuid_name_map.items():
        new_tbl.named_cols.bwd[uuid] = name

    return new_tbl


@builtin_verb()
def mutate(tbl: AbstractTableImpl, **kwargs: SymbolicExpression):
    validate_table_args(tbl)
    check_cols_available(tbl, cols_in_expressions(kwargs.values()), "mutate")

    new_tbl = tbl.copy()
    new_tbl.preverb_hook("mutate", **kwargs)
    kwargs = {k: new_tbl.resolve_lambda_cols(v) for k, v in kwargs.items()}

    for name, expr in kwargs.items():
        uid = Column.generate_col_uuid()
        col = ColumnMetaData.from_expr(uid, expr, new_tbl, verb="mutate")

        if dtypes.NoneDType().same_kind(col.dtype):
            raise ExpressionTypeError(
                f"Column '{name}' has an invalid type: {col.dtype}"
            )

        new_tbl.selects.add(name)
        new_tbl.named_cols.fwd[name] = uid
        new_tbl.available_cols.add(uid)
        new_tbl.cols[uid] = col

    new_tbl.mutate(**kwargs)
    return new_tbl


@builtin_verb()
def join(
    left: AbstractTableImpl,
    right: AbstractTableImpl,
    on: SymbolicExpression,
    how: Literal["inner", "left", "outer"],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,  # appended to cols of the right table
):
    validate_table_args(left, right)

    if left.grouped_by or right.grouped_by:
        raise ValueError("Can't join grouped tables. You first have to ungroup them.")

    # Check args only contains valid columns
    check_cols_available((left, right), cols_in_expression(on), "join")

    if how not in ("inner", "left", "outer"):
        raise ValueError(
            "join type must be one of 'inner', 'left' or 'outer' (value provided:"
            f" {how})"
        )

    new_left = left.copy()
    new_left.preverb_hook("join", right, on, how, validate=validate)

    if set(new_left.named_cols.fwd.values()) & set(right.named_cols.fwd.values()):
        raise ValueError(
            f"{how} join of `{left.name}` and `{right.name}` failed: "
            f"duplicate columns detected. If you want to do a self-join or join a "
            f"table twice, use `alias` on one table before the join."
        )

    if suffix is not None:
        # check that the user-provided suffix does not lead to collisions
        if collisions := set(new_left.named_cols.fwd.keys()) & set(
            name + suffix for name in right.named_cols.fwd.keys()
        ):
            raise ValueError(
                f"{how} join of `{left.name}` and `{right.name}` failed: "
                f"using the suffix `{suffix}` for right columns, the following column "
                f"names appear both in the left and right table: {collisions}"
            )
    else:
        # try `_{right.name}`, then `_{right.name}1`, `_{right.name}2` and so on
        cnt = 0
        suffix = "_" + right.name
        for rname in right.named_cols.fwd.keys():
            while rname + suffix in new_left.named_cols.fwd.keys():
                cnt += 1
                suffix = "_" + right.name + str(cnt)

    new_left.selects |= {name + suffix for name in right.selects}
    new_left.named_cols.fwd.update(
        {name + suffix: uuid for name, uuid in right.named_cols.fwd.items()}
    )
    new_left.available_cols.update(right.available_cols)
    new_left.cols.update(right.cols)

    # By resolving lambdas this late, we enable the user to use lambda columns
    # to reference mutated columns from the right side of the join.
    # -> `C.columnname_righttablename` is a valid lambda in the on condition.
    check_lambdas_valid(new_left, on)
    on = new_left.resolve_lambda_cols(on)

    new_left.join(right, on, how, validate=validate)
    return new_left


inner_join = functools.partial(join, how="inner")
left_join = functools.partial(join, how="left")
outer_join = functools.partial(join, how="outer")


@builtin_verb()
def filter(tbl: AbstractTableImpl, *args: SymbolicExpression):
    # TODO: Type check expression
    validate_table_args(tbl)
    check_cols_available(tbl, cols_in_expressions(args), "filter")
    args = [tbl.resolve_lambda_cols(arg) for arg in args]

    new_tbl = tbl.copy()
    new_tbl.preverb_hook("filter", *args)
    new_tbl.filter(*args)
    return new_tbl


@builtin_verb()
def arrange(tbl: AbstractTableImpl, *args: Column | LambdaColumn):
    if len(args) == 0:
        return tbl

    # Validate Input
    validate_table_args(tbl)
    check_cols_available(tbl, cols_in_expressions(args), "arrange")
    check_lambdas_valid(tbl, *args)

    ordering = translate_ordering(tbl, args)

    new_tbl = tbl.copy()
    new_tbl.preverb_hook("arrange", *args)
    new_tbl.arrange(ordering)
    return new_tbl


@builtin_verb()
def group_by(tbl: AbstractTableImpl, *args: Column | LambdaColumn, add=False):
    # Validate Input
    validate_table_args(tbl)
    check_cols_available(tbl, cols_in_expressions(args), "group_by")
    check_lambdas_valid(tbl, *args)

    # WARNING: Depending on the SQL backend, you might
    #          only be allowed to reference columns
    if not args:
        raise ValueError(
            "Expected columns to group by, but none were specified. To remove the"
            " grouping use the ungroup verb instead."
        )
    for col in args:
        if not isinstance(col, (Column, LambdaColumn)):
            raise TypeError(
                "Arguments to group_by verb must be of type 'Column' or 'LambdaColumn'"
                f" and not '{type(col)}'."
            )

    args = [tbl.resolve_lambda_cols(arg) for arg in args]

    new_tbl = tbl.copy()
    new_tbl.preverb_hook("group_by", *args, add=add)
    if add:
        new_tbl.grouped_by |= ordered_set(args)
    else:
        new_tbl.grouped_by = ordered_set(args)
    new_tbl.group_by(*args)
    return new_tbl


@builtin_verb()
def ungroup(tbl: AbstractTableImpl):
    """Remove all groupings from table."""
    validate_table_args(tbl)

    new_tbl = tbl.copy()
    new_tbl.preverb_hook("ungroup")
    new_tbl.grouped_by.clear()
    new_tbl.ungroup()
    return new_tbl


@builtin_verb()
def summarise(tbl: AbstractTableImpl, **kwargs: SymbolicExpression):
    # Validate Input
    validate_table_args(tbl)
    check_cols_available(tbl, cols_in_expressions(kwargs.values()), "summarise")

    new_tbl = tbl.copy()
    new_tbl.preverb_hook("summarise", **kwargs)
    kwargs = {k: new_tbl.resolve_lambda_cols(v) for k, v in kwargs.items()}

    # TODO: Validate that the functions are actually aggregating functions.
    ...

    # Calculate state for new table
    selects = ordered_set()
    named_cols = bidict()
    available_cols = set()
    cols = {}

    # Add grouping cols to beginning of select.
    for col in tbl.grouped_by:
        selects.add(tbl.named_cols.bwd[col.uuid])
        available_cols.add(col.uuid)
        named_cols.fwd[col.name] = col.uuid

    # Add summarizing cols to the end of the select.
    for name, expr in kwargs.items():
        if name in selects:
            raise ValueError(
                f"Column with name '{name}' already in select. The new summarised"
                " columns must have a different name than the grouping columns."
            )
        uid = Column.generate_col_uuid()
        col = ColumnMetaData.from_expr(uid, expr, new_tbl, verb="summarise")

        if dtypes.NoneDType().same_kind(col.dtype):
            raise ExpressionTypeError(
                f"Column '{name}' has an invalid type: {col.dtype}"
            )
        if col.ftype != OPType.AGGREGATE:
            raise FunctionTypeError(
                f"Expression for column '{name}' doesn't summarise any values."
            )

        selects.add(name)
        named_cols.fwd[name] = uid
        available_cols.add(uid)
        cols[uid] = col

    # Update new_tbl
    new_tbl.selects = ordered_set(selects)
    new_tbl.named_cols = named_cols
    new_tbl.available_cols = available_cols
    new_tbl.cols.update(cols)
    new_tbl.intrinsic_grouped_by = new_tbl.grouped_by.copy()
    new_tbl.summarise(**kwargs)

    # Reduce the grouping level by one -> drop last
    if len(new_tbl.grouped_by):
        new_tbl.grouped_by.pop_back()

    if len(new_tbl.grouped_by):
        new_tbl.group_by(*new_tbl.grouped_by)
    else:
        new_tbl.ungroup()

    return new_tbl


@builtin_verb()
def slice_head(tbl: AbstractTableImpl, n: int, *, offset: int = 0):
    validate_table_args(tbl)
    if not isinstance(n, int):
        raise TypeError("'n' must be an int")
    if not isinstance(offset, int):
        raise TypeError("'offset' must be an int")
    if n <= 0:
        raise ValueError(f"'n' must be a positive integer (value: {n})")
    if offset < 0:
        raise ValueError(f"'offset' can't be negative (value: {offset})")

    if tbl.grouped_by:
        raise ValueError("Can't slice table that is grouped. Must ungroup first.")

    new_tbl = tbl.copy()
    new_tbl.preverb_hook("slice_head")
    new_tbl.slice_head(n, offset)
    return new_tbl
