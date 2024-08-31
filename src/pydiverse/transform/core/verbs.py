from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import pydiverse.transform.core.expressions.expressions as expressions
from pydiverse.transform.core.dispatchers import builtin_verb
from pydiverse.transform.core.dtypes import DType
from pydiverse.transform.core.expressions import (
    Col,
    ColName,
)
from pydiverse.transform.core.expressions.expressions import ColExpr, Order
from pydiverse.transform.core.util import (
    ordered_set,
    sign_peeler,
)

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

JoinHow = Literal["inner", "left", "outer"]

JoinValidate = Literal["1:1", "1:m", "m:1", "m:m"]


class TableExpr: ...


@dataclass
class Alias(TableExpr):
    table: TableExpr
    new_name: str | None


@dataclass
class Select(TableExpr):
    table: TableExpr
    selects: list[Col | ColName]


@dataclass
class Rename(TableExpr):
    table: TableExpr
    name_map: dict[str, str]


@dataclass
class Mutate(TableExpr):
    table: TableExpr
    names: list[str]
    values: list[ColExpr]


@dataclass
class Join(TableExpr):
    left: TableExpr
    right: TableExpr
    on: ColExpr
    how: JoinHow
    validate: JoinValidate
    suffix: str


@dataclass
class Filter(TableExpr):
    table: TableExpr
    filters: list[ColExpr]


@dataclass
class Summarise(TableExpr):
    table: TableExpr
    names: list[str]
    values: list[ColExpr]


@dataclass
class Arrange(TableExpr):
    table: TableExpr
    order_by: list[Order]


@dataclass
class SliceHead(TableExpr):
    table: TableExpr
    n: int
    offset: int


@dataclass
class GroupBy(TableExpr):
    table: TableExpr
    group_by: list[Col | ColName]
    add: bool


@dataclass
class Ungroup(TableExpr):
    table: TableExpr


def propagate_col_names(
    expr: TableExpr, needed_tables: set[TableExpr]
) -> tuple[dict[Col, ColName], list[ColName]]:
    if isinstance(expr, (Alias, SliceHead, Ungroup)):
        col_to_name, cols = propagate_col_names(expr.table, needed_tables)

    elif isinstance(expr, Select):
        needed_tables |= set(col.table for col in expr.selects if isinstance(col, Col))
        col_to_name, cols = propagate_col_names(expr.table, needed_tables)
        expr.selects = [
            col_to_name[col] if col in col_to_name else col for col in expr.selects
        ]

    elif isinstance(expr, Rename):
        col_to_name, cols = propagate_col_names(expr.table, needed_tables)
        col_to_name = {
            col: ColName(expr.name_map[col_name.name])
            if col_name.name in expr.name_map
            else col_name
            for col, col_name in col_to_name
        }

    elif isinstance(expr, (Mutate, Summarise)):
        for v in expr.values:
            needed_tables |= expressions.get_needed_tables(v)
        col_to_name, cols = propagate_col_names(expr.table, needed_tables)
        expr.values = [
            expressions.propagate_col_names(v, col_to_name) for v in expr.values
        ]
        cols.extend(Col(name, expr) for name in expr.names)

    elif isinstance(expr, Join):
        for v in expr.on:
            needed_tables |= expressions.get_needed_tables(v)
        col_to_name_left, cols_left = propagate_col_names(expr.left, needed_tables)
        col_to_name_right, cols_right = propagate_col_names(expr.right, needed_tables)
        col_to_name = col_to_name_left | col_to_name_right
        cols = cols_left + [ColName(col.name + expr.suffix) for col in cols_right]
        expr.on = [expressions.propagate_col_names(v, col_to_name) for v in expr.on]

    elif isinstance(expr, Filter):
        for v in expr.filters:
            needed_tables |= expressions.get_needed_tables(v)
        col_to_name, cols = propagate_col_names(expr.table, needed_tables)
        expr.filters = [
            expressions.propagate_col_names(v, col_to_name) for v in expr.filters
        ]

    elif isinstance(expr, Arrange):
        for v in expr.order_by:
            needed_tables |= expressions.get_needed_tables(v)
        col_to_name, cols = propagate_col_names(expr.table, needed_tables)
        expr.order_by = [
            Order(
                expressions.propagate_col_names(order.order_by, col_to_name),
                order.descending,
                order.nulls_last,
            )
            for order in expr.order_by
        ]

    elif isinstance(expr, GroupBy):
        for v in expr.group_by:
            needed_tables |= expressions.get_needed_tables(v)
        col_to_name, cols = propagate_col_names(expr.table, needed_tables)
        expr.group_by = [
            expressions.propagate_col_names(v, col_to_name) for v in expr.group_by
        ]

    else:
        raise TypeError

    if expr in needed_tables:
        col_to_name |= {Col(col.name, expr): ColName(col.name) for col in cols}
    return col_to_name, cols


def propagate_types(expr: TableExpr) -> dict[ColName, DType]:
    if isinstance(
        expr, (Alias, SliceHead, Ungroup, Select, Rename, SliceHead, GroupBy)
    ):
        return propagate_types(expr.table)

    elif isinstance(expr, (Mutate, Summarise)):
        col_types = propagate_types(expr.table)
        expr.values = [expressions.propagate_types(v, col_types) for v in expr.values]
        col_types.update(
            {ColName(name): value._type for name, value in zip(expr.names, expr.values)}
        )
        return col_types

    elif isinstance(expr, Join):
        col_types_left = propagate_types(expr.left)
        col_types_right = {
            ColName(name + expr.suffix): col_type
            for name, col_type in propagate_types(expr.right)
        }
        return col_types_left | col_types_right

    elif isinstance(expr, Filter):
        col_types = propagate_types(expr.table)
        expr.filters = [expressions.propagate_types(v, col_types) for v in expr.filters]
        return col_types

    elif isinstance(expr, Arrange):
        col_types = propagate_types(expr.table)
        expr.order_by = [
            expressions.propagate_types(v, col_types) for v in expr.order_by
        ]
        return col_types

    else:
        raise TypeError


@builtin_verb()
def alias(table: TableExpr, new_name: str | None = None):
    return Alias(table, new_name)


@builtin_verb()
def collect(table: TableExpr):
    return table.collect()


@builtin_verb()
def export(table: TableExpr):
    table._validate_verb_level()


@builtin_verb()
def build_query(table: TableExpr):
    return table.build_query()


@builtin_verb()
def show_query(table: TableExpr):
    if query := table.build_query():
        print(query)
    else:
        print(f"No query to show for {type(table).__name__}")

    return table


@builtin_verb()
def select(table: TableExpr, *args: Col | ColName):
    return Select(table, list(args))
    if len(args) == 1 and args[0] is Ellipsis:
        # >> select(...)  ->  Select all columns
        args = [
            table.cols[uuid].as_column(name, table)
            for name, uuid in table.named_cols.fwd.items()
        ]

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

        if not isinstance(col, (Col, ColName)):
            raise TypeError(
                "Arguments to select verb must be of type `Col`'"
                f" and not {type(col)}."
            )
        cols.append(col)

    selects = []
    for col in cols:
        if isinstance(col, Col):
            selects.append(table.named_cols.bwd[col.uuid])
        elif isinstance(col, ColName):
            selects.append(col.name)

    # Invert selection
    if positive_selection is False:
        exclude = set(selects)
        selects.clear()
        for name in table.selects:
            if name not in exclude:
                selects.append(name)

    new_tbl = table.copy()
    new_tbl.preverb_hook("select", *args)
    new_tbl.selects = ordered_set(selects)
    new_tbl.select(*args)
    return new_tbl


@builtin_verb()
def rename(table: TableExpr, name_map: dict[str, str]):
    return Rename(table, name_map)
    # Type check
    for k, v in name_map.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError(
                f"Key and Value of `name_map` must both be strings: ({k!r}, {v!r})"
            )

    # Reference col that doesn't exist
    if missing_cols := name_map.keys() - table.named_cols.fwd.keys():
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
    unmodified_cols = table.named_cols.fwd.keys() - name_map.keys()
    if duplicate_names := unmodified_cols & set(name_map.values()):
        raise ValueError(
            "Table already contains columns named: " + ", ".join(duplicate_names)
        )

    # Rename
    new_tbl = table.copy()
    new_tbl.selects = ordered_set(name_map.get(name, name) for name in new_tbl.selects)

    uuid_name_map = {new_tbl.named_cols.fwd[old]: new for old, new in name_map.items()}
    for uuid in uuid_name_map:
        del new_tbl.named_cols.bwd[uuid]
    for uuid, name in uuid_name_map.items():
        new_tbl.named_cols.bwd[uuid] = name

    return new_tbl


@builtin_verb()
def mutate(table: TableExpr, **kwargs: ColExpr):
    return Mutate(table, list(kwargs.keys()), list(kwargs.values()))


@builtin_verb()
def join(
    left: TableExpr,
    right: TableExpr,
    on: ColExpr,
    how: Literal["inner", "left", "outer"],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,  # appended to cols of the right table
):
    return Join(left, right, on, how, validate, suffix)


inner_join = functools.partial(join, how="inner")
left_join = functools.partial(join, how="left")
outer_join = functools.partial(join, how="outer")


@builtin_verb()
def filter(table: TableExpr, *args: ColExpr):
    return Filter(table, list(args))


@builtin_verb()
def arrange(table: TableExpr, *args: ColExpr):
    return Arrange(table, list(Order.from_col_expr(arg) for arg in args))


@builtin_verb()
def group_by(table: TableExpr, *args: Col | ColName, add=False):
    return GroupBy(table, list(args), add)


@builtin_verb()
def ungroup(table: TableExpr):
    return Ungroup(table)


@builtin_verb()
def summarise(table: TableExpr, **kwargs: ColExpr):
    return Summarise(table, list(kwargs.keys()), list(kwargs.values()))


@builtin_verb()
def slice_head(table: TableExpr, n: int, *, offset: int = 0):
    return SliceHead(table, n, offset)
