from __future__ import annotations

import dataclasses
from typing import Literal

from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import col_expr
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColName, Order
from pydiverse.transform.tree.dtypes import DType
from pydiverse.transform.tree.table_expr import TableExpr

JoinHow = Literal["inner", "left", "outer"]

JoinValidate = Literal["1:1", "1:m", "m:1", "m:m"]


@dataclasses.dataclass(eq=False)
class Alias(TableExpr):
    table: TableExpr
    new_name: str | None


@dataclasses.dataclass(eq=False)
class Select(TableExpr):
    table: TableExpr
    selects: list[Col | ColName]


@dataclasses.dataclass(eq=False)
class Rename(TableExpr):
    table: TableExpr
    name_map: dict[str, str]


@dataclasses.dataclass(eq=False)
class Mutate(TableExpr):
    table: TableExpr
    names: list[str]
    values: list[ColExpr]


@dataclasses.dataclass(eq=False)
class Join(TableExpr):
    left: TableExpr
    right: TableExpr
    on: ColExpr
    how: JoinHow
    validate: JoinValidate
    suffix: str


@dataclasses.dataclass(eq=False)
class Filter(TableExpr):
    table: TableExpr
    filters: list[ColExpr]


@dataclasses.dataclass(eq=False)
class Summarise(TableExpr):
    table: TableExpr
    names: list[str]
    values: list[ColExpr]


@dataclasses.dataclass(eq=False)
class Arrange(TableExpr):
    table: TableExpr
    order_by: list[Order]


@dataclasses.dataclass(eq=False)
class SliceHead(TableExpr):
    table: TableExpr
    n: int
    offset: int


@dataclasses.dataclass(eq=False)
class GroupBy(TableExpr):
    table: TableExpr
    group_by: list[Col | ColName]
    add: bool


@dataclasses.dataclass(eq=False)
class Ungroup(TableExpr):
    table: TableExpr


# returns Col -> ColName mapping and the list of available columns
def propagate_names(expr: TableExpr, needed_cols: set[Col]) -> dict[Col, ColName]:
    if isinstance(expr, (Alias, SliceHead, Ungroup)):
        col_to_name = propagate_names(expr.table, needed_cols)

    elif isinstance(expr, Select):
        needed_cols |= set(col.table for col in expr.selects if isinstance(col, Col))
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.selects = [
            col_to_name[col] if col in col_to_name else col for col in expr.selects
        ]

    elif isinstance(expr, Rename):
        col_to_name = propagate_names(expr.table, needed_cols)
        col_to_name = {
            col: ColName(expr.name_map[col_name.name])
            if col_name.name in expr.name_map
            else col_name
            for col, col_name in col_to_name
        }

    elif isinstance(expr, Mutate):
        for v in expr.values:
            needed_cols |= col_expr.get_needed_cols(v)
        col_to_name = propagate_names(expr.table, needed_cols)
        # overwritten columns still need to be stored since the user may access them
        # later. They're not in the C-space anymore, however, so we give them
        # {name}_{hash of the previous table} as a dummy name.
        overwritten = set(
            name for name in expr.names if Col(expr, name) in set(needed_cols)
        )
        col_to_name = {
            col: ColName(col_name.name + str(hash(expr.table)))
            if col_name.name in overwritten
            else col_name
            for col, col_name in col_to_name.items()
        }
        expr.values = [col_expr.propagate_names(v, col_to_name) for v in expr.values]

    elif isinstance(expr, Join):
        for v in expr.on:
            needed_cols |= col_expr.get_needed_cols(v)
        col_to_name_left, cols_left = propagate_names(expr.left, needed_cols)
        col_to_name_right, cols_right = propagate_names(expr.right, needed_cols)
        col_to_name = col_to_name_left | col_to_name_right
        expr.on = [propagate_names(v, col_to_name) for v in expr.on]

    elif isinstance(expr, Filter):
        for v in expr.filters:
            needed_cols |= col_expr.get_needed_cols(v)
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.filters = [propagate_names(v, col_to_name) for v in expr.filters]

    elif isinstance(expr, Arrange):
        for v in expr.order_by:
            needed_cols |= col_expr.get_needed_cols(v)
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.order_by = [
            Order(
                propagate_names(order.order_by, col_to_name),
                order.descending,
                order.nulls_last,
            )
            for order in expr.order_by
        ]

    elif isinstance(expr, GroupBy):
        for v in expr.group_by:
            needed_cols |= col_expr.get_needed_cols(v)
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.group_by = [propagate_names(v, col_to_name) for v in expr.group_by]

    elif isinstance(expr, Summarise):
        for v in expr.values:
            needed_cols |= col_expr.get_needed_cols(v)
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.values = [propagate_names(v, col_to_name) for v in expr.values]

    elif isinstance(expr, Table):
        col_to_name = dict()

    else:
        raise TypeError

    for col in needed_cols:
        if col.table == expr:
            col_to_name[col] = ColName(col.name)

    return col_to_name


def propagate_types(expr: TableExpr) -> dict[Col | ColName, DType]:
    if isinstance(
        expr, (Alias, SliceHead, Ungroup, Select, Rename, SliceHead, GroupBy)
    ):
        return propagate_types(expr.table)

    elif isinstance(expr, (Mutate, Summarise)):
        col_types = propagate_types(expr.table)
        expr.values = [col_expr.propagate_types(v, col_types) for v in expr.values]
        col_types.update(
            {name: value.dtype for name, value in zip(expr.names, expr.values)}
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
        expr.filters = [col_expr.propagate_types(v, col_types) for v in expr.filters]
        return col_types

    elif isinstance(expr, Arrange):
        col_types = propagate_types(expr.table)
        expr.order_by = [col_expr.propagate_types(v, col_types) for v in expr.order_by]
        return col_types

    elif isinstance(expr, Table):
        return expr.schema()

    else:
        raise TypeError
