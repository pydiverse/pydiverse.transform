from __future__ import annotations

import dataclasses
from typing import Literal

from pydiverse.transform.tree import col_expr
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColName, Order
from pydiverse.transform.tree.dtypes import DType
from pydiverse.transform.tree.table_expr import TableExpr

JoinHow = Literal["inner", "left", "outer"]

JoinValidate = Literal["1:1", "1:m", "m:1", "m:m"]


@dataclasses.dataclass
class Alias(TableExpr):
    table: TableExpr
    new_name: str | None


@dataclasses.dataclass
class Select(TableExpr):
    table: TableExpr
    selects: list[Col | ColName]


@dataclasses.dataclass
class Rename(TableExpr):
    table: TableExpr
    name_map: dict[str, str]


@dataclasses.dataclass
class Mutate(TableExpr):
    table: TableExpr
    names: list[str]
    values: list[ColExpr]


@dataclasses.dataclass
class Join(TableExpr):
    left: TableExpr
    right: TableExpr
    on: ColExpr
    how: JoinHow
    validate: JoinValidate
    suffix: str


@dataclasses.dataclass
class Filter(TableExpr):
    table: TableExpr
    filters: list[ColExpr]


@dataclasses.dataclass
class Summarise(TableExpr):
    table: TableExpr
    names: list[str]
    values: list[ColExpr]


@dataclasses.dataclass
class Arrange(TableExpr):
    table: TableExpr
    order_by: list[Order]


@dataclasses.dataclass
class SliceHead(TableExpr):
    table: TableExpr
    n: int
    offset: int


@dataclasses.dataclass
class GroupBy(TableExpr):
    table: TableExpr
    group_by: list[Col | ColName]
    add: bool


@dataclasses.dataclass
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
            needed_tables |= col_expr.get_needed_tables(v)
        col_to_name, cols = propagate_col_names(expr.table, needed_tables)
        expr.values = [
            col_expr.propagate_col_names(v, col_to_name) for v in expr.values
        ]
        cols.extend(Col(name, expr) for name in expr.names)

    elif isinstance(expr, Join):
        for v in expr.on:
            needed_tables |= col_expr.get_needed_tables(v)
        col_to_name_left, cols_left = propagate_col_names(expr.left, needed_tables)
        col_to_name_right, cols_right = propagate_col_names(expr.right, needed_tables)
        col_to_name = col_to_name_left | col_to_name_right
        cols = cols_left + [ColName(col.name + expr.suffix) for col in cols_right]
        expr.on = [col_expr.propagate_col_names(v, col_to_name) for v in expr.on]

    elif isinstance(expr, Filter):
        for v in expr.filters:
            needed_tables |= col_expr.get_needed_tables(v)
        col_to_name, cols = propagate_col_names(expr.table, needed_tables)
        expr.filters = [
            col_expr.propagate_col_names(v, col_to_name) for v in expr.filters
        ]

    elif isinstance(expr, Arrange):
        for v in expr.order_by:
            needed_tables |= col_expr.get_needed_tables(v)
        col_to_name, cols = propagate_col_names(expr.table, needed_tables)
        expr.order_by = [
            Order(
                col_expr.propagate_col_names(order.order_by, col_to_name),
                order.descending,
                order.nulls_last,
            )
            for order in expr.order_by
        ]

    elif isinstance(expr, GroupBy):
        for v in expr.group_by:
            needed_tables |= col_expr.get_needed_tables(v)
        col_to_name, cols = propagate_col_names(expr.table, needed_tables)
        expr.group_by = [
            col_expr.propagate_col_names(v, col_to_name) for v in expr.group_by
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
        expr.values = [col_expr.propagate_types(v, col_types) for v in expr.values]
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
        expr.filters = [col_expr.propagate_types(v, col_types) for v in expr.filters]
        return col_types

    elif isinstance(expr, Arrange):
        col_types = propagate_types(expr.table)
        expr.order_by = [col_expr.propagate_types(v, col_types) for v in expr.order_by]
        return col_types

    else:
        raise TypeError
