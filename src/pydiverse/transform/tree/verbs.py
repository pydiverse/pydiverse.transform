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


def propagate_types(
    expr: TableExpr, needed_cols: set[Col]
) -> dict[Col | ColName, DType]:
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
