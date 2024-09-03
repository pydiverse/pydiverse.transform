from __future__ import annotations

import copy
import dataclasses
import itertools
from typing import Literal

from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import col_expr
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColName, Map2d, Order
from pydiverse.transform.tree.dtypes import DType
from pydiverse.transform.tree.table_expr import TableExpr

JoinHow = Literal["inner", "left", "outer"]

JoinValidate = Literal["1:1", "1:m", "m:1", "m:m"]


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
def propagate_names(
    expr: TableExpr, needed_cols: Map2d[TableExpr, set[str]]
) -> Map2d[TableExpr, dict[str, str]]:
    if isinstance(expr, Select):
        for col in expr.selects:
            if isinstance(col, Col):
                if col.table in needed_cols:
                    needed_cols[col.table].add(col.name)
                else:
                    needed_cols[col.table] = set({col.name})
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.selects = [
            ColName(col_to_name[col.table][col.name])
            for col in expr.selects
            if isinstance(col, Col)
        ]

    elif isinstance(expr, Rename):
        col_to_name = propagate_names(expr.table, needed_cols)
        col_to_name.inner_map(lambda s: expr.name_map[s] if s in expr.name_map else s)

    elif isinstance(expr, Mutate):
        for v in expr.values:
            needed_cols.inner_update(col_expr.get_needed_cols(v))
        col_to_name = propagate_names(expr.table, needed_cols)
        # overwritten columns still need to be stored since the user may access them
        # later. They're not in the C-space anymore, however, so we give them
        # {name}_{hash of the previous table} as a dummy name.
        overwritten = set(
            name
            for name in expr.names
            if name
            in set(
                itertools.chain.from_iterable(v.values() for v in col_to_name.values())
            )
        )
        # for the backends, we insert a Rename here that gives the overwritten cols
        # their dummy names. The backends may thus assume that the user never overwrites
        # column names
        if overwritten:
            rn = Rename(
                expr.table, {name: name + str(hash(expr.table)) for name in overwritten}
            )
            col_to_name.inner_map(
                lambda s: s + str(hash(expr.table)) if s in overwritten else s
            )
            expr.table = rn
        expr.values = [col_expr.propagate_names(v, col_to_name) for v in expr.values]

    elif isinstance(expr, Join):
        needed_cols.inner_update(col_expr.get_needed_cols(expr.on))
        col_to_name = propagate_names(expr.left, needed_cols)
        col_to_name_right = propagate_names(expr.right, needed_cols)
        col_to_name_right.inner_map(lambda s: s + expr.suffix)
        col_to_name.inner_update(col_to_name_right)
        expr.on = col_expr.propagate_names(expr.on, col_to_name)

    elif isinstance(expr, Filter):
        for v in expr.filters:
            needed_cols.inner_update(col_expr.get_needed_cols(v))
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.filters = [col_expr.propagate_names(v, col_to_name) for v in expr.filters]

    elif isinstance(expr, Arrange):
        for order in expr.order_by:
            needed_cols.inner_update(col_expr.get_needed_cols(order.order_by))
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.order_by = [
            Order(
                col_expr.propagate_names(order.order_by, col_to_name),
                order.descending,
                order.nulls_last,
            )
            for order in expr.order_by
        ]

    elif isinstance(expr, GroupBy):
        for v in expr.group_by:
            needed_cols.inner_update(col_expr.get_needed_cols(v))
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.group_by = [propagate_names(v, col_to_name) for v in expr.group_by]

    elif isinstance(expr, Summarise):
        for v in expr.values:
            needed_cols.inner_update(col_expr.get_needed_cols(v))
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.values = [col_expr.propagate_names(v, col_to_name) for v in expr.values]

    elif isinstance(expr, Table):
        col_to_name = Map2d()

    else:
        raise TypeError

    if expr in needed_cols:
        col_to_name.inner_update(
            Map2d({expr: {name: name for name in needed_cols[expr]}})
        )
        del needed_cols[expr]

    return col_to_name


def propagate_types(expr: TableExpr) -> dict[str, DType]:
    if isinstance(expr, (SliceHead, Ungroup, Select, SliceHead, GroupBy)):
        return propagate_types(expr.table)

    if isinstance(expr, Rename):
        col_types = propagate_types(expr.table)
        return {
            (expr.name_map[name] if name in expr.name_map else name): dtype
            for name, dtype in col_types.items()
        }

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
            name + expr.suffix: dtype
            for name, dtype in propagate_types(expr.right).items()
        }
        return col_types_left | col_types_right

    elif isinstance(expr, Filter):
        col_types = propagate_types(expr.table)
        expr.filters = [col_expr.propagate_types(v, col_types) for v in expr.filters]
        return col_types

    elif isinstance(expr, Arrange):
        col_types = propagate_types(expr.table)
        expr.order_by = [
            Order(
                col_expr.propagate_types(ord.order_by, col_types),
                ord.descending,
                ord.nulls_last,
            )
            for ord in expr.order_by
        ]
        return col_types

    elif isinstance(expr, Table):
        return expr.schema()

    else:
        raise TypeError


def recursive_copy(expr: TableExpr) -> TableExpr:
    new_expr = copy.copy(expr)
    if isinstance(expr, Join):
        new_expr.left = recursive_copy(expr.left)
        new_expr.right = recursive_copy(expr.right)
    elif isinstance(expr, Table):
        new_expr._impl = copy.copy(expr._impl)
    else:
        new_expr.table = recursive_copy(expr.table)
    return new_expr
