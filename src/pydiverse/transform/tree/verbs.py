from __future__ import annotations

import dataclasses
import itertools
from typing import Literal

from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import col_expr
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColName, Order
from pydiverse.transform.tree.dtypes import DType
from pydiverse.transform.tree.table_expr import TableExpr
from pydiverse.transform.util.map2d import Map2d

JoinHow = Literal["inner", "left", "outer"]

JoinValidate = Literal["1:1", "1:m", "m:1", "m:m"]


@dataclasses.dataclass(eq=False, slots=True)
class Select(TableExpr):
    table: TableExpr
    selects: list[Col | ColName]

    def clone(self) -> tuple[Select, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Select(
            table,
            [col.clone(table_map) for col in self.selects],
        )
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Rename(TableExpr):
    table: TableExpr
    name_map: dict[str, str]

    def clone(self) -> tuple[Rename, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Rename(table, self.name_map)
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Mutate(TableExpr):
    table: TableExpr
    names: list[str]
    values: list[ColExpr]

    def clone(self) -> tuple[Mutate, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Mutate(table, self.names, [z.clone(table_map) for z in self.values])
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Join(TableExpr):
    left: TableExpr
    right: TableExpr
    on: ColExpr
    how: JoinHow
    validate: JoinValidate
    suffix: str

    def clone(self) -> tuple[Join, dict[TableExpr, TableExpr]]:
        left, left_map = self.left.clone()
        right, right_map = self.right.clone()
        left_map.update(right_map)
        new_self = Join(
            left, right, self.on.clone(left_map), self.how, self.validate, self.suffix
        )
        left_map[self] = new_self
        return new_self, left_map


@dataclasses.dataclass(eq=False, slots=True)
class Filter(TableExpr):
    table: TableExpr
    filters: list[ColExpr]

    def clone(self) -> tuple[Filter, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Filter(table, [z.clone(table_map) for z in self.filters])
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Summarise(TableExpr):
    table: TableExpr
    names: list[str]
    values: list[ColExpr]

    def clone(self) -> tuple[Summarise, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Summarise(
            table, self.names, [z.clone(table_map) for z in self.values]
        )
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Arrange(TableExpr):
    table: TableExpr
    order_by: list[Order]

    def clone(self) -> tuple[Arrange, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Arrange(
            table,
            [
                Order(z.order_by.clone(table_map), z.descending, z.nulls_last)
                for z in self.order_by
            ],
        )
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class SliceHead(TableExpr):
    table: TableExpr
    n: int
    offset: int

    def clone(self) -> tuple[SliceHead, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = SliceHead(table, self.n, self.offset)
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class GroupBy(TableExpr):
    table: TableExpr
    group_by: list[Col | ColName]
    add: bool

    def clone(self) -> tuple[GroupBy, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Mutate(table, [z.clone(table_map) for z in self.group_by], self.add)
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Ungroup(TableExpr):
    table: TableExpr

    def clone(self) -> tuple[Ungroup, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Ungroup(table)
        table_map[self] = new_self
        return new_self, table_map


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
            (ColName(col_to_name[col.table][col.name]) if isinstance(col, Col) else col)
            for col in expr.selects
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
        # {name}{hash of the previous table} as a dummy name.
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
            col_expr.propagate_names(ord, col_to_name) for ord in expr.order_by
        ]

    elif isinstance(expr, GroupBy):
        for v in expr.group_by:
            needed_cols.inner_update(col_expr.get_needed_cols(v))
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.group_by = [
            col_expr.propagate_names(v, col_to_name) for v in expr.group_by
        ]

    elif isinstance(expr, (Ungroup, SliceHead)):
        return propagate_names(expr.table, needed_cols)

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
    if isinstance(expr, (SliceHead, Ungroup, Select, GroupBy)):
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
            col_expr.propagate_types(ord, col_types) for ord in expr.order_by
        ]
        return col_types

    elif isinstance(expr, Table):
        return expr.schema

    else:
        raise TypeError
