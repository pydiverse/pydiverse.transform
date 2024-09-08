from __future__ import annotations

import copy
import dataclasses
import functools
from collections.abc import Callable, Iterable
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
class UnaryVerb(TableExpr):
    table: TableExpr

    def __post_init__(self):
        # propagates the table name up the tree
        self.name = self.table.name

    def col_exprs(self) -> Iterable[ColExpr]:
        return iter(())

    def replace_col_exprs(self, g: Callable[[ColExpr], ColExpr]): ...

    def clone(self) -> tuple[UnaryVerb, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        cloned = copy.copy(self)
        cloned.table = table
        cloned.replace_col_exprs(lambda c: col_expr.clone(c, table_map))
        table_map[self] = cloned
        return cloned, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Select(UnaryVerb):
    selected: list[Col | ColName]

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from self.selected

    def replace_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        self.selected = [g(c) for c in self.selected]


@dataclasses.dataclass(eq=False, slots=True)
class Drop(UnaryVerb):
    dropped: list[Col | ColName]

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from self.dropped

    def replace_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        self.dropped = [g(c) for c in self.dropped]


@dataclasses.dataclass(eq=False, slots=True)
class Rename(UnaryVerb):
    name_map: dict[str, str]

    def clone(self) -> tuple[UnaryVerb, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        cloned = Rename(table, copy.copy(self.name_map))
        table_map[self] = cloned
        return cloned, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Mutate(UnaryVerb):
    names: list[str]
    values: list[ColExpr]

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from self.values

    def replace_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        self.values = [g(c) for c in self.values]

    def clone(self) -> tuple[UnaryVerb, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        cloned = Mutate(
            table,
            copy.copy(self.names),
            [col_expr.clone(val, table_map) for val in self.values],
        )
        table_map[self] = cloned
        return cloned, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Filter(UnaryVerb):
    filters: list[ColExpr]

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from self.filters

    def replace_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        self.filters = [g(c) for c in self.filters]


@dataclasses.dataclass(eq=False, slots=True)
class Summarise(UnaryVerb):
    names: list[str]
    values: list[ColExpr]

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from self.values

    def replace_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        self.values = [g(c) for c in self.values]

    def clone(self) -> tuple[UnaryVerb, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        cloned = Summarise(
            table,
            copy.copy(self.names),
            [col_expr.clone(val, table_map) for val in self.values],
        )
        table_map[self] = cloned
        return cloned, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Arrange(UnaryVerb):
    order_by: list[Order]

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from (ord.order_by for ord in self.order_by)

    def replace_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        self.order_by = [
            Order(g(ord.order_by), ord.descending, ord.nulls_last)
            for ord in self.order_by
        ]


@dataclasses.dataclass(eq=False, slots=True)
class SliceHead(UnaryVerb):
    n: int
    offset: int


@dataclasses.dataclass(eq=False, slots=True)
class GroupBy(UnaryVerb):
    group_by: list[Col | ColName]
    add: bool

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from self.group_by

    def replace_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        self.group_by = [g(c) for c in self.group_by]


@dataclasses.dataclass(eq=False, slots=True)
class Ungroup(UnaryVerb): ...


@dataclasses.dataclass(eq=False, slots=True)
class Join(TableExpr):
    left: TableExpr
    right: TableExpr
    on: ColExpr
    how: JoinHow
    validate: JoinValidate
    suffix: str

    def __post_init__(self):
        self.name = self.left.name

    def clone(self) -> tuple[Join, dict[TableExpr, TableExpr]]:
        left, left_map = self.left.clone()
        right, right_map = self.right.clone()
        left_map.update(right_map)
        cloned = Join(
            left,
            right,
            col_expr.clone(self.on, left_map),
            self.how,
            self.validate,
            self.suffix,
        )
        left_map[self] = cloned
        return cloned, left_map


# inserts renames before Mutate, Summarise or Join to prevent duplicate column names.
def rename_overwritten_cols(expr: TableExpr) -> tuple[set[str], list[str]]:
    if isinstance(expr, UnaryVerb) and not isinstance(
        expr, (Mutate, Summarise, GroupBy, Ungroup)
    ):
        return rename_overwritten_cols(expr.table)

    elif isinstance(expr, (Mutate, Summarise)):
        available_cols, group_by = rename_overwritten_cols(expr.table)
        if isinstance(expr, Summarise):
            available_cols = set(group_by)
        overwritten = set(name for name in expr.names if name in available_cols)

        if overwritten:
            expr.table = Rename(
                expr.table, {name: name + str(hash(expr)) for name in overwritten}
            )
            for val in expr.values:
                col_expr.rename_overwritten_cols(val, expr.table.name_map)
            expr.table = Drop(
                expr.table, [ColName(name) for name in expr.table.name_map.values()]
            )

        available_cols |= set(
            {
                (name if name not in overwritten else name + str(hash(expr)))
                for name in expr.names
            }
        )

    elif isinstance(expr, GroupBy):
        available_cols, group_by = rename_overwritten_cols(expr.table)
        group_by = expr.group_by + group_by if expr.add else expr.group_by

    elif isinstance(expr, Ungroup):
        available_cols, _ = rename_overwritten_cols(expr.table)
        group_by = []

    elif isinstance(expr, Join):
        left_available, _ = rename_overwritten_cols(expr.left)
        right_avaialable, _ = rename_overwritten_cols(expr.right)
        available_cols = left_available | set(
            {name + expr.suffix for name in right_avaialable}
        )
        group_by = []

    elif isinstance(expr, Table):
        available_cols = set(expr.col_names())
        group_by = []

    else:
        raise AssertionError

    return available_cols, group_by


# returns Col -> ColName mapping and the list of available columns
def propagate_names(
    expr: TableExpr, needed_cols: Map2d[TableExpr, set[str]]
) -> Map2d[TableExpr, dict[str, str]]:
    if isinstance(expr, UnaryVerb):
        for c in expr.col_exprs():
            needed_cols.inner_update(col_expr.get_needed_cols(c))
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.replace_col_exprs(
            functools.partial(col_expr.propagate_names, col_to_name=col_to_name)
        )

        if isinstance(expr, Rename):
            col_to_name.inner_map(
                lambda s: expr.name_map[s] if s in expr.name_map else s
            )

    elif isinstance(expr, Join):
        needed_cols.inner_update(col_expr.get_needed_cols(expr.on))
        col_to_name = propagate_names(expr.left, needed_cols)
        col_to_name_right = propagate_names(expr.right, needed_cols)
        col_to_name_right.inner_map(lambda name: name + expr.suffix)
        col_to_name.inner_update(col_to_name_right)
        expr.on = col_expr.propagate_names(expr.on, col_to_name)

    elif isinstance(expr, Table):
        col_to_name = Map2d()

    else:
        raise AssertionError

    if expr in needed_cols:
        col_to_name.inner_update(
            Map2d({expr: {name: name for name in needed_cols[expr]}})
        )
        del needed_cols[expr]

    return col_to_name


def propagate_types(expr: TableExpr) -> dict[str, DType]:
    if isinstance(expr, (UnaryVerb)):
        col_types = propagate_types(expr.table)
        expr.replace_col_exprs(
            functools.partial(col_expr.propagate_types, col_types=col_types)
        )

        if isinstance(expr, Rename):
            col_types = {
                (expr.name_map[name] if name in expr.name_map else name): dtype
                for name, dtype in propagate_types(expr.table).items()
            }

        elif isinstance(expr, (Mutate, Summarise)):
            col_types.update(
                {name: value.dtype for name, value in zip(expr.names, expr.values)}
            )

    elif isinstance(expr, Join):
        col_types = propagate_types(expr.left) | {
            name + expr.suffix: dtype
            for name, dtype in propagate_types(expr.right).items()
        }
        expr.on = col_expr.propagate_types(expr.on, col_types)

    elif isinstance(expr, Table):
        col_types = expr.schema

    else:
        raise AssertionError

    return col_types


# returns the list of cols the table is currently grouped by
def update_partition_by_kwarg(expr: TableExpr) -> list[ColExpr]:
    if isinstance(expr, UnaryVerb) and not isinstance(expr, Summarise):
        group_by = update_partition_by_kwarg(expr.table)
        for c in expr.col_exprs():
            col_expr.update_partition_by_kwarg(c, group_by)

        if isinstance(expr, GroupBy):
            group_by = expr.group_by

        elif isinstance(expr, Ungroup):
            group_by = []

    elif isinstance(expr, Join):
        update_partition_by_kwarg(expr.left)
        update_partition_by_kwarg(expr.right)
        group_by = []

    elif isinstance(expr, (Summarise, Table)):
        group_by = []

    else:
        raise AssertionError

    return group_by
