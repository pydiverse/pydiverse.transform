from __future__ import annotations

import dataclasses
import functools
import itertools
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

    def mutate_col_exprs(self, g: Callable[[ColExpr], ColExpr]): ...


@dataclasses.dataclass(eq=False, slots=True)
class Select(UnaryVerb):
    selected: list[Col | ColName]

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from self.selected

    def mutate_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        self.selected = [g(c) for c in self.selected]

    def clone(self) -> tuple[Select, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Select(
            table,
            [col.clone(table_map) for col in self.selected],
        )
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Rename(UnaryVerb):
    name_map: dict[str, str]

    def clone(self) -> tuple[Rename, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Rename(table, self.name_map)
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Mutate(UnaryVerb):
    names: list[str]
    values: list[ColExpr]

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from self.values

    def mutate_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        self.values = [g(c) for c in self.values]

    def clone(self) -> tuple[Mutate, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Mutate(table, self.names, [z.clone(table_map) for z in self.values])
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Filter(UnaryVerb):
    filters: list[ColExpr]

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from self.filters

    def mutate_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        self.filters = [g(c) for c in self.filters]

    def clone(self) -> tuple[Filter, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Filter(table, [z.clone(table_map) for z in self.filters])
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Summarise(UnaryVerb):
    names: list[str]
    values: list[ColExpr]

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from self.values

    def mutate_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        self.values = [g(c) for c in self.values]

    def clone(self) -> tuple[Summarise, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Summarise(
            table, self.names, [z.clone(table_map) for z in self.values]
        )
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Arrange(UnaryVerb):
    order_by: list[Order]

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from (ord.order_by for ord in self.order_by)

    def mutate_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        for ord in self.order_by:
            ord.order_by = g(ord.order_by)

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
class SliceHead(UnaryVerb):
    n: int
    offset: int

    def clone(self) -> tuple[SliceHead, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = SliceHead(table, self.n, self.offset)
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class GroupBy(UnaryVerb):
    group_by: list[Col | ColName]
    add: bool

    def col_exprs(self) -> Iterable[ColExpr]:
        yield from self.group_by

    def mutate_col_exprs(self, g: Callable[[ColExpr], ColExpr]):
        self.group_by = [g(c) for c in self.group_by]

    def clone(self) -> tuple[GroupBy, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Mutate(table, [z.clone(table_map) for z in self.group_by], self.add)
        table_map[self] = new_self
        return new_self, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Ungroup(UnaryVerb):
    def clone(self) -> tuple[Ungroup, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        new_self = Ungroup(table)
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

    def __post_init__(self):
        self.name = self.left.name

    def clone(self) -> tuple[Join, dict[TableExpr, TableExpr]]:
        left, left_map = self.left.clone()
        right, right_map = self.right.clone()
        left_map.update(right_map)
        new_self = Join(
            left, right, self.on.clone(left_map), self.how, self.validate, self.suffix
        )
        left_map[self] = new_self
        return new_self, left_map


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


# returns Col -> ColName mapping and the list of available columns
def propagate_names(
    expr: TableExpr, needed_cols: Map2d[TableExpr, set[str]]
) -> Map2d[TableExpr, dict[str, str]]:
    if isinstance(expr, UnaryVerb) and not isinstance(expr, Mutate):
        for c in expr.col_exprs():
            needed_cols.inner_update(col_expr.get_needed_cols(c))
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.mutate_col_exprs(
            functools.partial(col_expr.propagate_names, col_to_name=col_to_name)
        )

        if isinstance(expr, Rename):
            col_to_name.inner_map(
                lambda s: expr.name_map[s] if s in expr.name_map else s
            )

    elif isinstance(expr, Mutate):
        # TODO: also need to do this for summarise, when the user overwrites a grouping
        # col, e.g.
        # s = t >> group_by(u) >> summarise(u=...)
        # s >> mutate(v=(some expression containing t.u and s.u))
        # maybe we could do this in the course of a more general rewrite of summarise
        # to an empty summarise and a mutate

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
        expr.mutate_col_exprs(
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
        col_types = propagate_types(expr.left)
        col_types |= {
            name + expr.suffix: dtype
            for name, dtype in propagate_types(expr.right).items()
        }
        expr.on = col_expr.propagate_types(expr.on, col_types)

    elif isinstance(expr, Table):
        col_types = expr.schema

    else:
        raise AssertionError

    return col_types
