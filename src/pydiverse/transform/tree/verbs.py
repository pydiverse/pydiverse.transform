from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Iterable
from typing import Literal

from pydiverse.transform.tree import col_expr
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColName, Order
from pydiverse.transform.tree.table_expr import TableExpr

JoinHow = Literal["inner", "left", "outer"]

JoinValidate = Literal["1:1", "1:m", "m:1", "m:m"]


@dataclasses.dataclass(eq=False, slots=True)
class UnaryVerb(TableExpr):
    table: TableExpr

    def __post_init__(self):
        # propagates the table name up the tree
        self.name = self.table.name

    def iter_col_roots(self) -> Iterable[ColExpr]:
        return iter(())

    def iter_col_nodes(self) -> Iterable[ColExpr]:
        for col in self.iter_col_roots():
            yield from col.iter_nodes()

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]): ...

    def map_col_nodes(
        self, g: Callable[[ColExpr], ColExpr]
    ): ...  # TODO simplify things with this

    def clone(self) -> tuple[UnaryVerb, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        cloned = copy.copy(self)
        cloned.table = table
        cloned.map_col_roots(lambda c: col_expr.clone(c, table_map))
        table_map[self] = cloned
        return cloned, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Select(UnaryVerb):
    selected: list[Col | ColName]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.selected

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.selected = [g(c) for c in self.selected]


@dataclasses.dataclass(eq=False, slots=True)
class Drop(UnaryVerb):
    dropped: list[Col | ColName]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.dropped

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
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

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.values

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
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

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.filters

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.filters = [g(c) for c in self.filters]


@dataclasses.dataclass(eq=False, slots=True)
class Summarise(UnaryVerb):
    names: list[str]
    values: list[ColExpr]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.values

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
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

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from (ord.order_by for ord in self.order_by)

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
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

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.group_by

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
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
