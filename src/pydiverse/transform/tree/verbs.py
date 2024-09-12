from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Iterable
from typing import Literal

from pydiverse.transform.errors import FunctionTypeError
from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.tree import col_expr
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColName, Order
from pydiverse.transform.tree.table_expr import TableExpr

JoinHow = Literal["inner", "left", "outer"]

JoinValidate = Literal["1:1", "1:m", "m:1", "m:m"]


@dataclasses.dataclass(eq=False, slots=True)
class Verb(TableExpr):
    table: TableExpr

    def __post_init__(self):
        # propagates the table name and schema up the tree
        self.name = self.table.name
        self._schema = self.table._schema
        self._group_by = self.table._group_by
        self.map_col_nodes(
            lambda expr: expr
            if not isinstance(expr, ColName)
            else Col(expr.name, self.table)
        )

    def iter_col_roots(self) -> Iterable[ColExpr]:
        return iter(())

    def iter_col_nodes(self) -> Iterable[ColExpr]:
        for col in self.iter_col_roots():
            yield from col.iter_nodes()

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]): ...

    def map_col_nodes(
        self, g: Callable[[ColExpr], ColExpr]
    ): ...  # TODO simplify things with this

    def clone(self) -> tuple[Verb, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        cloned = copy.copy(self)
        cloned.table = table
        cloned.map_col_roots(lambda c: col_expr.clone(c, table_map))
        table_map[self] = cloned
        return cloned, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Select(Verb):
    selected: list[Col]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.selected

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.selected = [g(c) for c in self.selected]


@dataclasses.dataclass(eq=False, slots=True)
class Drop(Verb):
    dropped: list[Col]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.dropped

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.dropped = [g(c) for c in self.dropped]


@dataclasses.dataclass(eq=False, slots=True)
class Rename(Verb):
    name_map: dict[str, str]

    def __post_init__(self):
        Verb.__post_init__(self)
        new_schema = copy.copy(self._schema)
        for name, _ in self.name_map.items():
            if name not in self._schema:
                raise ValueError(f"no column with name `{name}` in table `{self.name}`")
            del new_schema[name]
        for name, replacement in self.name_map.items():
            if replacement in new_schema:
                raise ValueError(f"duplicate column name `{replacement}`")
            new_schema[replacement] = self._schema[name]
        self._schema = new_schema

    def clone(self) -> tuple[Verb, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        cloned = Rename(table, copy.copy(self.name_map))
        table_map[self] = cloned
        return cloned, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Mutate(Verb):
    names: list[str]
    values: list[ColExpr]

    def __post_init__(self):
        Verb.__post_init__(self)
        self._schema = copy.copy(self._schema)
        for name, val in zip(self.names, self.values):
            self._schema[name] = val.dtype(), val.ftype(False)

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.values

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.values = [g(c) for c in self.values]

    def clone(self) -> tuple[Verb, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        cloned = Mutate(
            table,
            copy.copy(self.names),
            [col_expr.clone(val, table_map) for val in self.values],
        )
        table_map[self] = cloned
        return cloned, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Filter(Verb):
    filters: list[ColExpr]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.filters

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.filters = [g(c) for c in self.filters]


@dataclasses.dataclass(eq=False, slots=True)
class Summarise(Verb):
    names: list[str]
    values: list[ColExpr]

    def __post_init__(self):
        Verb.__post_init__(self)
        self._schema = copy.copy(self._schema)
        for name, val in zip(self.names, self.values):
            self._schema[name] = val.dtype(), val.ftype(False)

        for node in self.iter_col_nodes():
            if node.ftype == Ftype.WINDOW:
                # TODO: traverse thet expression and find the name of the window fn. It
                # does not matter if this means traversing the whole tree since we're
                # stopping execution anyway.
                raise FunctionTypeError(
                    f"forbidden window function in expression `{node}` in "
                    "`summarise`"
                )

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.values

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.values = [g(c) for c in self.values]

    def clone(self) -> tuple[Verb, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        cloned = Summarise(
            table,
            copy.copy(self.names),
            [col_expr.clone(val, table_map) for val in self.values],
        )
        table_map[self] = cloned
        return cloned, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Arrange(Verb):
    order_by: list[Order]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from (ord.order_by for ord in self.order_by)

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.order_by = [
            Order(g(ord.order_by), ord.descending, ord.nulls_last)
            for ord in self.order_by
        ]


@dataclasses.dataclass(eq=False, slots=True)
class SliceHead(Verb):
    n: int
    offset: int

    def __post_init__(self):
        Verb.__post_init__(self)
        if self._group_by:
            raise ValueError("cannot apply `slice_head` to a grouped table")


@dataclasses.dataclass(eq=False, slots=True)
class GroupBy(Verb):
    group_by: list[Col]
    add: bool

    def __post_init__(self):
        Verb.__post_init__(self)
        if self.add:
            self._group_by += self.group_by
        else:
            self._group_by = self.group_by

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.group_by

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.group_by = [g(c) for c in self.group_by]


@dataclasses.dataclass(eq=False, slots=True)
class Ungroup(Verb):
    def __post_init__(self):
        Verb.__post_init__(self)
        self._group_by = []


@dataclasses.dataclass(eq=False, slots=True)
class Join(Verb):
    table: TableExpr
    right: TableExpr
    on: ColExpr
    how: JoinHow
    validate: JoinValidate
    suffix: str

    def __post_init__(self):
        if self.table._group_by:
            raise ValueError(f"cannot join grouped table `{self.table.name}`")
        elif self.right._group_by:
            raise ValueError(f"cannot join grouped table `{self.right.name}`")
        Verb.__post_init__(self)
        self._schema |= {
            name + self.suffix: val for name, val in self.right._schema.items()
        }

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield self.on

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.on = g(self.on)

    def clone(self) -> tuple[Join, dict[TableExpr, TableExpr]]:
        left, left_map = self.table.clone()
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
