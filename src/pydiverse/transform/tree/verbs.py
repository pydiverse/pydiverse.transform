from __future__ import annotations

import copy
import dataclasses
import uuid
from collections.abc import Callable, Iterable
from typing import Literal

from pydiverse.transform.errors import FunctionTypeError
from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColFn, ColName, Order
from pydiverse.transform.tree.table_expr import TableExpr

JoinHow = Literal["inner", "left", "outer"]

JoinValidate = Literal["1:1", "1:m", "m:1", "m:m"]


@dataclasses.dataclass(eq=False, slots=True)
class Verb(TableExpr):
    table: TableExpr

    def __post_init__(self):
        # propagate the table name and schema up the tree
        TableExpr.__init__(
            self,
            self.table.name,
            self.table._schema,
            self.table._select,
            self.table._partition_by,
            self.table._name_to_uuid,
        )

        # resolve C columns
        self.map_col_nodes(
            lambda node: node
            if not isinstance(node, ColName)
            else Col(node.name, self.table)
        )

        # TODO: backend agnostic registry
        from pydiverse.transform.backend.polars import PolarsImpl

        # update partition_by kwarg in aggregate functions
        if not isinstance(self, Summarise):
            for node in self.iter_col_nodes():
                if (
                    isinstance(node, ColFn)
                    and "partition_by" not in node.context_kwargs
                    and (
                        PolarsImpl.registry.get_op(node.name).ftype
                        in (Ftype.WINDOW, Ftype.AGGREGATE)
                    )
                ):
                    node.context_kwargs["partition_by"] = self._partition_by

    def iter_col_roots(self) -> Iterable[ColExpr]:
        return iter(())

    def iter_col_nodes(self) -> Iterable[ColExpr]:
        for col in self.iter_col_roots():
            yield from col.iter_nodes()

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]): ...

    def map_col_nodes(self, g: Callable[[ColExpr], ColExpr]):
        self.map_col_roots(lambda root: root.map_nodes(g))

    def clone(self) -> tuple[Verb, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        cloned = copy.copy(self)

        cloned.map_col_nodes(
            lambda node: Col(node.name, table_map[node.table])
            if isinstance(node, Col)
            else copy.copy(node)
        )

        # necessary to make the magic in __post_init__ happen
        cloned = self.__class__(
            table, *(getattr(cloned, attr) for attr in cloned.__slots__)
        )

        table_map[self] = cloned
        return cloned, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Select(Verb):
    selected: list[Col | ColName]

    def __post_init__(self):
        Verb.__post_init__(self)
        self._select = [
            col
            for col in self._select
            if col.uuid in set({col.uuid for col in self.selected})
        ]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.selected

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.selected = [g(c) for c in self.selected]

    def clone(self) -> tuple[Verb, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        cloned = Select(
            table, [Col(col.name, table_map[col.table]) for col in self.selected]
        )
        table_map[self] = cloned
        return cloned, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Drop(Verb):
    dropped: list[Col | ColName]

    def __post_init__(self):
        Verb.__post_init__(self)
        self._select = {
            col
            for col in self._select
            if col.uuid not in set({col.uuid for col in self.dropped})
        }

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.dropped

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.dropped = [g(c) for c in self.dropped]

    def clone(self) -> tuple[Verb, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        cloned = Drop(
            table, [Col(col.name, table_map[col.table]) for col in self.dropped]
        )
        table_map[self] = cloned
        return cloned, table_map


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
        cloned, table_map = Verb.clone(self)
        cloned.name_map = copy.copy(self.name_map)
        return cloned, table_map


@dataclasses.dataclass(eq=False, slots=True)
class Mutate(Verb):
    names: list[str]
    values: list[ColExpr]

    def __post_init__(self):
        Verb.__post_init__(self)

        self._schema = copy.copy(self._schema)
        for name, val in zip(self.names, self.values):
            self._schema[name] = val.dtype(), val.ftype(agg_is_window=True)

        overwritten = {
            self._name_to_uuid[name]
            for name in self.names
            if name in self._name_to_uuid
        }
        self._select = [col for col in self._select if col.uuid not in overwritten]

        self._name_to_uuid = self._name_to_uuid | {
            name: uuid.uuid1() for name in self.names
        }

        self._select = self._select + [Col(name, self) for name in self.names]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.values

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.values = [g(c) for c in self.values]

    def clone(self) -> tuple[Verb, dict[TableExpr, TableExpr]]:
        cloned, table_map = Verb.clone(self)
        cloned.names = copy.copy(self.names)
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

        self._name_to_uuid = self._name_to_uuid | {
            name: uuid.uuid1() for name in self.names
        }
        self._schema = copy.copy(self._schema)
        for name, val in zip(self.names, self.values):
            self._schema[name] = val.dtype(), val.ftype(agg_is_window=False)

        self._select = self._partition_by + [Col(name, self) for name in self.names]
        self._partition_by = []

        for node in self.iter_col_nodes():
            if (
                isinstance(node, ColFn)
                and node.ftype(agg_is_window=False) == Ftype.WINDOW
            ):
                raise FunctionTypeError(
                    f"forbidden window function `{node.name}` in `summarise`"
                )

        for name, val in zip(self.names, self.values):
            if not any(
                isinstance(node, ColFn)
                and node.ftype(agg_is_window=False) == Ftype.AGGREGATE
                for node in val.iter_nodes()
            ):
                raise FunctionTypeError(
                    f"expression of new column `{name}` in `summarise` does not "
                    "contain an aggregation function."
                )

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.values

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.values = [g(c) for c in self.values]

    def clone(self) -> tuple[Verb, dict[TableExpr, TableExpr]]:
        cloned, table_map = Verb.clone(self)
        cloned.names = copy.copy(self.names)
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
        if self._partition_by:
            raise ValueError("cannot apply `slice_head` to a grouped table")


@dataclasses.dataclass(eq=False, slots=True)
class GroupBy(Verb):
    group_by: list[Col | ColName]
    add: bool

    def __post_init__(self):
        Verb.__post_init__(self)
        if self.add:
            self._partition_by = self._partition_by + self.group_by
        else:
            self._partition_by = self.group_by

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.group_by

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.group_by = [g(c) for c in self.group_by]


@dataclasses.dataclass(eq=False, slots=True)
class Ungroup(Verb):
    def __post_init__(self):
        Verb.__post_init__(self)
        self._partition_by = []


@dataclasses.dataclass(eq=False, slots=True)
class Join(Verb):
    table: TableExpr
    right: TableExpr
    on: ColExpr
    how: JoinHow
    validate: JoinValidate
    suffix: str

    def __post_init__(self):
        if self.table._partition_by:
            raise ValueError(f"cannot join grouped table `{self.table.name}`")
        elif self.right._partition_by:
            raise ValueError(f"cannot join grouped table `{self.right.name}`")

        TableExpr.__init__(
            self,
            self.table.name,
            self.table._schema
            | {name + self.suffix: val for name, val in self.right._schema.items()},
            self.table._select + self.right._select,
            [],
            self.table._name_to_uuid
            | {
                name + self.suffix: uid
                for name, uid in self.right._name_to_uuid.items()
            },
        )

        self.map_col_nodes(
            lambda expr: expr
            if not isinstance(expr, ColName)
            else Col(expr.name, self.table)
        )

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield self.on

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.on = g(self.on)

    def clone(self) -> tuple[Join, dict[TableExpr, TableExpr]]:
        table, table_map = self.table.clone()
        right, right_map = self.right.clone()
        table_map.update(right_map)

        cloned = Join(
            table,
            right,
            self.on.map_nodes(
                lambda node: Col(node.name, table_map[node.table])
                if isinstance(node, Col)
                else copy.copy(node)
            ),
            self.how,
            self.validate,
            self.suffix,
        )

        table_map[self] = cloned
        return cloned, table_map
