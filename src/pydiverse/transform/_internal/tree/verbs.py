from __future__ import annotations

import copy
import dataclasses
import uuid
from collections.abc import Callable, Generator, Iterable
from typing import Literal
from uuid import UUID

from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import Col, ColExpr, Order


@dataclasses.dataclass(eq=False, slots=True)
class Verb(AstNode):
    child: AstNode

    def __post_init__(self):
        self.name = self.child.name

    def _clone(self) -> tuple[Verb, dict[AstNode, AstNode], dict[UUID, UUID]]:
        child, nd_map, uuid_map = self.child._clone()
        cloned = copy.copy(self)
        cloned.child = child

        cloned.map_col_nodes(
            lambda col: Col(
                col.name,
                # If the current ast is not in nd_map (happens after collect with
                # keep_col_refs=True), the node wasn't really present anyway.
                nd_map.get(col._ast, col._ast),
                uuid_map[col._uuid],
                col._dtype,
                col._ftype,
            )
            if isinstance(col, Col)
            else copy.copy(col)
        )
        nd_map[self] = cloned

        return cloned, nd_map, uuid_map

    def iter_subtree_postorder(self) -> Iterable[AstNode]:
        yield from self.child.iter_subtree_postorder()
        yield self

    def iter_subtree_preorder(self) -> Generator[AstNode, bool | None, None]:
        if (yield self):
            return
        yield from self.child.iter_subtree_preorder()

    def iter_col_roots(self) -> Iterable[ColExpr]:
        return iter(())

    def iter_col_nodes(self) -> Iterable[ColExpr]:
        for col in self.iter_col_roots():
            yield from col.iter_subtree()

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]): ...

    def map_col_nodes(self, g: Callable[[ColExpr], ColExpr]):
        self.map_col_roots(lambda root: root.map_subtree(g))


@dataclasses.dataclass(eq=False, slots=True)
class Alias(Verb):
    uuid_map: dict[UUID, UUID] | None

    def _clone(self) -> tuple[Verb, dict[AstNode, AstNode], dict[UUID, UUID]]:
        cloned, nd_map, uuid_map = Verb._clone(self)
        if self.uuid_map is not None:  # happens if and only if keep_col_refs=False
            assert set(self.uuid_map.keys()).issubset(uuid_map.keys())
            uuid_map = {
                self.uuid_map[old_uid]: new_uid
                for old_uid, new_uid in uuid_map.items()
                if old_uid in self.uuid_map
            }
            cloned.uuid_map = None
        return cloned, nd_map, uuid_map


@dataclasses.dataclass(eq=False, slots=True)
class Select(Verb):
    select: list[Col]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.select

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.select = [g(col) for col in self.select]


@dataclasses.dataclass(eq=False, slots=True)
class Rename(Verb):
    name_map: dict[str, str]


@dataclasses.dataclass(eq=False, slots=True)
class Mutate(Verb):
    names: list[str]
    values: list[ColExpr]
    uuids: list[UUID]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.values

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.values = [g(val) for val in self.values]

    def _clone(self) -> tuple[Verb, dict[AstNode, AstNode], dict[UUID, UUID]]:
        cloned, nd_map, uuid_map = Verb._clone(self)
        assert isinstance(cloned, Mutate)
        cloned.uuids = [uuid.uuid1() for _ in self.names]
        uuid_map.update(
            {
                old_uid: new_uid
                for old_uid, new_uid in zip(self.uuids, cloned.uuids, strict=True)
            }
        )
        return cloned, nd_map, uuid_map


@dataclasses.dataclass(eq=False, slots=True)
class Filter(Verb):
    predicates: list[ColExpr]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.predicates

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.predicates = [g(predicate) for predicate in self.predicates]


@dataclasses.dataclass(eq=False, slots=True)
class Summarize(Verb):
    names: list[str]
    values: list[ColExpr]
    uuids: list[UUID]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.values

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.values = [g(val) for val in self.values]

    def _clone(self) -> tuple[Verb, dict[AstNode, AstNode], dict[UUID, UUID]]:
        cloned, nd_map, uuid_map = Verb._clone(self)
        assert isinstance(cloned, Summarize)
        cloned.uuids = [uuid.uuid1() for _ in self.names]
        uuid_map.update(
            {
                old_uid: new_uid
                for old_uid, new_uid in zip(self.uuids, cloned.uuids, strict=True)
            }
        )
        return cloned, nd_map, uuid_map


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


@dataclasses.dataclass(eq=False, slots=True)
class GroupBy(Verb):
    group_by: list[Col]
    add: bool

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.group_by

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.group_by = [g(col) for col in self.group_by]


@dataclasses.dataclass(eq=False, slots=True)
class Ungroup(Verb): ...


@dataclasses.dataclass(eq=False, slots=True)
class Join(Verb):
    right: AstNode
    on: ColExpr
    how: Literal["inner", "left", "full"]
    validate: Literal["1:1", "1:m", "m:1", "m:m"]
    suffix: str

    def _clone(self) -> tuple[Join, dict[AstNode, AstNode], dict[UUID, UUID]]:
        child, nd_map, uuid_map = self.child._clone()
        right_child, right_nd_map, right_uuid_map = self.right._clone()
        nd_map.update(right_nd_map)
        uuid_map.update(right_uuid_map)

        cloned = copy.copy(self)
        cloned.child = child
        cloned.right = right_child
        cloned.on = self.on.map_subtree(
            lambda col: Col(
                col.name,
                nd_map.get(col._ast, col._ast),
                uuid_map[col._uuid],
                col._dtype,
                col._ftype,
            )
            if isinstance(col, Col)
            else copy.copy(col)
        )

        nd_map[self] = cloned
        return cloned, nd_map, uuid_map

    def iter_subtree_postorder(self) -> Iterable[AstNode]:
        yield from self.child.iter_subtree_postorder()
        yield from self.right.iter_subtree_postorder()
        yield self

    def iter_subtree_preorder(self) -> Generator[AstNode, bool | None, None]:
        if (yield self):
            return
        yield from self.child.iter_subtree_preorder()
        yield from self.right.iter_subtree_preorder()

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield self.on

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.on = g(self.on)


class SubqueryMarker(Verb): ...
