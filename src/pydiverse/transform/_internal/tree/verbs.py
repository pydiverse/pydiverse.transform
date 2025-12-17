# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses
import re
import uuid
from collections.abc import Callable, Iterable
from typing import Literal
from uuid import UUID

from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import Col, ColExpr, Order


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class Verb(AstNode):
    child: AstNode

    def __post_init__(self):
        self.name = self.child.name

    def _unformatted_ast_repr(self, verb_depth: int, expr_depth: int, display_name_map):
        nd_repr = self._ast_node_repr(expr_depth, display_name_map)
        return (
            self.child._unformatted_ast_repr(verb_depth - 1, expr_depth, display_name_map) if verb_depth != 0 else "..."
        ) + f" >> {nd_repr}"

    def _clone(self) -> tuple["Verb", dict[AstNode, AstNode], dict[UUID, UUID]]:
        child, nd_map, uuid_map = self.child._clone()
        cloned = copy.copy(self)
        cloned.child = child

        cloned.map_col_nodes(
            lambda col: Col(
                col.name,
                # If the current ast is not in nd_map (happens after collect with
                # keep_col_refs=True), the node wasn't really present anyway.
                nd_map.get(col._ast, col._ast),
                uuid_map.get(col._uuid, col._uuid),
                col._dtype,
                col._ftype,
            )
            if isinstance(col, Col)
            else copy.copy(col)
        )
        nd_map[self] = cloned

        return cloned, nd_map, uuid_map

    def short_name(self):
        last_verb = self if not isinstance(self, SubqueryMarker) else self.child
        return (
            "?" if self.name is None else self.name
        ) + f" (last verb: {camel_to_snake(last_verb.__class__.__name__)})"

    def iter_subtree_postorder(self) -> Iterable[AstNode]:
        yield from self.child.iter_subtree_postorder()
        yield self

    def iter_subtree_preorder(self):
        yield self
        yield from self.child.iter_subtree_preorder()

    def iter_col_roots(self) -> Iterable[ColExpr]:
        return iter(())

    def iter_col_nodes(self) -> Iterable[ColExpr]:
        for col in self.iter_col_roots():
            yield from col.iter_subtree_postorder()

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]): ...

    def map_col_nodes(self, g: Callable[[ColExpr], ColExpr]):
        self.map_col_roots(lambda root: root.map_subtree(g))


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class Alias(Verb):
    uuid_map: dict[UUID, UUID] | None

    # TODO: currently we kinda misuse Alias for column reference transfer. Maybe it
    # would be nice to create a separate marker node for this to distinguish it from a
    # user-created alias.
    def _ast_node_repr(self, expr_depth: int, display_name_map) -> str:
        return "alias(" + (f"'{self.name}'" if self.name != self.child.name else "") + ")"

    def _clone(self) -> tuple[Verb, dict[AstNode, AstNode], dict[UUID, UUID]]:
        cloned, nd_map, uuid_map = Verb._clone(self)
        if self.uuid_map is not None:  # happens if and only if keep_col_refs=False
            assert set(self.uuid_map.keys()).issubset(uuid_map.keys())
            uuid_map = {
                self.uuid_map[old_uid]: new_uid for old_uid, new_uid in uuid_map.items() if old_uid in self.uuid_map
            }
            cloned.uuid_map = None
        return cloned, nd_map, uuid_map


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class Select(Verb):
    select: list[Col]

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.select

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.select = [g(col) for col in self.select]

    def _ast_node_repr(self, expr_depth: int, display_name_map) -> str:
        return "select(" + ", ".join(col._ast_repr(expr_depth, False, display_name_map) for col in self.select) + ")"


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class Rename(Verb):
    name_map: dict[str, str]

    def _ast_node_repr(self, expr_depth: int, display_name_map) -> str:
        return "rename({" + ", ".join(f"'{k}': '{v}'" for k, v in self.name_map.items()) + "})"


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class Mutate(Verb):
    names: list[str]
    values: list[ColExpr]
    uuids: list[UUID]

    def _ast_node_repr(self, expr_depth: int, display_name_map) -> str:
        return "mutate(" + (
            ", ".join(
                f"{k}={v._ast_repr(expr_depth, False, display_name_map)}"
                for k, v in zip(self.names, self.values, strict=True)
            )
            + ")"
        )

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.values

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.values = [g(val) for val in self.values]

    def _clone(self) -> tuple[Verb, dict[AstNode, AstNode], dict[UUID, UUID]]:
        cloned, nd_map, uuid_map = Verb._clone(self)
        assert isinstance(cloned, Mutate)
        cloned.uuids = [uuid.uuid1() for _ in self.names]
        uuid_map.update({old_uid: new_uid for old_uid, new_uid in zip(self.uuids, cloned.uuids, strict=True)})
        return cloned, nd_map, uuid_map


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class Filter(Verb):
    predicates: list[ColExpr]

    def _ast_node_repr(self, expr_depth: int, display_name_map) -> str:
        return (
            "filter(" + ", ".join(pred._ast_repr(expr_depth, False, display_name_map) for pred in self.predicates) + ")"
        )

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.predicates

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.predicates = [g(predicate) for predicate in self.predicates]


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class Summarize(Verb):
    names: list[str]
    values: list[ColExpr]
    uuids: list[UUID]

    def _ast_node_repr(self, expr_depth: int, display_name_map: dict) -> str:
        return "summarize(" + (
            ", ".join(
                f"{k}={v._ast_repr(expr_depth, False, display_name_map)}"
                for k, v in zip(self.names, self.values, strict=True)
            )
            + ")"
        )

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.values

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.values = [g(val) for val in self.values]

    def _clone(self) -> tuple[Verb, dict[AstNode, AstNode], dict[UUID, UUID]]:
        cloned, nd_map, uuid_map = Verb._clone(self)
        assert isinstance(cloned, Summarize)
        cloned.uuids = [uuid.uuid1() for _ in self.names]
        uuid_map.update({old_uid: new_uid for old_uid, new_uid in zip(self.uuids, cloned.uuids, strict=True)})
        return cloned, nd_map, uuid_map


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class Arrange(Verb):
    order_by: list[Order]

    def _ast_node_repr(self, expr_depth: int, display_name_map) -> str:
        return "arrange(" + ", ".join(ord._ast_repr(expr_depth, False, display_name_map) for ord in self.order_by) + ")"

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from (ord.order_by for ord in self.order_by)

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.order_by = [Order(g(ord.order_by), ord.descending, ord.nulls_last) for ord in self.order_by]


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class SliceHead(Verb):
    n: int
    offset: int

    def _ast_node_repr(self, expr_depth: int, display_name_map) -> str:
        return f"slice_head(n={self.n}, offset={self.offset})"


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class GroupBy(Verb):
    group_by: list[Col]
    add: bool

    def _ast_node_repr(self, expr_depth: int, display_name_map) -> str:
        return (
            "group_by(" + ", ".join(col._ast_repr(expr_depth, False, display_name_map) for col in self.group_by) + ")"
        )

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield from self.group_by

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.group_by = [g(col) for col in self.group_by]


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class Ungroup(Verb):
    def _ast_node_repr(self, expr_depth, display_name_map):
        return "ungroup()"


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class Join(Verb):
    right: AstNode
    on: ColExpr
    how: Literal["inner", "left", "full"]
    validate: Literal["1:1", "1:m", "m:1", "m:m"]

    def _unformatted_ast_repr(self, verb_depth: int, expr_depth: int, display_name_map) -> str:
        return (
            self.child._unformatted_ast_repr(verb_depth - 1, expr_depth, display_name_map) if verb_depth != 0 else "..."
        ) + (
            ">> join("
            + (
                self.right._unformatted_ast_repr(verb_depth - 1, expr_depth, display_name_map)
                if verb_depth != 0
                else self.right._unformatted_ast_repr(0, expr_depth, display_name_map)
            )
            + ", "
            + f"how='{self.how}', "
            + f"on={self.on._ast_repr(expr_depth, False, display_name_map)}, "
            + f"validate='{self.validate}')"
        )

    def _clone(self) -> tuple["Join", dict[AstNode, AstNode], dict[UUID, UUID]]:
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

    def iter_subtree_preorder(self):
        yield self
        yield from self.child.iter_subtree_preorder()
        yield from self.right.iter_subtree_preorder()

    def iter_col_roots(self) -> Iterable[ColExpr]:
        yield self.on

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        self.on = g(self.on)


@dataclasses.dataclass(eq=False, slots=True, repr=False)
class Union(Verb):
    right: AstNode
    distinct: bool

    def _unformatted_ast_repr(self, verb_depth: int, expr_depth: int, display_name_map) -> str:
        return (
            self.child._unformatted_ast_repr(verb_depth - 1, expr_depth, display_name_map) if verb_depth != 0 else "..."
        ) + (
            ">> union("
            + (
                self.right._unformatted_ast_repr(verb_depth - 1, expr_depth, display_name_map)
                if verb_depth != 0
                else self.right._unformatted_ast_repr(0, expr_depth, display_name_map)
            )
            + f", distinct={self.distinct})"
        )

    def _clone(self) -> tuple["Union", dict[AstNode, AstNode], dict[UUID, UUID]]:
        child, nd_map, uuid_map = self.child._clone()
        right_child, right_nd_map, right_uuid_map = self.right._clone()
        nd_map.update(right_nd_map)
        uuid_map.update(right_uuid_map)

        cloned = copy.copy(self)
        cloned.child = child
        cloned.right = right_child

        nd_map[self] = cloned
        return cloned, nd_map, uuid_map

    def iter_subtree_postorder(self) -> Iterable[AstNode]:
        yield from self.child.iter_subtree_postorder()
        yield from self.right.iter_subtree_postorder()
        yield self

    def iter_subtree_preorder(self):
        yield self
        yield from self.child.iter_subtree_preorder()
        yield from self.right.iter_subtree_preorder()

    def iter_col_roots(self) -> Iterable[ColExpr]:
        return iter(())

    def map_col_roots(self, g: Callable[[ColExpr], ColExpr]):
        pass


class SubqueryMarker(Verb):
    def _ast_node_repr(self, expr_depth, display_name_map):
        return "subquery_marker"


CAMEL_TO_SNAKE = re.compile(r"(?<!^)(?=[A-Z])")


def camel_to_snake(s: str) -> str:
    return CAMEL_TO_SNAKE.sub("_", s).lower()
