from __future__ import annotations

import copy
import dataclasses
from typing import Literal
from uuid import UUID

from pydiverse.transform._internal.backend.table_impl import TableImpl
from pydiverse.transform._internal.ops.op import Ftype
from pydiverse.transform._internal.tree import types, verbs
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import Col, ColFn


@dataclasses.dataclass(slots=True)
class Cache:
    name_to_uuid: dict[str, UUID]  # the selected columns, in order
    uuid_to_name: dict[UUID, str]  # again, only the selected columns + in order
    partition_by: list[UUID]
    derived_from: set[AstNode]  # for detecting invalid columns and self-joins
    cols: dict[UUID, Col]  # all columns in current scope (including hidden ones)

    # the following are only necessary for subquery detection
    limit: int
    group_by: set[UUID]
    is_filtered: bool

    backend: Literal["polars", "sqlite", "postgres", "duckdb", "mssql"]

    # For a column to be usable in an expression, the table it comes from must be an
    # ancestor of the current table AND the column's UUID must be in `all_cols` of the
    # current table (the latter is necessary because not every column is usable after
    # `summarize``)

    @staticmethod
    def from_ast(node: AstNode) -> Cache:
        if isinstance(node, verbs.Verb):
            if isinstance(node, verbs.Join):
                return Cache.from_ast(node.child).update(
                    node, right_cache=Cache.from_ast(node.right)
                )
            else:
                return Cache.from_ast(node.child).update(node)

        assert isinstance(node, TableImpl)
        return Cache(
            name_to_uuid={col.name: col._uuid for col in node.cols.values()},
            uuid_to_name={col._uuid: col.name for col in node.cols.values()},
            partition_by=[],
            derived_from={node},
            cols={col._uuid: col for col in node.cols.values()},
            limit=0,
            group_by=set(),
            is_filtered=False,
            backend=node.backend_name,
        )

    def update(self, node: verbs.Verb, *, right_cache: Cache | None = None) -> Cache:
        """
        Returns a new cache for `node`, assuming `self` is the cache of `node.child`.
        Does not modify `self`.
        """

        res = copy.copy(self)

        if isinstance(node, verbs.Alias):
            if node.uuid_map is not None:
                res.name_to_uuid = {
                    name: node.uuid_map[uid] for name, uid in self.name_to_uuid.items()
                }
                res.uuid_to_name = {uid: name for name, uid in res.name_to_uuid.items()}
                res.cols = {
                    node.uuid_map[uid]: Col(
                        col.name, node, node.uuid_map[uid], col._dtype, col._ftype
                    )
                    for uid, col in self.cols.items()
                }
                res.partition_by = [node.uuid_map[uid] for uid in self.partition_by]
                res.derived_from = set()

        elif isinstance(node, verbs.Select):
            selected_uuids = set(col._uuid for col in node.select)
            res.uuid_to_name = {
                uid: name
                for uid, name in self.uuid_to_name.items()
                if uid in selected_uuids
            }
            res.name_to_uuid = {name: uid for uid, name in res.uuid_to_name.items()}

        elif isinstance(node, verbs.Rename):
            res.name_to_uuid = {
                (new_name if (new_name := node.name_map.get(name)) else name): uid
                for name, uid in self.name_to_uuid.items()
            }
            res.uuid_to_name = {uid: name for name, uid in res.name_to_uuid.items()}

        elif isinstance(node, verbs.Mutate):
            res.cols = self.cols | {
                uid: Col(name, node, uid, val.dtype(), val.ftype(agg_is_window=True))
                for name, val, uid in zip(
                    node.names, node.values, node.uuids, strict=True
                )
            }
            res.name_to_uuid = self.name_to_uuid | {
                name: uid for name, uid in zip(node.names, node.uuids, strict=True)
            }
            res.uuid_to_name = {uid: name for name, uid in res.name_to_uuid.items()}

        elif isinstance(node, verbs.Filter):
            res.is_filtered = True

        elif isinstance(node, verbs.GroupBy):
            res.partition_by = [col._uuid for col in node.group_by]
            if node.add:
                res.partition_by = self.partition_by + res.partition_by

        elif isinstance(node, verbs.Ungroup):
            res.partition_by = []

        elif isinstance(node, verbs.Summarize):
            overwritten = {
                col_name for col_name in node.names if col_name in self.name_to_uuid
            }
            cols = {
                self.uuid_to_name[uid]: self.cols[uid]
                for uid in self.partition_by
                if self.uuid_to_name[uid] not in overwritten
            } | {
                name: Col(name, node, uid, val.dtype(), val.ftype())
                for name, val, uid in zip(
                    node.names, node.values, node.uuids, strict=True
                )
            }

            res.cols = {col._uuid: col for _, col in cols.items()}
            res.name_to_uuid = {name: col._uuid for name, col in cols.items()}
            res.uuid_to_name = {uid: name for name, uid in res.name_to_uuid.items()}
            res.group_by = res.group_by | set(res.partition_by)
            res.partition_by = []

        elif isinstance(node, verbs.SliceHead):
            res.limit = node.n

        elif isinstance(node, verbs.Join):
            assert right_cache is not None

            res.cols = self.cols | right_cache.cols
            res.name_to_uuid = self.name_to_uuid | {
                name + node.suffix: uid
                for name, uid in right_cache.name_to_uuid.items()
            }
            res.uuid_to_name = {uid: name for name, uid in res.name_to_uuid.items()}

            res.derived_from = self.derived_from | right_cache.derived_from
            res.limit = 0
            res.group_by = set()

        elif isinstance(node, verbs.SubqueryMarker):
            res.cols = {
                uid: Col(
                    col.name,
                    node.child,
                    uid,
                    types.without_const(col._dtype),
                    Ftype.ELEMENT_WISE,
                )
                for uid, col in self.cols.items()
            }
            res.limit = 0
            res.group_by = set()
            res.is_filtered = False

        assert len(res.name_to_uuid) == len(res.uuid_to_name)
        res.derived_from = res.derived_from | {node}

        return res

    # Returns whether applying `node` to this table would require it to be wrapped in a
    # subquery. (so `self` is the old cache and `node` the new AST)
    def requires_subquery(self, node: verbs.Verb) -> bool:
        if self.backend == "polars":
            return False

        return (
            (
                isinstance(
                    node,
                    verbs.Filter
                    | verbs.Summarize
                    | verbs.Arrange
                    | verbs.GroupBy
                    | verbs.Join,
                )
                and self.limit != 0
            )
            or (
                isinstance(node, verbs.Mutate)
                and any(
                    any(
                        col.ftype(agg_is_window=True) in (Ftype.WINDOW, Ftype.AGGREGATE)
                        for col in fn.iter_subtree()
                        if isinstance(col, Col)
                    )
                    for fn in node.iter_col_nodes()
                    if (
                        isinstance(fn, ColFn)
                        and fn.op.ftype in (Ftype.AGGREGATE, Ftype.WINDOW)
                    )
                )
            )
            or (
                isinstance(node, verbs.Filter)
                and any(
                    col.ftype(agg_is_window=True) == Ftype.WINDOW
                    for col in node.iter_col_nodes()
                    if isinstance(col, Col)
                )
            )
            or (
                isinstance(node, verbs.Summarize)
                and (
                    (self.group_by and self.group_by != set(self.partition_by))
                    or any(
                        (
                            col.ftype(agg_is_window=False)
                            in (Ftype.WINDOW, Ftype.AGGREGATE)
                        )
                        for col in node.iter_col_nodes()
                        if isinstance(col, Col)
                    )
                    or any(
                        self.cols[uid].ftype() == Ftype.WINDOW
                        for uid in self.partition_by
                    )
                )
            )
            or (
                isinstance(node, verbs.Join)
                and (
                    self.group_by
                    or (
                        (
                            node.how == "full"
                            or (node.right in self.derived_from and node.how == "left")
                        )
                        and any(
                            types.is_const(self.cols[uid].dtype())
                            for uid in self.uuid_to_name.keys()
                        )
                    )
                    or any(
                        self.cols[uid].ftype() == Ftype.WINDOW
                        for uid in self.uuid_to_name.keys()
                    )
                    or any(
                        col.ftype() != Ftype.ELEMENT_WISE and col._uuid in self.cols
                        for col in node.on.iter_subtree()
                        if isinstance(col, Col)
                    )
                    or (self.is_filtered and node.how == "full")
                )
            )
        )

    def selected_cols(self) -> list[Col]:
        return [self.cols[uid] for uid in self.uuid_to_name.keys()]
