from __future__ import annotations

import copy
import dataclasses
from uuid import UUID

from pydiverse.transform._internal.backend.table_impl import TableImpl
from pydiverse.transform._internal.tree import verbs
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import Col


@dataclasses.dataclass(slots=True)
class Cache:
    name_to_uuid: dict[str, UUID]  # the selected columns, in order
    uuid_to_name: dict[UUID, str]  # again, only the selected columns + in order
    partition_by: list[Col]
    derived_from: set[AstNode]  # for detecting invalid columns and self-joins
    cols: dict[UUID, Col]  # all columns in current scope (including hidden ones)

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
                res.partition_by = [
                    res.cols[node.uuid_map[col._uuid]] for col in self.partition_by
                ]
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

        elif isinstance(node, verbs.GroupBy):
            if node.add:
                res.partition_by = self.partition_by + node.group_by
            else:
                res.partition_by = node.group_by

        elif isinstance(node, verbs.Ungroup):
            res.partition_by = []

        elif isinstance(node, verbs.Summarize):
            overwritten = {
                col_name for col_name in node.names if col_name in self.name_to_uuid
            }
            cols = {
                self.uuid_to_name[col._uuid]: col
                for col in self.partition_by
                if self.uuid_to_name[col._uuid] not in overwritten
            } | {
                name: Col(name, node, uid, val.dtype(), val.ftype())
                for name, val, uid in zip(
                    node.names, node.values, node.uuids, strict=True
                )
            }

            res.cols = {col._uuid: col for _, col in cols.items()}
            res.name_to_uuid = {name: col._uuid for name, col in cols.items()}
            res.uuid_to_name = {uid: name for name, uid in res.name_to_uuid.items()}
            res.partition_by = []

        elif isinstance(node, verbs.Join):
            assert right_cache is not None

            res.cols = self.cols | right_cache.cols
            res.name_to_uuid = self.name_to_uuid | {
                name + node.suffix: uid
                for name, uid in right_cache.name_to_uuid.items()
            }
            res.uuid_to_name = {uid: name for name, uid in res.name_to_uuid.items()}

            res.derived_from = self.derived_from | right_cache.derived_from

        assert len(res.name_to_uuid) == len(res.uuid_to_name)

        res.derived_from = res.derived_from | {node}

        return res

    def get_selected_cols(self) -> list[Col]:
        return [self.cols[uid] for uid in self.uuid_to_name.keys()]
