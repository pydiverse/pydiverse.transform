# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import copy
import dataclasses
from pprint import pformat
from typing import Optional
from uuid import UUID

from pydiverse.transform._internal import errors
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

    backend: type[TableImpl]

    def __repr__(self) -> str:
        return (
            "Cache(\n"
            + "  name_to_uuid="
            + f"""{
                "{" + pformat(self.name_to_uuid, indent=16)[16:]
                if len(self.name_to_uuid) > 1
                else pformat(self.name_to_uuid)
            }"""
            + ",\n"
            + "  uuid_to_name="
            + f"""{
                "{" + pformat(self.uuid_to_name, indent=16)[16:]
                if len(self.uuid_to_name) > 1
                else pformat(self.name_to_uuid)
            }"""
            + ",\n"
            + f"  partition_by={self.partition_by}\n"
            + "  derived_from={"
            + f"{', '.join(d.short_name() for d in self.derived_from)}"
            + "},\n"
            + "  cols={"
            + f"""{
                pformat(
                    {uid: col.ast_repr(depth=0) for uid, col in self.cols.items()},
                    indent=8,
                )[8:]
            }"""
            + ",\n"
            + f"  limit={self.limit},\n"
            + f"  group_by={self.group_by},\n"
            + f"  is_filtered={self.is_filtered},\n)"
        )

    # For a column to be usable in an expression, the table it comes from must be an
    # ancestor of the current table AND the column's UUID must be in `all_cols` of the
    # current table (the latter is necessary because not every column is usable after
    # `summarize``)

    @staticmethod
    def from_ast(node: AstNode) -> "Cache":
        if isinstance(node, verbs.Verb):
            if isinstance(node, verbs.Join | verbs.Union):
                return Cache.from_ast(node.child).update(node, right_cache=Cache.from_ast(node.right))
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
            backend=type(node),
        )

    def update(self, node: verbs.Verb, *, right_cache: Optional["Cache"] = None) -> "Cache":
        """
        Returns a new cache for `node`, assuming `self` is the cache of `node.child`.
        Does not modify `self`.
        """

        res = copy.copy(self)

        if isinstance(node, verbs.Alias):
            if node.uuid_map is not None:
                res.name_to_uuid = {name: node.uuid_map[uid] for name, uid in self.name_to_uuid.items()}
                res.uuid_to_name = {uid: name for name, uid in res.name_to_uuid.items()}
                res.cols = {
                    node.uuid_map[uid]: Col(col.name, node, node.uuid_map[uid], col._dtype, col._ftype)
                    for uid, col in self.cols.items()
                }
                res.partition_by = [node.uuid_map[uid] for uid in self.partition_by]
                res.derived_from = set()

        elif isinstance(node, verbs.Select):
            selected_uuids = set(col._uuid for col in node.select)
            res.uuid_to_name = {uid: name for uid, name in self.uuid_to_name.items() if uid in selected_uuids}
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
                for name, val, uid in zip(node.names, node.values, node.uuids, strict=True)
            }
            res.name_to_uuid = self.name_to_uuid | {name: uid for name, uid in zip(node.names, node.uuids, strict=True)}
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
            overwritten = {col_name for col_name in node.names if col_name in self.name_to_uuid}
            cols = {
                self.uuid_to_name[uid]: self.cols[uid]
                for uid in self.partition_by
                if self.uuid_to_name[uid] not in overwritten
            } | {
                name: Col(name, node, uid, val.dtype(), val.ftype())
                for name, val, uid in zip(node.names, node.values, node.uuids, strict=True)
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
            res.name_to_uuid = self.name_to_uuid | right_cache.name_to_uuid
            res.uuid_to_name = {uid: name for name, uid in res.name_to_uuid.items()}

            res.derived_from = self.derived_from | right_cache.derived_from
            res.limit = 0
            res.group_by = set()

        elif isinstance(node, verbs.Union):
            assert right_cache is not None

            # For union, visible columns must match (validated in verb function)
            # Hidden columns: are removed (we don't keep names for them and it is unlike they match in uuid)
            res.cols = {uid: col for uid, col in self.cols.items() if uid in self.uuid_to_name}
            # Visible columns should match, so we keep left table's name_to_uuid
            # (right table's visible columns are the same by validation)
            res.name_to_uuid = self.name_to_uuid.copy()
            res.uuid_to_name = self.uuid_to_name.copy()

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
    # subquery. (so `self` is the old cache and `node` the new AST).
    # Either returns a reason for the subquery or None.
    def requires_subquery(self, node: verbs.Verb) -> str | None:
        if self.backend.backend_name == "polars":
            return None

        if (
            isinstance(
                node,
                verbs.Filter | verbs.Summarize | verbs.Arrange | verbs.GroupBy | verbs.Join | verbs.Union,
            )
            and self.limit != 0
        ):
            return f"`{node.__class__.__name__.lower()}` after `slice_head`"

        if isinstance(node, verbs.Mutate) and any(
            any(
                col.ftype(agg_is_window=True) in (Ftype.WINDOW, Ftype.AGGREGATE)
                for col in fn.iter_subtree_postorder()
                if isinstance(col, Col)
            )
            for fn in node.iter_col_nodes()
            if (isinstance(fn, ColFn) and fn.op.ftype in (Ftype.AGGREGATE, Ftype.WINDOW))
        ):
            return "nested window / aggregation functions in `mutate`"

        if isinstance(node, verbs.Filter) and any(
            col.ftype(agg_is_window=True) == Ftype.WINDOW for col in node.iter_col_nodes() if isinstance(col, Col)
        ):
            return "window function in `filter`"

        if isinstance(node, verbs.Summarize):
            if self.group_by and self.group_by != set(self.partition_by):
                return "nested summarize"
            if any(
                (col.ftype(agg_is_window=False) in (Ftype.WINDOW, Ftype.AGGREGATE))
                for col in node.iter_col_nodes()
                if isinstance(col, Col)
            ):
                return "nested window / aggregation functions in `summarize`"
            if any(self.cols[uid].ftype() == Ftype.WINDOW for uid in self.partition_by):
                return "window function among grouping columns"

        if isinstance(node, verbs.Join):
            if self.group_by:
                return "join with a grouped table"

            if (node.how == "full" or (node.child not in self.derived_from and node.how == "left")) and any(
                types.is_const(self.cols[uid].dtype()) for uid in self.uuid_to_name.keys()
            ):
                return "left / full join with a table containing a constant column"

            if any(self.cols[uid].ftype() == Ftype.WINDOW for uid in self.uuid_to_name.keys()):
                return "join with a table containing window function expression"

            if any(
                col.ftype() != Ftype.ELEMENT_WISE and col._uuid in self.cols
                for col in node.on.iter_subtree_postorder()
                if isinstance(col, Col)
            ):
                return "window / aggregation functions in join condition"

            if self.is_filtered and node.how == "full":
                return "full join with a filtered table"

        if isinstance(node, verbs.Union):
            if self.group_by:
                return "union with a grouped table"

            if any(self.cols[uid].ftype() == Ftype.WINDOW for uid in self.uuid_to_name.keys()):
                return "union with a table containing window function expression"

        return None

    def selected_cols(self) -> list[Col]:
        return [self.cols[uid] for uid in self.uuid_to_name.keys()]


def transfer_col_references(table, ref_source):
    """
    Transfers the column references from `ref_source` to `table`.

    The returned table contains all selected columns of `table`, but its columns are
    now referenced by the columns from `ref_source`. All column names selected in
    `table` must also be present in `ref_source`.

    :param table:
        The table from which the data is taken.

    :param ref_source:
        The table from which the column references are taken.


    Examples
    --------
    **Materialization without breaking the functional flow.** Say you have a function
    `your_materialization_fn` that writes a transform table to a database and returns a
    transform table again. Then you can define a custom verb

    >>> @verb
    ... def materialize(table) -> pdt.Table:
    ...     new = your_materialization_fn(table)
    ...     return pdt.transfer_col_references(new, table)

    With this verb, it is possible to write

    >>> t = pdt.Table(dict(a=[1, 2, 5], b=["x", "y", "z"]), name="t")
    >>> t >> filter(t.a >= 2) >> materialize() >> mutate(z=t.a + t.b.str.len())
    Table `t` (backend: polars)
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ z   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ str ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 2   ┆ y   ┆ 3   │
    │ 5   ┆ z   ┆ 6   │
    └─────┴─────┴─────┘

    Without `transfer_col_references`, it would not be possible to use `t.a` and `t.b`
    in the `mutate`. (Of course, you would normally have a SQL backend when
    materializing, not a polars backend like in the example here.)
    """
    from pydiverse.transform._internal.pipe.table import Table
    from pydiverse.transform._internal.tree.verbs import Alias

    errors.check_arg_type(Table, "transfer_col_references", "table", table)
    errors.check_arg_type(Table, "transfer_col_references", "ref_source", ref_source)

    if (col := next((col for col in table if col.name not in ref_source), None)) is not None:
        raise ValueError(
            f"column {col.ast_repr()} of the table `{table._ast.short_name()}` does "
            "not exist in the reference source table "
            f"`{ref_source._ast.short_name()}`"
        )

    new = copy.copy(table)
    new._ast = Alias(
        new._ast,
        uuid_map={uid: ref_source._cache.name_to_uuid[name] for uid, name in table._cache.uuid_to_name.items()},
    )
    new._cache = table._cache.update(new._ast)

    return new
