# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from collections.abc import Callable, Iterable
from uuid import UUID

from pydiverse.transform._internal.util.warnings import warn


class AstNode:
    __slots__ = ["name"]

    name: str

    def clone(self) -> "AstNode":
        return self._clone()[0]

    def _clone(
        self,
    ) -> tuple["AstNode", dict["AstNode", "AstNode"], dict[UUID, UUID]]: ...

    def iter_subtree_postorder(self) -> Iterable["AstNode"]: ...

    def iter_subtree_preorder(self) -> Iterable["AstNode"]: ...

    def __repr__(self) -> str:
        return self.short_name()

    # Formatted, almost source code like representation of the AST.
    def ast_repr(self, verb_depth: int = -1, expr_depth: int = -1) -> str:
        from pydiverse.transform._internal.backend.table_impl import TableImpl
        from pydiverse.transform._internal.tree.col_expr import Col
        from pydiverse.transform._internal.tree.verbs import Alias, Verb

        def next_nd(root: AstNode, cond: Callable[["AstNode"], bool]):
            for nd in root.iter_subtree_preorder():
                if cond(nd):
                    return nd

        source_tbls = set(nd for nd in self.iter_subtree_preorder() if isinstance(nd, TableImpl | Alias))
        # Add required ASTs from aligned columns (they need not be in the subtree of
        # `self`)
        source_tbls |= set(
            next_nd(col._ast, lambda x: isinstance(x, Alias | TableImpl))
            for col in itertools.chain(
                *(nd.iter_col_nodes() for nd in self.iter_subtree_preorder() if isinstance(nd, Verb))
            )
            if isinstance(col, Col)
        )

        table_display_name_map: dict[TableImpl, str] = dict()
        used = set()
        for nd in source_tbls:
            display_name = nd.name or "tbl"
            # try to achieve valid python identifier names
            display_name = display_name.replace(" ", "_").replace(".", "_").replace("-", "_")

            if display_name not in used:
                used.add(display_name)
                table_display_name_map[nd] = display_name
            else:
                cnt = 1
                while f"{display_name}_{cnt}" in used:
                    cnt += 1
                table_display_name_map[nd] = f"{display_name}_{cnt}"
                used.add(f"{display_name}_{cnt}")

        # Find the last source table / alias for every node in the AST and use the
        # corresponding name.
        for nd in self.iter_subtree_postorder():
            table_display_name_map[nd] = table_display_name_map[next_nd(nd, lambda x: x in table_display_name_map)]

            if isinstance(nd, Verb):
                for col in nd.iter_col_nodes():
                    if isinstance(col, Col):
                        table_display_name_map[col._ast] = table_display_name_map[
                            next_nd(col._ast, lambda x: x in table_display_name_map)
                        ]

        unformatted = "\n".join(
            f"{display_name} = {tbl._table_def_repr()}"
            for tbl, display_name in table_display_name_map.items()
            if isinstance(tbl, TableImpl)
        ) + ("\n\n(" + self._unformatted_ast_repr(verb_depth, expr_depth, table_display_name_map) + ")")
        try:
            import black

            formatted = black.format_str(unformatted, mode=black.Mode(line_length=120))
            return formatted
        except Exception:
            warn("Could not format AST representation with `black`.")
            return unformatted

    def short_name(self) -> str:
        raise NotImplementedError()

    # Recursive, builds up the verb chain.
    def _unformatted_ast_repr(self, verb_depth: int, expr_depth: int, display_name_map):
        raise NotImplementedError()

    # Just the verb call of a single verb, without `>>`.
    def _ast_node_repr(self, expr_depth: int, display_name_map):
        raise NotImplementedError()
