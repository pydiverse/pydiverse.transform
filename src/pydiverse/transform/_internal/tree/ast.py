# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
from collections.abc import Iterable
from uuid import UUID


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
        return self.ast_repr(verb_depth=7, expr_depth=2)

    def ast_repr(self, verb_depth: int = -1, expr_depth: int = -1) -> str:
        from pydiverse.transform._internal.backend.table_impl import TableImpl
        from pydiverse.transform._internal.tree.verbs import Alias

        source_tbls = set(
            tbl
            for tbl in self.iter_subtree_preorder()
            if isinstance(tbl, TableImpl | Alias)
        )
        table_display_name_map: dict[TableImpl, str] = dict()
        used = set()
        for tbl in source_tbls:
            display_name = tbl.name or "tbl"
            # try to achieve valid python identifier names
            display_name.replace(" ", "_")
            display_name.replace(".", "_")
            display_name.replace("-", "_")

            if display_name not in used:
                used.add(tbl.name)
                table_display_name_map[tbl] = display_name
            else:
                cnt = 1
                while f"{display_name}_{cnt}" in used:
                    cnt += 1
                table_display_name_map[tbl] = f"{display_name}_{cnt}"
                used.add(f"{display_name}_{cnt}")

        unformatted = "\n".join(
            f"{display_name} = {tbl._table_def_repr()}"
            for tbl, display_name in table_display_name_map.items()
            if isinstance(tbl, TableImpl)
        ) + (
            "\n\n("
            + self._unformatted_ast_repr(verb_depth, expr_depth, table_display_name_map)
            + ")"
        )
        try:
            proc = subprocess.run(
                ["ruff", "format", "-"],  # '-' tells Ruff to read from stdin
                input=unformatted.encode(),
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(unformatted)
            print(e.stderr.decode())
            raise e
        return proc.stdout.decode()

    def short_name(self) -> str:
        raise NotImplementedError()

    def _unformatted_ast_repr(self, verb_depth: int, expr_depth: int, display_name_map):
        raise NotImplementedError()

    def _ast_node_repr(self, expr_depth: int, display_name_map):
        raise NotImplementedError()
