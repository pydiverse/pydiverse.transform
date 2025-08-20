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

    def ast_repr(
        self, verb_depth: int = -1, expr_depth: int = -1, *, oneline: bool = False
    ) -> str:
        unformatted = (
            "("
            + self._unformatted_ast_repr(verb_depth, expr_depth, oneline=oneline)
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

    def _unformatted_ast_repr(
        self, verb_depth: int = -1, expr_depth: int = -1, *, oneline: bool
    ):
        raise NotImplementedError()

    def _ast_node_repr(self, expr_depth: int):
        raise NotImplementedError()
