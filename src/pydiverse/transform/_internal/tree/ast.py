from __future__ import annotations

from collections.abc import Generator, Iterable
from uuid import UUID


class AstNode:
    __slots__ = ["name"]

    name: str

    def clone(self) -> AstNode:
        return self._clone()[0]

    def _clone(self) -> tuple[AstNode, dict[AstNode, AstNode], dict[UUID, UUID]]: ...

    def iter_subtree_postorder(self) -> Iterable[AstNode]: ...

    # Iterates over the AST nodes in the subtree of the current node using preorder
    # traversal. By sending `True`, one can tell the DFS to not explore the subtree
    # any further.
    def iter_subtree_preorder(self) -> Generator[AstNode, bool | None, None]: ...
