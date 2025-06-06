# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

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

    def iter_subtree(self) -> Iterable["AstNode"]: ...
