from __future__ import annotations

from pydiverse.transform.tree import col_expr


class TableExpr:
    name: str | None

    __slots__ = ["name"]

    def __getitem__(self, key: str) -> col_expr.Col:
        if not isinstance(key, str):
            raise TypeError(
                f"argument to __getitem__ (bracket `[]` operator) on a Table must be a "
                f"str, got {type(key)} instead."
            )
        return col_expr.Col(key, self)

    def __getattr__(self, name: str) -> col_expr.Col:
        if name in ("__copy__", "__deepcopy__", "__setstate__", "__getstate__"):
            # for hasattr to work correctly on dunder methods (e.g. __copy__)
            raise AttributeError
        return col_expr.Col(name, self)

    def __eq__(self, rhs):
        if not isinstance(rhs, TableExpr):
            return False
        return id(self) == id(rhs)

    def __hash__(self):
        return id(self)
