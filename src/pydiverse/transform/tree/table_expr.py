from __future__ import annotations

from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.tree import col_expr
from pydiverse.transform.tree.dtypes import Dtype


class TableExpr:
    __slots__ = ["name", "_schema", "_group_by", "_needed_cols"]

    def __init__(
        self,
        name: str,
        _schema: dict[str, tuple[Dtype, Ftype]],
        _group_by: list[col_expr.Col],
    ):
        self.name = name
        self._schema = _schema
        self._group_by = _group_by
        self._needed_cols: list[col_expr.Col] = []

    __slots__ = ["name", "_schema", "_group_by", "_needed_cols"]

    def __getitem__(self, key: str) -> col_expr.Col:
        if not isinstance(key, str):
            raise TypeError(
                f"argument to __getitem__ (bracket `[]` operator) on a Table must be a "
                f"str, got {type(key)} instead."
            )
        return col_expr.Col(key, self)

    def __getattr__(self, name: str) -> col_expr.Col:
        if name in ("__copy__", "__deepcopy__", "__setstate__", "__getstate__"):
            # for hasattr to work correctly on dunder methods
            raise AttributeError
        return col_expr.Col(name, self)

    def __eq__(self, rhs):
        if not isinstance(rhs, TableExpr):
            return False
        return id(self) == id(rhs)

    def __hash__(self):
        return id(self)

    def schema(self):
        return {name: val[0] for name, val in self._schema}

    def clone(self) -> tuple[TableExpr, dict[TableExpr, TableExpr]]: ...
