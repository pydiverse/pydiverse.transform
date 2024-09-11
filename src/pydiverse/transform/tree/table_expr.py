from __future__ import annotations

from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.tree.col_expr import Col
from pydiverse.transform.tree.dtypes import Dtype


class TableExpr:
    name: str | None
    _schema: dict[str, tuple[Dtype, Ftype]]

    __slots__ = ["name", "schema", "ftype_schema"]

    def __getitem__(self, key: str) -> Col:
        if not isinstance(key, str):
            raise TypeError(
                f"argument to __getitem__ (bracket `[]` operator) on a Table must be a "
                f"str, got {type(key)} instead."
            )
        return Col(key, self)

    def __getattr__(self, name: str) -> Col:
        if name in ("__copy__", "__deepcopy__", "__setstate__", "__getstate__"):
            # for hasattr to work correctly on dunder methods
            raise AttributeError
        return Col(name, self)

    def __eq__(self, rhs):
        if not isinstance(rhs, TableExpr):
            return False
        return id(self) == id(rhs)

    def __hash__(self):
        return id(self)

    def schema(self):
        return {name: val[0] for name, val in self._schema}

    def clone(self) -> tuple[TableExpr, dict[TableExpr, TableExpr]]: ...
