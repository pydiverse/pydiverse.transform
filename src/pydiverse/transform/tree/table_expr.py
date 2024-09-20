from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.tree import col_expr
from pydiverse.transform.tree.dtypes import Dtype


class TableExpr:
    __slots__ = [
        "name",
        "_schema",
        "_select",
        "_partition_by",
        "_name_to_uuid",
    ]
    # _schema stores the data / function types of all columns in the current C-space
    # (i.e. the ones accessible via `C.`). _select stores the columns that are actually
    # in the table (i.e. the ones accessible via `table.` and that are exported).

    def __init__(
        self,
        name: str,
        _schema: dict[str, tuple[Dtype, Ftype]],
        _select: list[col_expr.Col],
        _partition_by: list[col_expr.Col],
        _name_to_uuid: dict[str, UUID],
    ):
        self.name = name
        self._schema = _schema
        self._select = _select
        self._partition_by = _partition_by
        self._name_to_uuid = _name_to_uuid

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

    def cols(self) -> list[col_expr.Col]:
        return [col_expr.Col(name, self) for name in self._schema.keys()]

    def col_names(self) -> list[str]:
        return list(self._schema.keys())

    def schema(self) -> dict[str, Dtype]:
        return {
            name: val[0]
            for name, val in self._schema.items()
            if name in set(self._select)
        }

    def col_type(self, col_name: str) -> Dtype:
        return self._schema[col_name][0]

    def _clone(self) -> tuple[TableExpr, dict[TableExpr, TableExpr]]: ...

    def _iter_descendants(self) -> Iterable[TableExpr]: ...
