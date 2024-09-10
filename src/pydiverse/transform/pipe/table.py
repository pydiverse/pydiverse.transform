from __future__ import annotations

import copy
from collections.abc import Iterable
from html import escape
from typing import Generic

from pydiverse.transform._typing import ImplT
from pydiverse.transform.tree.col_expr import (
    Col,
    ColName,
)
from pydiverse.transform.tree.table_expr import TableExpr


class Table(TableExpr, Generic[ImplT]):
    """
    All attributes of a table are columns except for the `_impl` attribute
    which is a reference to the underlying table implementation.
    """

    # TODO: define exactly what can be given for the two
    def __init__(self, resource, backend=None, *, name: str | None = None):
        import polars as pl

        from pydiverse.transform.backend import (
            PolarsImpl,
            SqlAlchemy,
            SqlImpl,
            TableImpl,
        )

        if isinstance(resource, (pl.DataFrame, pl.LazyFrame)):
            self._impl = PolarsImpl(resource)
        elif isinstance(resource, TableImpl):
            self._impl = resource
        elif isinstance(resource, str):
            if isinstance(backend, SqlAlchemy):
                self._impl = SqlImpl(resource, backend)
                if name is None:
                    name = self._impl.table.name

        if self._impl is None:
            raise AssertionError

        self.name = name
        self.schema = self._impl.schema()

    def __getitem__(self, key: str) -> Col:
        if not isinstance(key, str):
            raise TypeError(
                f"argument to __getitem__ (bracket `[]` operator) on a Table must be a "
                f"str, got {type(key)} instead."
            )
        col = super().__getitem__(key)
        col.dtype = self.schema[key]
        return col

    def __getattr__(self, name: str) -> Col:
        col = super().__getattr__(name)
        col.dtype = self.schema[name]
        return col

    def __iter__(self) -> Iterable[Col]:
        return iter(self.cols())

    def __contains__(self, item: str | Col | ColName):
        if isinstance(item, (Col, ColName)):
            item = item.name
        return item in self.col_names()

    def __str__(self):
        try:
            from pydiverse.transform.backend.targets import Polars
            from pydiverse.transform.pipe.verbs import export

            return (
                f"Table: {self.name}, backend: {type(self._impl).__name__}\n"
                f"{self >> export(Polars())}"
            )
        except Exception as e:
            return (
                f"Table: {self.name}, backend: {type(self._impl).__name__}\n"
                "failed to collect table due to an exception. "
                f"{type(e).__name__}: {str(e)}"
            )

    def _repr_html_(self) -> str | None:
        html = (
            f"Table <code>{self.name}</code> using"
            f" <code>{type(self._impl).__name__}</code> backend:</br>"
        )
        try:
            from pydiverse.transform.backend.targets import Polars
            from pydiverse.transform.pipe.verbs import export

            # TODO: For lazy backend only show preview (eg. take first 20 rows)
            html += (self >> export(Polars()))._repr_html_()
        except Exception as e:
            html += (
                "</br><pre>Failed to collect table due to an exception:\n"
                f"{escape(e.__class__.__name__)}: {escape(str(e))}</pre>"
            )
        return html

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def cols(self) -> list[Col]:
        return [Col(name, self) for name in self._impl.col_names()]

    def col_names(self) -> list[str]:
        return self._impl.col_names()

    def clone(self) -> tuple[TableExpr, dict[TableExpr, TableExpr]]:
        cloned = copy.copy(self)
        cloned._impl = cloned._impl.clone()
        return cloned, {self: cloned}
