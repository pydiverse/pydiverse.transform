from __future__ import annotations

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

    def __init__(self, implementation: ImplT):
        self._impl = implementation

    def __getitem__(self, key: str) -> Col:
        if not isinstance(key, str):
            raise TypeError(
                f"argument to __getitem__ (bracket `[]` operator) on a Table must be a "
                f"str, got {type(key)} instead."
            )
        return Col(key, self, self._impl.col_type(key))

    def __getattr__(self, name: str) -> Col:
        return Col(name, self, self._impl.col_type(name))

    def __iter__(self) -> Iterable[Col]:
        return iter(self.cols())

    def __contains__(self, item: str | Col | ColName):
        if isinstance(item, (Col, ColName)):
            item = item.name
        return item in self.col_names()

    def __copy__(self):
        impl_copy = self._impl.copy()
        return self.__class__(impl_copy)

    def __str__(self):
        try:
            return (
                f"Table: {self._impl.name}, backend: {type(self._impl).__name__}\n"
                f"{self._impl.to_polars().df}"
            )
        except Exception as e:
            return (
                f"Table: {self._impl.name}, backend: {type(self._impl).__name__}\n"
                "Failed to collect table due to an exception:\n"
                f"{type(e).__name__}: {str(e)}"
            )

    def _repr_html_(self) -> str | None:
        html = (
            f"Table <code>{self._impl.name}</code> using"
            f" <code>{type(self._impl).__name__}</code> backend:</br>"
        )
        try:
            # TODO: For lazy backend only show preview (eg. take first 20 rows)
            html += (self._impl.to_polars().df)._repr_html_()
        except Exception as e:
            html += (
                "</br><pre>Failed to collect table due to an exception:\n"
                f"{escape(e.__class__.__name__)}: {escape(str(e))}</pre>"
            )
        return html

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def cols(self) -> list[Col]:
        return [Col(name, self) for name in self._impl.cols()]

    def col_names(self) -> list[str]:
        return self._impl.cols()
