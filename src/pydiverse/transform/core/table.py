from __future__ import annotations

from html import escape
from typing import Generic, Iterable, TypeVar

from pydiverse.transform.core import verbs
from pydiverse.transform.core.column import Column, LambdaColumn
from pydiverse.transform.core.expressions import SymbolicExpression
from pydiverse.transform.core.table_impl import AbstractTableImpl

ImplT = TypeVar("ImplT", bound=AbstractTableImpl)


class Table(Generic[ImplT]):
    """
    All attributes of a table are columns except for the `_impl` attribute
    which is a reference to the underlying table implementation.
    """

    def __init__(self, implementation: ImplT):
        self._impl = implementation

    def __getitem__(self, key) -> SymbolicExpression[Column]:
        if isinstance(key, SymbolicExpression):
            key = key._
        return SymbolicExpression(self._impl.get_col(key))

    def __setitem__(self, col, expr):
        """Mutate a column
        :param col: Either a str or SymbolicColumn
        """
        col_name = None

        if isinstance(col, SymbolicExpression):
            underlying = col._
            if isinstance(underlying, (Column, LambdaColumn)):
                col_name = underlying.name
        elif isinstance(col, str):
            col_name = col

        if not col_name:
            raise KeyError(
                f"Invalid key {col}. Must be either a string, Column or LambdaColumn."
            )
        self._impl = (self >> verbs.mutate(**{col_name: expr}))._impl

    def __getattr__(self, name) -> SymbolicExpression[Column]:
        return SymbolicExpression(self._impl.get_col(name))

    def __iter__(self) -> Iterable[SymbolicExpression[Column]]:
        # Capture current state (this allows modifying the table inside a loop)
        cols = [
            SymbolicExpression(self._impl.get_col(name))
            for name, _ in self._impl.selected_cols()
        ]
        return iter(cols)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self._impl == other._impl

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._impl)

    def __dir__(self):
        return sorted(self._impl.named_cols.fwd.keys())

    def __contains__(self, item):
        if isinstance(item, SymbolicExpression):
            item = item._
        if isinstance(item, LambdaColumn):
            return item.name in self._impl.named_cols.fwd
        if isinstance(item, Column):
            return item.uuid in self._impl.available_cols
        return False

    def __copy__(self):
        impl_copy = self._impl.copy()
        return self.__class__(impl_copy)

    def __str__(self):
        try:
            return (
                f"Table: {self._impl.name}, backend: {type(self._impl).__name__}\n"
                f"{self >> verbs.collect()}"
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
            html += (self >> verbs.collect())._repr_html_()
        except Exception as e:
            html += (
                "</br><pre>Failed to collect table due to an exception:\n"
                f"{escape(e.__class__.__name__)}: {escape(str(e))}</pre>"
            )
        return html

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")
