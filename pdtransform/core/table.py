import itertools
from typing import Generic, Iterable, TypeVar
from html import escape

import pdtransform.core.verbs as verbs
from pdtransform.core.column import Column, LambdaColumn
from pdtransform.core.expressions import SymbolicExpression
from pdtransform.core.table_impl import AbstractTableImpl

ImplT = TypeVar('ImplT', bound=AbstractTableImpl)
class Table(Generic[ImplT]):
    """
    All attributes of a table are columns except for the `_impl` attribute
    which is a reference to the underlying table implementation.
    """

    def __init__(self, implementation: ImplT):
        self._impl = implementation

    def __getitem__(self, key) -> SymbolicExpression[Column]:
        return SymbolicExpression(self._impl.get_col(key))

    def __setitem__(self, col, expr):
        """ Mutate a column
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
            raise KeyError(f'Invalid key {col}. Must be either a string, Column or LambdaColumn.')
        self._impl = (self >> verbs.mutate(**{col_name: expr}))._impl

    def __getattr__(self, name) -> SymbolicExpression[Column]:
        return SymbolicExpression(self._impl.get_col(name))

    def __iter__(self) -> Iterable[SymbolicExpression[LambdaColumn]]:
        return iter(SymbolicExpression(LambdaColumn(name)) for name, _ in self._impl.selected_cols())

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._impl == other._impl

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._impl)

    def __dir__(self):
        return sorted(self._impl.named_cols.fwd.keys())

    def _repr_html_(self) -> str | None:
        html = f"Table <code>{self._impl.name}</code> using <code>{self._impl.__class__.__name__}</code> backend:</br>"
        try:
            html += (self >> verbs.collect())._repr_html_()
        except Exception as e:
            html += f"</br><pre>Failed to collect table due to an exception:\n" \
                    f"{escape(e.__class__.__name__)}: {escape(str(e))}</pre>"
        return html