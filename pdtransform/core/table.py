import itertools
from typing import TypeVar, Generic, Iterable

import pdtransform.core.verbs as verbs
from pdtransform.core.column import Column
from pdtransform.core.expressions.lambda_column import LambdaColumn
from pdtransform.core.table_impl import AbstractTableImpl


ImplT = TypeVar('ImplT', bound=AbstractTableImpl)
class Table(Generic[ImplT]):
    """
    All attributes of a table are columns except for the `_impl` attribute
    which is a reference to the underlying table implementation.
    """

    def __init__(self, implementation: ImplT):
        self._impl = implementation

    def __getitem__(self, key) -> Column:
        return self._impl.get_col(key)

    def __setitem__(self, col, expr):
        """ Mutate a column
        :param col: Either a str, Column or LambdaColumn
        """
        if isinstance(col, (Column, LambdaColumn)):
            col_name = col._name
        elif isinstance(col, str):
            col_name = col
        else:
            raise KeyError(f'Invalid key {col}. Must be either a string, Column or LambdaColumn.')
        self._impl = (self >> verbs.mutate(**{col_name: expr}))._impl

    def __getattr__(self, name) -> Column:
        return self._impl.get_col(name)

    def __dir__(self):
        return itertools.chain(self._impl.columns.keys(), ('_impl', ))

    def __iter__(self) -> Iterable[LambdaColumn]:
        return iter(LambdaColumn(name) for name, _ in self._impl.selected_cols())

    def __eq__(self, other):
        return self._impl == other._impl

    def __hash__(self):
        return hash(self._impl)