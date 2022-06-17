import typing

import pdtransform.core.verbs as verbs
from pdtransform.core.column import Column
from pdtransform.core.expressions.lambda_column import LambdaColumn
from pdtransform.core.table_impl import AbstractTableImpl


class Table:
    """
    All attributes of a table are columns except for the `_impl` attribute
    which is a reference to the underlying table implementation.
    """

    def __init__(self, implementation: AbstractTableImpl):
        self._impl = implementation

    def __getitem__(self, key) -> Column:
        return self._impl.get_col(key)

    def __setitem__(self, key, value):
        # TODO: Key doesn't have to be a Column but could also be a string
        self._impl = (self >> verbs.mutate(**{key._name: value}))._impl

    def __getattr__(self, name) -> Column:
        return self._impl.get_col(name)

    def __iter__(self) -> typing.Iterable[LambdaColumn]:
        return iter(LambdaColumn(name) for name, _ in self._impl.selected_cols())

    def __eq__(self, other):
        return self._impl == other._impl

    def __hash__(self):
        return hash(self._impl)