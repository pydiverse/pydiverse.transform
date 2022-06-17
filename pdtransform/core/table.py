from .table_impl import AbstractTableImpl


class Table:
    """
    All attributes of a table are columns except for the `_impl` attribute
    which is a reference to the underlying table implementation.
    """

    def __init__(self, implementation: AbstractTableImpl):
        self._impl = implementation

    def __getitem__(self, key):
        return self._impl.get_col(key)

    def __getattr__(self, name):
        return self._impl.get_col(name)

    def __eq__(self, other):
        return self._impl == other._impl

    def __hash__(self):
        return hash(self._impl)