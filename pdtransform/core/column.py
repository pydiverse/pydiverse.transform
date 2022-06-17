import uuid

from .expressions.expression import SymbolicExpression


def generate_col_uuid():
    return uuid.uuid1()


class Column(SymbolicExpression):

    __slots__ = ('_name', '_table', '_dtype', '_uuid')

    _name: str
    _table: object
    _dtype: str
    _uuid: uuid.UUID

    def __init__(self, name, table, dtype: str, uuid: uuid.UUID = None):
        self._name = name
        self._table = table
        self._dtype = dtype
        self._uuid = uuid or generate_col_uuid()
        super().__init__()

    def __repr__(self):
        return f'<{self._table.name}.{self._name}({self._dtype})>'