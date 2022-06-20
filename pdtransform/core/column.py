import uuid
from typing import Generic, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from pdtransform.core.table_impl import AbstractTableImpl


def generate_col_uuid():
    return uuid.uuid1()


ImplT = TypeVar('ImplT', bound = 'AbstractTableImpl')


class Column(Generic[ImplT]):
    __slots__ = ('name', 'table', 'dtype', 'uuid')

    def __init__(self, name: str, table: ImplT, dtype: str, uuid: uuid.UUID = None):
        self.name = name
        self.table = table
        self.dtype = dtype
        self.uuid = uuid or generate_col_uuid()

    def __repr__(self):
        return f'<{self.table.name}.{self.name}({self.dtype})>'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        # TODO: Determine what is the correct way to compare
        return (self.name == other.name and self.uuid == other.uuid)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.uuid)


class LambdaColumn:
    """Anonymous Column

    A lambda column is a column without an associated table or UUID. This means
    that it can be used to reference columns in the same pipe as it was created.

    Example:
      The following fails because `table.a` gets referenced before it gets created.
        table >> mutate(a = table.x) >> mutate(b = table.a)
      Instead you can use a lambda column to achieve this:
        table >> mutate(a = table.x) >> mutate(b = λ.a)
    """

    __slots__ = ('name')

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f'<λ.{self.name}>'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(('λ', self.name))
