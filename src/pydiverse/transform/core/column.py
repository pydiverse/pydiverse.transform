from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Generic, Type, TypeVar

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from pydiverse.transform.core.expressions.translator import TypedValue
    from pydiverse.transform.core.table_impl import AbstractTableImpl


def generate_col_uuid():
    return uuid.uuid1()


ImplT = TypeVar("ImplT", bound="AbstractTableImpl")
LiteralT = TypeVar("LiteralT")


class Column(Generic[ImplT]):
    __slots__ = ("name", "table", "dtype", "uuid")

    def __init__(self, name: str, table: ImplT, dtype: str, uuid: uuid.UUID = None):
        self.name = name
        self.table = table
        self.dtype = dtype
        self.uuid = uuid or generate_col_uuid()

    def __repr__(self):
        return f"<{self.table.name}.{self.name}({self.dtype})>"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.uuid == other.uuid

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

    __slots__ = "name"

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"<λ.{self.name}>"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("λ", self.name))


class LiteralColumn(Generic[LiteralT]):
    __slots__ = ("typed_value", "expr", "backend")

    def __init__(
        self,
        typed_value: TypedValue[LiteralT],
        expr: Any,
        backend: type[AbstractTableImpl],
    ):
        self.typed_value = typed_value
        self.expr = expr
        self.backend = backend

    def __repr__(self):
        return f"<Lit: {self.expr} ({self.typed_value.dtype})>"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.typed_value == other.typed_value
            and self.expr == other.expr
            and self.backend == other.backend
        )

    def __ne__(self, other):
        return not self.__eq__(other)
