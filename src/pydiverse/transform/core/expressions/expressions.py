from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Generic

from pydiverse.transform._typing import ImplT, T
from pydiverse.transform.core.dtypes import DType

if TYPE_CHECKING:
    from pydiverse.transform.core.expressions.translator import TypedValue
    from pydiverse.transform.core.table_impl import AbstractTableImpl


class Column(Generic[ImplT]):
    __slots__ = ("name", "table", "dtype", "uuid")

    def __init__(self, name: str, table: ImplT, dtype: DType, uuid: uuid.UUID = None):
        self.name = name
        self.table = table
        self.dtype = dtype
        self.uuid = uuid or Column.generate_col_uuid()

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

    @classmethod
    def generate_col_uuid(cls) -> uuid.UUID:
        return uuid.uuid1()


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


class LiteralColumn(Generic[T]):
    __slots__ = ("typed_value", "expr", "backend")

    def __init__(
        self,
        typed_value: TypedValue[T],
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


class FunctionCall:
    """
    AST node to represent a function / operator call.
    """

    def __init__(self, name: str, *args, **kwargs):
        from pydiverse.transform.core.expressions.symbolic_expressions import (
            unwrap_symbolic_expressions,
        )

        # Unwrap all symbolic expressions in the input
        args = unwrap_symbolic_expressions(args)
        kwargs = unwrap_symbolic_expressions(kwargs)

        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        args = [repr(e) for e in self.args] + [
            f"{k}={repr(v)}" for k, v in self.kwargs.items()
        ]
        return f'{self.name}({", ".join(args)})'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, self.args, tuple(self.kwargs.items())))

    def iter_children(self):
        yield from self.args
