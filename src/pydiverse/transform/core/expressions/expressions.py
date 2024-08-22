from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic

from pydiverse.transform._typing import ImplT, T
from pydiverse.transform.core.dtypes import DType

if TYPE_CHECKING:
    from pydiverse.transform.core.expressions.translator import TypedValue
    from pydiverse.transform.core.table_impl import AbstractTableImpl


def expr_repr(it: Any):
    from pydiverse.transform.core.expressions import SymbolicExpression

    if isinstance(it, SymbolicExpression):
        return expr_repr(it._)
    if isinstance(it, BaseExpression):
        return it._expr_repr()
    if isinstance(it, (list, tuple)):
        return f"[{ ', '.join([expr_repr(e) for e in it]) }]"
    return repr(it)


_dunder_expr_repr = {
    "__add__": lambda lhs, rhs: f"({lhs} + {rhs})",
    "__radd__": lambda rhs, lhs: f"({lhs} + {rhs})",
    "__sub__": lambda lhs, rhs: f"({lhs} - {rhs})",
    "__rsub__": lambda rhs, lhs: f"({lhs} - {rhs})",
    "__mul__": lambda lhs, rhs: f"({lhs} * {rhs})",
    "__rmul__": lambda rhs, lhs: f"({lhs} * {rhs})",
    "__truediv__": lambda lhs, rhs: f"({lhs} / {rhs})",
    "__rtruediv__": lambda rhs, lhs: f"({lhs} / {rhs})",
    "__floordiv__": lambda lhs, rhs: f"({lhs} // {rhs})",
    "__rfloordiv__": lambda rhs, lhs: f"({lhs} // {rhs})",
    "__pow__": lambda lhs, rhs: f"({lhs} ** {rhs})",
    "__rpow__": lambda rhs, lhs: f"({lhs} ** {rhs})",
    "__mod__": lambda lhs, rhs: f"({lhs} % {rhs})",
    "__rmod__": lambda rhs, lhs: f"({lhs} % {rhs})",
    "__round__": lambda x, y=None: f"round({x}, {y})" if y else f"round({x})",
    "__pos__": lambda x: f"(+{x})",
    "__neg__": lambda x: f"(-{x})",
    "__abs__": lambda x: f"abs({x})",
    "__and__": lambda lhs, rhs: f"({lhs} & {rhs})",
    "__rand__": lambda rhs, lhs: f"({lhs} & {rhs})",
    "__or__": lambda lhs, rhs: f"({lhs} | {rhs})",
    "__ror__": lambda rhs, lhs: f"({lhs} | {rhs})",
    "__xor__": lambda lhs, rhs: f"({lhs} ^ {rhs})",
    "__rxor__": lambda rhs, lhs: f"({lhs} ^ {rhs})",
    "__invert__": lambda x: f"(~{x})",
    "__lt__": lambda lhs, rhs: f"({lhs} < {rhs})",
    "__le__": lambda lhs, rhs: f"({lhs} <= {rhs})",
    "__eq__": lambda lhs, rhs: f"({lhs} == {rhs})",
    "__ne__": lambda lhs, rhs: f"({lhs} != {rhs})",
    "__gt__": lambda lhs, rhs: f"({lhs} > {rhs})",
    "__ge__": lambda lhs, rhs: f"({lhs} >= {rhs})",
}


class BaseExpression:
    def _expr_repr(self) -> str:
        """String repr that, when executed, returns the same expression"""
        raise NotImplementedError


class Column(BaseExpression, Generic[ImplT]):
    __slots__ = ("name", "table", "dtype", "uuid")

    def __init__(self, name: str, table: ImplT, dtype: DType, uuid: uuid.UUID = None):
        self.name = name
        self.table = table
        self.dtype = dtype
        self.uuid = uuid or Column.generate_col_uuid()

    def __repr__(self):
        return f"<{self.table.name}.{self.name}({self.dtype})>"

    def _expr_repr(self) -> str:
        return f"{self.table.name}.{self.name}"

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


class LambdaColumn(BaseExpression):
    """Anonymous Column

    A lambda column is a column without an associated table or UUID. This means
    that it can be used to reference columns in the same pipe as it was created.

    Example:
      The following fails because `table.a` gets referenced before it gets created.
        table >> mutate(a = table.x) >> mutate(b = table.a)
      Instead you can use a lambda column to achieve this:
        table >> mutate(a = table.x) >> mutate(b = C.a)
    """

    __slots__ = "name"

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"<C.{self.name}>"

    def _expr_repr(self) -> str:
        return f"C.{self.name}"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("C", self.name))


class LiteralColumn(BaseExpression, Generic[T]):
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

    def _expr_repr(self) -> str:
        return repr(self)

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


class FunctionCall(BaseExpression):
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

    def _expr_repr(self) -> str:
        args = [expr_repr(e) for e in self.args] + [
            f"{k}={expr_repr(v)}" for k, v in self.kwargs.items()
        ]

        if self.name in _dunder_expr_repr:
            return _dunder_expr_repr[self.name](*args)

        if len(self.args) == 0:
            args_str = ", ".join(args)
            return f"f.{self.name}({args_str})"
        else:
            args_str = ", ".join(args[1:])
            return f"{args[0]}.{self.name}({args_str})"

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


class CaseExpression(BaseExpression):
    def __init__(
        self, switching_on: Any | None, cases: Iterable[tuple[Any, Any]], default: Any
    ):
        from pydiverse.transform.core.expressions.symbolic_expressions import (
            unwrap_symbolic_expressions,
        )

        # Unwrap all symbolic expressions in the input
        switching_on = unwrap_symbolic_expressions(switching_on)
        cases = unwrap_symbolic_expressions(list(cases))
        default = unwrap_symbolic_expressions(default)

        self.switching_on = switching_on
        self.cases = cases
        self.default = default

    def __repr__(self):
        if self.switching_on:
            return f"case({self.switching_on}, {self.cases}, default={self.default})"
        else:
            return f"case({self.cases}, default={self.default})"

    def _expr_repr(self) -> str:
        prefix = "f"
        if self.switching_on:
            prefix = expr_repr(self.switching_on)

        args = [expr_repr(case) for case in self.cases]
        args.append(f"default={expr_repr(self.default)}")
        return f"{prefix}.case({', '.join(args)})"

    def iter_children(self):
        if self.switching_on:
            yield self.switching_on

        for k, v in self.cases:
            yield k
            yield v

        yield self.default
