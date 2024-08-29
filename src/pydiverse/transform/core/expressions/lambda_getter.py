from __future__ import annotations

from pydiverse.transform.core.expressions import LambdaColumn
from pydiverse.transform.core.expressions.symbolic_expressions import SymbolicExpression

__all__ = ["C"]


class MC(type):
    def __getattr__(cls, name: str) -> SymbolicExpression:
        return SymbolicExpression(LambdaColumn(name))

    def __getitem__(cls, name: str) -> SymbolicExpression:
        return SymbolicExpression(LambdaColumn(name))


class C(metaclass=MC):
    pass
