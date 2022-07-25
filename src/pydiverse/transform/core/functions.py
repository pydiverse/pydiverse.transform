from __future__ import annotations

from .expressions import FunctionCall, SymbolicExpression

__all__ = [
    "count",
    "row_number",
]


def _sym_f_call(name, *args, **kwargs) -> SymbolicExpression[FunctionCall]:
    return SymbolicExpression(FunctionCall(name, *args, **kwargs))


def count(expr: SymbolicExpression = None):
    if expr is None:
        return _sym_f_call("count")
    else:
        return _sym_f_call("count", expr)


def row_number(*, arrange: list):
    return _sym_f_call("row_number", arrange=arrange)
