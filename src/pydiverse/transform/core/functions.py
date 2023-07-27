from __future__ import annotations

from typing import Any

from pydiverse.transform.core.expressions import (
    CaseExpression,
    FunctionCall,
    SymbolicExpression,
)

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


def case(*cases: tuple[Any, Any], default: Any = None):
    case_expression = CaseExpression(
        switching_on=None,
        cases=cases,
        default=default,
    )

    return SymbolicExpression(case_expression)
