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


def count(expr: SymbolicExpression | None = None):
    if expr is None:
        return _sym_f_call("count")
    else:
        return _sym_f_call("count", expr)


def row_number(*, arrange: list, partition_by: list | None = None):
    return _sym_f_call("row_number", arrange=arrange, partition_by=partition_by)


def rank(*, arrange: list, partition_by: list | None = None):
    return _sym_f_call("rank", arrange=arrange, partition_by=partition_by)


def dense_rank(*, arrange: list, partition_by: list | None = None):
    return _sym_f_call("dense_rank", arrange=arrange, partition_by=partition_by)


def case(*cases: tuple[Any, Any], default: Any = None):
    case_expression = CaseExpression(
        switching_on=None,
        cases=cases,
        default=default,
    )

    return SymbolicExpression(case_expression)


def min(first: Any, *expr: Any):
    return _sym_f_call("__least", first, *expr)


def max(first: Any, *expr: Any):
    return _sym_f_call("__greatest", first, *expr)
