from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from pydiverse.transform._typing import T
from pydiverse.transform.core import registry
from pydiverse.transform.core.expressions import (
    CaseExpression,
    Column,
    FunctionCall,
    LiteralColumn,
)
from pydiverse.transform.ops.core import Operator, OPType
from pydiverse.transform.util import reraise

if TYPE_CHECKING:
    from pydiverse.transform.core.dtypes import DType


# Basic container to store value and associated type metadata
@dataclass
class TypedValue(Generic[T]):
    value: T
    dtype: DType
    ftype: OPType = dataclasses.field(default=OPType.EWISE)

    def __iter__(self):
        return iter((self.value, self.dtype))


class Translator(Generic[T]):
    def translate(self, expr, **kwargs) -> T:
        """Translate an expression recursively."""
        try:
            return bottom_up_replace(expr, lambda e: self._translate(e, **kwargs))
        except Exception as e:
            msg = f"This exception occurred while translating the expression: {expr}"
            reraise(e, suffix=msg)

    def _translate(self, expr, **kwargs) -> T:
        """Translate an expression non recursively."""
        raise NotImplementedError


class DelegatingTranslator(Translator[T], Generic[T]):
    """
    Translator that dispatches to different translate functions based on
    the type of the expression.
    """

    def __init__(self, operator_registry: registry.OperatorRegistry):
        self.operator_registry = operator_registry

    def translate(self, expr, **kwargs):
        """Translate an expression recursively."""
        try:
            return self._translate(expr, **kwargs)
        except Exception as e:
            msg = f"This exception occurred while translating the expression: {expr}"
            reraise(e, suffix=msg)

    def _translate(self, expr, **kwargs):
        if isinstance(expr, Column):
            return self._translate_col(expr, **kwargs)

        if isinstance(expr, LiteralColumn):
            return self._translate_literal_col(expr, **kwargs)

        if isinstance(expr, FunctionCall):
            operator = self.operator_registry.get_operator(expr.name)
            expr = FunctionCall(expr.name, *expr.args, **expr.kwargs)

            op_args, op_kwargs, context_kwargs = self._translate_function_arguments(
                expr, operator, **kwargs
            )

            if op_kwargs:
                raise NotImplementedError

            signature = tuple(arg.dtype for arg in op_args)
            implementation = self.operator_registry.get_implementation(
                expr.name, signature
            )

            return self._translate_function(
                implementation, op_args, context_kwargs, **kwargs
            )

        if isinstance(expr, CaseExpression):
            switching_on = (
                self._translate(expr.switching_on, **{**kwargs, "context": "case_val"})
                if expr.switching_on is not None
                else None
            )

            cases = []
            for cond, value in expr.cases:
                cases.append(
                    (
                        self._translate(cond, **{**kwargs, "context": "case_cond"}),
                        self._translate(value, **{**kwargs, "context": "case_val"}),
                    )
                )

            default = self._translate(expr.default, **{**kwargs, "context": "case_val"})
            return self._translate_case(expr, switching_on, cases, default, **kwargs)

        if literal_result := self._translate_literal(expr, **kwargs):
            return literal_result

        raise NotImplementedError(
            f"Couldn't find a way to translate object of type {type(expr)} with value"
            f" {expr}."
        )

    def _translate_col(self, col: Column, **kwargs) -> T:
        raise NotImplementedError

    def _translate_literal_col(self, col: LiteralColumn, **kwargs) -> T:
        raise NotImplementedError

    def _translate_function(
        self,
        implementation: registry.TypedOperatorImpl,
        op_args: list[T],
        context_kwargs: dict[str, Any],
        **kwargs,
    ) -> T:
        raise NotImplementedError

    def _translate_case(
        self,
        expr: CaseExpression,
        switching_on: T | None,
        cases: list[tuple[T, T]],
        default: T,
        **kwargs,
    ) -> T:
        raise NotImplementedError

    def _translate_literal(self, expr, **kwargs) -> T:
        raise NotImplementedError

    def _translate_function_arguments(
        self, expr: FunctionCall, operator: Operator, **kwargs
    ) -> tuple[list[T], dict[str, T], dict[str, Any]]:
        op_args = [self._translate(arg, **kwargs) for arg in expr.args]
        op_kwargs = {}
        context_kwargs = {}

        for k, v in expr.kwargs.items():
            if k in operator.context_kwargs:
                context_kwargs[k] = v
            else:
                op_kwargs[k] = self._translate(v, **kwargs)

        return op_args, op_kwargs, context_kwargs


def bottom_up_replace(expr, replace):
    def transform(expr):
        if isinstance(expr, FunctionCall):
            f = FunctionCall(
                expr.name,
                *(transform(arg) for arg in expr.args),
                **{k: transform(v) for k, v in expr.kwargs.items()},
            )
            return replace(f)

        if isinstance(expr, CaseExpression):
            c = CaseExpression(
                switching_on=transform(expr.switching_on),
                cases=[(transform(k), transform(v)) for k, v in expr.cases],
                default=transform(expr.default),
            )
            return replace(c)

        return replace(expr)

    return transform(expr)
