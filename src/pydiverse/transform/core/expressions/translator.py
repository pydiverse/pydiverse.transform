from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Generic

from pydiverse.transform._typing import T
from pydiverse.transform.core import column
from pydiverse.transform.core.expressions import expressions
from pydiverse.transform.core.ops import Operator, OPType, registry
from pydiverse.transform.core.ops.dtypes import DType


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
            raise ValueError(
                "An exception occured while trying to translate the expression"
                f" '{expr}':\n{e}"
            ) from e

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
            raise ValueError(
                "An exception occured while trying to translate the expression"
                f" '{expr}':\n{e}"
            ) from e

    def _translate(self, expr, accept_literal_col=True, **kwargs):
        if isinstance(expr, column.Column):
            return self._translate_col(expr, **kwargs)

        if isinstance(expr, column.LiteralColumn):
            if accept_literal_col:
                return self._translate_literal_col(expr, **kwargs)
            else:
                raise ValueError("Literal columns aren't allowed in this context.")

        if isinstance(expr, expressions.FunctionCall):
            operator = self.operator_registry.get_operator(expr.name)

            # Mutate function call arguments using operator
            expr_args, expr_kwargs = operator.mutate_args(expr.args, expr.kwargs)
            expr = expressions.FunctionCall(expr.name, *expr_args, **expr_kwargs)

            op_args, op_kwargs, context_kwargs = self.__translate_function_arguments(
                expr, operator, **kwargs
            )

            if op_kwargs:
                raise NotImplementedError

            signature = tuple(arg.dtype for arg in op_args)
            implementation = self.operator_registry.get_implementation(
                expr.name, signature
            )

            return self._translate_function(
                expr, implementation, op_args, context_kwargs, **kwargs
            )

        if literal_result := self._translate_literal(expr, **kwargs):
            return literal_result

        raise NotImplementedError(
            f"Couldn't find a way to translate object of type {type(expr)} with value"
            f" {expr}."
        )

    def _translate_col(self, expr: column.Column, **kwargs) -> T:
        raise NotImplementedError

    def _translate_literal_col(self, expr: column.LiteralColumn, **kwargs) -> T:
        raise NotImplementedError

    def _translate_function(
        self,
        expr: expressions.FunctionCall,
        implementation: registry.TypedOperatorImpl,
        op_args: list[T],
        context_kwargs: dict[str, Any],
        **kwargs,
    ) -> T:
        raise NotImplementedError

    def _translate_literal(self, expr, **kwargs) -> T:
        raise NotImplementedError

    def __translate_function_arguments(
        self, expr: expressions.FunctionCall, operator: Operator, **kwargs
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
        if isinstance(expr, expressions.FunctionCall):
            f = expressions.FunctionCall(
                expr.name,
                *(transform(arg) for arg in expr.args),
                **{k: transform(v) for k, v in expr.kwargs.items()},
            )
            return replace(f)
        else:
            return replace(expr)

    return transform(expr)
