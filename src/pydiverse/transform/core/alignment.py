from __future__ import annotations

import inspect

from pydiverse.transform.core import column, table, table_impl
from pydiverse.transform.core.expressions import symbolic_expressions, util


def aligned(*, with_: str):
    """Decorator for aligned functions."""
    if callable(with_):
        raise ValueError("Decorator @aligned requires with_ argument.")

    def decorator(func):
        signature = inspect.signature(func)
        if not isinstance(with_, str):
            raise Exception(
                f"Argument 'with_' must be of type str, not '{type(with_).__name__}'."
            )
        if with_ not in signature.parameters:
            raise Exception(f"Function has no argument named '{with_}'")

        def wrapper(*args, **kwargs):
            # Execute func
            result = func(*args, **kwargs)
            if not isinstance(result, symbolic_expressions.SymbolicExpression):
                raise ValueError(
                    "Aligned function must return a symbolic expression not"
                    f" '{result}'."
                )

            # Extract the correct `with_` argument for eval_aligned
            bound_sig = signature.bind(*args, **kwargs)
            bound_sig.apply_defaults()

            alignment_param = bound_sig.arguments[with_]
            if isinstance(alignment_param, symbolic_expressions.SymbolicExpression):
                alignment_param = alignment_param._

            if isinstance(alignment_param, column.Column):
                aligned_with = alignment_param.table
            elif isinstance(
                alignment_param, (table.Table, table_impl.AbstractTableImpl)
            ):
                aligned_with = alignment_param
            else:
                raise NotImplementedError

            # Evaluate aligned
            return eval_aligned(result, with_=aligned_with)

        return wrapper

    return decorator


def eval_aligned(
    sexpr: symbolic_expressions.SymbolicExpression,
    with_: table_impl.AbstractTableImpl | table.Table = None,
    **kwargs,
) -> symbolic_expressions.SymbolicExpression[column.LiteralColumn]:
    """Evaluates an expression using the AlignedExpressionEvaluator."""

    expr = (
        sexpr._ if isinstance(sexpr, symbolic_expressions.SymbolicExpression) else sexpr
    )

    # Determine Backend
    backend = util.determine_expr_backend(expr)
    if backend is None:
        # TODO: Handle this case. Should return some value...
        raise NotImplementedError

    # Evaluate the function calls on the shared backend
    alignedEvaluator = backend.AlignedExpressionEvaluator(backend.operator_registry)
    result = alignedEvaluator.translate(expr, **kwargs)

    literal_column = column.LiteralColumn(
        typed_value=result, expr=expr, backend=backend
    )

    # Check if alignment condition holds
    if with_ is not None:
        if isinstance(with_, table.Table):
            with_ = with_._impl
        if not isinstance(with_, table_impl.AbstractTableImpl):
            raise ValueError(
                "'with_' must either be an instance of a Table or TableImpl. Not"
                f" '{with_}'."
            )

        if not with_.is_aligned_with(literal_column):
            raise ValueError(f"Result of eval_aligned isn't aligned with {with_}.")

    # Convert to sexpr so that the user can easily continue transforming
    # it symbolically.
    return symbolic_expressions.SymbolicExpression(literal_column)
