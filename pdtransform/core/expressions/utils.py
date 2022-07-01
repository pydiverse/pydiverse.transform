from typing import Type, TYPE_CHECKING

from . import expressions
from pdtransform.core import column

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from pdtransform.core.table_impl import AbstractTableImpl


def iterate_over_expr(expr, expand_literal_col=False):
    """
    Iterate in depth-first preorder over the expression and yield all components.
    """

    yield expr

    if isinstance(expr, expressions.FunctionCall):
        for child in expr.iter_children():
            yield from iterate_over_expr(child, expand_literal_col=expand_literal_col)
        return

    if expand_literal_col and isinstance(expr, column.LiteralColumn):
        yield from iterate_over_expr(expr.expr, expand_literal_col=expand_literal_col)
        return


def determine_expr_backend(expr) -> Type['AbstractTableImpl'] | None:
    """Returns the backend used in an expression.

    Iterates over an expression and extracts the underlying backend type used.
    If no backend can be determined (because the expression doesn't contain a
    column), None is returned instead. If different backends are being used,
    throws an exception.
    """

    backends = set()
    for atom in iterate_over_expr(expr):
        if isinstance(atom, column.Column):
            backends.add(type(atom.table))
        if isinstance(atom, column.LiteralColumn):
            backends.add(atom.backend)

    if len(backends) == 1:
        return backends.pop()
    if len(backends) >= 2:
        raise ValueError(f"Expression contains different backends "
                         f"(found: {[backend.__name__ for backend in backends]}).")
    return None
