from dataclasses import dataclass
from typing import Any, Generic, TYPE_CHECKING, TypeVar

from pdtransform.core.expressions import expressions

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from pdtransform.core.table_impl import AbstractTableImpl

# Basic container to store value and associated type metadata
@dataclass(slots = True)
class TypedValue:
    value: Any
    dtype: str

    def __iter__(self):
        return iter((self.value, self.dtype))


ImplT = TypeVar('ImplT', bound = 'AbstractTableImpl')
T = TypeVar('T')


class Translator(Generic[ImplT, T]):

    def __init__(self, backend: ImplT):
        self.backend = backend

    def translate(self, expr, **kwargs) -> T:
        """ Translate an expression recursively. """
        return bottom_up_replace(expr, lambda e: self._translate(e, **kwargs))

    def _translate(self, expr, **kwargs) -> T:
        """ Translate an expression non recursively. """
        raise NotImplementedError


def bottom_up_replace(expr, replace):
    # TODO: This is bad... At some point this should be refactored
    #       and replaced with something less hacky.

    def clone(expr):
        if isinstance(expr, expressions.FunctionCall):
            f = expressions.FunctionCall(
                expr.operator,
                *(clone(arg) for arg in expr.args),
                **{k: clone(v) for k, v in expr.kwargs.items()}
            )
            return replace(f)
        else:
            return replace(expr)

    return clone(expr)
