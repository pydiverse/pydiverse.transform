from dataclasses import dataclass
from typing import Any, Generic, TYPE_CHECKING, TypeVar

from pdtransform.core.expressions import FunctionCall

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


class Translator(Generic[ImplT]):

    def __init__(self, backend: ImplT):
        self.backend = backend

    def translate(self, expr):
        """ Translate an expression recursively. """
        return bottom_up_replace(expr, self._translate)

    def _translate(self, expr):
        """ Translate an expression non recursively. """
        raise NotImplementedError


def bottom_up_replace(expr, replace):
    # TODO: This is bad... At some point this should be refactored
    #       and replaced with something less hacky.

    def clone(expr):
        if isinstance(expr, FunctionCall):
            f = FunctionCall(
                expr.operator,
                *(clone(arg) for arg in expr.args),
                **{k: clone(v) for k, v in expr.kwargs.items()}
            )
            return replace(f)
        else:
            return replace(expr)

    return clone(expr)
