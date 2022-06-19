from typing import Any, Generic, TypeVar

from . import expressions
from .operator_registry import OperatorRegistry

T = TypeVar('T')
class SymbolicExpression(Generic[T]):
    """
    Base class to represent a symbolic expression. It can be manipulated using
    standard python operators (for example you can add them) or by calling
    attributes of it.

    To get the non-symbolic version of this expression you use the
    underscore `_` attribute.
    """

    __slots__ = ('_', )

    def __init__(self, underlying: T):
        self._ = underlying

    def __getattr__(self, item):
        return SymbolAttribute(item, self)

    def __getitem__(self, item):
        return SymbolicExpression(expressions.FunctionCall('__getitem__', self, item))

    def __repr__(self):
        return f'<Sym: {self._}>'


class SymbolAttribute:
    def __init__(self, name: str, on: SymbolicExpression):
        self.__name = name
        self.__on = on

    def __getattr__(self, item):
        return SymbolAttribute(self.__name + '.' + item, self.__on)

    def __call__(self, *args, **kwargs):
        return expressions.FunctionCall(self.__name, self.__on, *args, **kwargs)

    def __hash__(self):
        raise Exception(f"Nope... You probably didn't want to do this. Did you misspell the attribute name '{self.__name}' of '{self.__on}'? Maybe you forgot a leading underscore.")


def unwrap_symbolic_expressions(arg: Any = None):
    """
    Replaces all symbolic expressions in the input with their underlying value.
    """

    # Potential alternative approach: Iterate over object like deepcopy does

    if isinstance(arg, list):
        return [unwrap_symbolic_expressions(x) for x in arg]
    if isinstance(arg, tuple):
        if type(arg) != tuple:
            raise Exception  # This is the case with named tuples for example
        return tuple(unwrap_symbolic_expressions(x) for x in arg)
    if isinstance(arg, dict):
        return { k: unwrap_symbolic_expressions(v) for k, v in arg.items() }
    return arg._ if isinstance(arg, SymbolicExpression) else arg



# Add all supported dunder methods to `SymbolicExpression`.
# This has to be done, because Python doesn't call __getattr__ for
# dunder methods.
def create_operator(op):
    def impl(*args, **kwargs):
        return SymbolicExpression(expressions.FunctionCall(op, *args, **kwargs))
    return impl
for dunder in OperatorRegistry.SUPPORTED_DUNDER:
    setattr(SymbolicExpression, dunder, create_operator(dunder))
del create_operator
