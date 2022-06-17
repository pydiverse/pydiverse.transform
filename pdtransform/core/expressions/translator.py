from collections import namedtuple
from pdtransform import core
from pdtransform.core.expressions import SymbolicExpression, FunctionCall


# Basic container to store value and associated type metadata
TypedValue = namedtuple('TypedValue', ['value', 'dtype'])

class Translator:

    def __init__(self, backend: 'core.AbstractTableImpl'):
         self.backend = backend

    def translate(self, expr):
        """ Translate an expression recursively. """
        return bottom_up_replace(expr, self._translate)

    def _translate(self, expr):
        """ Translate an expression non recursively. """
        raise NotImplementedError


def bottom_up_replace(expr: SymbolicExpression, _replace):
    # TODO: This is bad... At some point this should be refactored
    #       and replaced with something less hacky.

    replaced = dict()
    def replace(expr):
        if expr in replaced:
            return replaced[expr]
        v = _replace(expr)
        replaced[expr] = v
        return v

    def clone(expr):
        if isinstance(expr, FunctionCall):
            f = FunctionCall(
                expr._operator,
                *(clone(arg) for arg in expr._args),
                **{k: clone(v) for k, v in expr._kwargs.items()}
            )
            return replace(f)
        else:
            return replace(expr)

    return clone(expr)
