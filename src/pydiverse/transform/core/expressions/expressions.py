from __future__ import annotations

from . import symbolic_expressions


class FunctionCall:
    """
    AST node to represent a function / operator call.
    """

    def __init__(self, name: str, *args, **kwargs):
        # Unwrap all symbolic expressions in the input
        args = symbolic_expressions.unwrap_symbolic_expressions(args)
        kwargs = symbolic_expressions.unwrap_symbolic_expressions(kwargs)

        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        args = [repr(e) for e in self.args] + [
            f"{k}={repr(v)}" for k, v in self.kwargs.items()
        ]
        return f'{self.name}({", ".join(args)})'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, self.args, tuple(self.kwargs.items())))

    def iter_children(self):
        yield from self.args
