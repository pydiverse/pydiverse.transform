from .operator_registry import OperatorRegistry


class SymbolicExpression:
    """

    Attributes:

        __dtype: type

        def __add__
        def __sub__
        def   ...

        - Type arithmetic
        - Some way to traverse like sqlalchemy visitors.
        - A translator (like suiba) to turn this into a backend specific expression.

    Possible expressions:

        c1 + c2                ← Symbolic algebra
        ~(c1)                  ← For descending ordering
        c1.strip()             ← Strip String
        (c1 * c2).sum()        ← Aggregates
        c1.replace('a', 'b')   ← Operations with arguments
        sum(c1 * c2)           ← Alternative aggregates    (possible downside: Namespace pollution)

    x:

        Symbol
        Unary Operation
        Binary Operation
        Method

    """

    def __getattr__(self, item):
        return SymbolAttribute(item, self)

    def __getitem__(self, item):
        return FunctionCall('__getitem__', self, item)

    def _iter_children(self):
        # Body must contain a yield, else python doesn't know that this
        # is supposed to be a generator.
        yield from ()


class FunctionCall(SymbolicExpression):

    def __init__(self, operator: str, *args, **kwargs):
        self._operator = operator
        self._args = args
        self._kwargs = kwargs
        super().__init__()

    def __repr__(self):
        args = [
            repr(e) for e in self._args
        ] + [
            f'{k}={repr(v)}' for k, v in self._kwargs.items()
        ]
        return f'{self._operator}({", ".join(args)})'

    def _iter_children(self):
        yield from self._args


class SymbolAttribute:
    def __init__(self, name: str, on: SymbolicExpression):
        self.__name = name
        self.__on = on

    def __getattr__(self, item):
        return SymbolAttribute(self.__name + '.' + item, self.__on)

    def __call__(self, *args, **kwargs):
        return FunctionCall(self.__name, self.__on, *args, **kwargs)

    def __hash__(self):
        raise Exception(f"Nope... You probably didn't want to do this. Did you misspell the attribute name '{self.__name}' of '{self.__on}'? Maybe you forgot a leading underscore.")


# Add all supported dunder methods to `SymbolicExpression`.
# This has to be done, because Python doesn't call __getattr__ for
# dunder methods.
def create_operator(op):
    def impl(*args, **kwargs):
        return FunctionCall(op, *args, **kwargs)
    return impl
for dunder in OperatorRegistry.SUPPORTED_DUNDER:
    setattr(SymbolicExpression, dunder, create_operator(dunder))
del create_operator

# Utils

def iterate_over_expr(expr):
    """
    Iterate in depth-first preorder over the expression and yield all components.
    """

    yield expr
    if isinstance(expr, SymbolicExpression):
        for child in expr._iter_children():
            yield from iterate_over_expr(child)