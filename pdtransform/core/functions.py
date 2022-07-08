from .expressions import SymbolicExpression, FunctionCall


__all__ = [
    'count',
    'row_number',
]


def sym_f_call(name, *args, **kwargs) -> SymbolicExpression[FunctionCall]:
    return SymbolicExpression(FunctionCall(name, *args, **kwargs))


def count(expr: SymbolicExpression = None):
    if expr is None:
        return sym_f_call('count')
    else:
        return sym_f_call('count', expr)


def row_number():
    return sym_f_call('row_number')
