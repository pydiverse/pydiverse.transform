import typing
from dataclasses import dataclass

from pdtransform.core import column
from pdtransform.core.expressions import expressions

__all__ = (
    'traverse',
    'OrderingDescriptor',
    'translate_ordering',
)

T = typing.TypeVar('T')


def traverse(obj: T, callback: typing.Callable) -> T:
    if isinstance(obj, list):
        return [traverse(elem, callback) for elem in obj]
    if isinstance(obj, dict):
        return {k: traverse(v, callback) for k, v in obj.items()}
    if isinstance(obj, tuple):
        if type(obj) != tuple:
            # Named tuples cause problems
            raise Exception
        return tuple(traverse(elem, callback) for elem in obj)

    return callback(obj)


####


@dataclass(slots = True)
class OrderingDescriptor:
    order: typing.Any
    asc: bool
    nulls_first: bool


def translate_ordering(tbl, order_list) -> list[OrderingDescriptor]:
    def ordering_peeler(expr):
        num_neg = 0
        while isinstance(expr, expressions.FunctionCall) and expr.name == '__neg__':
            num_neg += 1
            expr = expr.args[0]
        return expr, num_neg % 2 == 0

    ordering = []
    for arg in order_list:
        col, ascending = ordering_peeler(arg)
        if not isinstance(col, (column.Column, column.LambdaColumn)):
            raise ValueError(f"Arguments to arrange must be of type 'Column' or 'LambdaColumn' and not '{type(col)}'.")
        col = tbl.resolve_lambda_cols(col)
        ordering.append(OrderingDescriptor(col, ascending, False))

    return ordering
