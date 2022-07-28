from __future__ import annotations

import typing
from dataclasses import dataclass

from pydiverse.transform.core import column
from pydiverse.transform.core.expressions import expressions

__all__ = (
    "traverse",
    "sign_peeler",
    "OrderingDescriptor",
    "translate_ordering",
)

T = typing.TypeVar("T")


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


def sign_peeler(expr):
    """Remove unary - and + prefix and return the sign
    :return: `True` for `+` and `False` for `-`
    """
    num_neg = 0
    while isinstance(expr, expressions.FunctionCall):
        if expr.name == "__neg__":
            num_neg += 1
            expr = expr.args[0]
        elif expr.name == "__pos__":
            expr = expr.args[0]
        else:
            break
    return expr, num_neg % 2 == 0


####


@dataclass
class OrderingDescriptor:
    __slots__ = ("order", "asc", "nulls_first")

    order: typing.Any
    asc: bool
    nulls_first: bool


def translate_ordering(tbl, order_list) -> list[OrderingDescriptor]:
    ordering = []
    for arg in order_list:
        col, ascending = sign_peeler(arg)
        if not isinstance(col, (column.Column, column.LambdaColumn)):
            raise ValueError(
                "Arguments to arrange must be of type 'Column' or 'LambdaColumn' and"
                f" not '{type(col)}'."
            )
        col = tbl.resolve_lambda_cols(col)
        ordering.append(OrderingDescriptor(col, ascending, False))

    return ordering
