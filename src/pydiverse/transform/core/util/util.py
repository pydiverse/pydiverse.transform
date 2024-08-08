from __future__ import annotations

import typing
from dataclasses import dataclass

from pydiverse.transform._typing import T
from pydiverse.transform.core.expressions import FunctionCall

__all__ = (
    "traverse",
    "sign_peeler",
    "OrderingDescriptor",
    "translate_ordering",
)


def traverse(obj: T, callback: typing.Callable) -> T:
    if isinstance(obj, list):
        return [traverse(elem, callback) for elem in obj]
    if isinstance(obj, dict):
        return {k: traverse(v, callback) for k, v in obj.items()}
    if isinstance(obj, tuple):
        if type(obj) is not tuple:
            # Named tuples cause problems
            raise Exception
        return tuple(traverse(elem, callback) for elem in obj)

    return callback(obj)


def peel_markers(expr, markers):
    found_markers = []
    while isinstance(expr, FunctionCall):
        if expr.name in markers:
            found_markers.append(expr.name)
            assert len(expr.args) == 1
            expr = expr.args[0]
        else:
            break
    return expr, found_markers


def sign_peeler(expr):
    """
    Remove unary - and + prefix and return the sign
    :return: `True` for `+` and `False` for `-`
    """

    expr, markers = peel_markers(expr, {"__neg__", "__pos__"})
    num_neg = markers.count("__neg__")
    return expr, num_neg % 2 == 0


def ordering_peeler(expr):
    expr, markers = peel_markers(
        expr, {"__neg__", "__pos__", "nulls_first", "nulls_last"}
    )

    ascending = markers.count("__neg__") % 2 == 0
    nulls_first = False
    for marker in markers:
        if marker == "nulls_first":
            nulls_first = True
            break
        if marker == "nulls_last":
            break

    return expr, ascending, nulls_first


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
        col, ascending, nulls_first = ordering_peeler(arg)
        col = tbl.resolve_lambda_cols(col)
        ordering.append(OrderingDescriptor(col, ascending, nulls_first))

    return ordering
