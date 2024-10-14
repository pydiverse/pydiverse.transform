from __future__ import annotations

import enum

from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.dtypes import Dtype

__all__ = [
    "Ftype",
    "Operator",
    "ElementWise",
    "Aggregate",
    "Window",
]


class Ftype(enum.IntEnum):
    EWISE = 1
    AGGREGATE = 2
    WINDOW = 3


class Operator:
    """
    Base class to define an operator. All class level attributes that have the
    value `NotImplemented` must be set or else the operator can't be initialized.

    name:
        The name of the operator. The main mechanism to access an operator is
        using `(s-expression).operator_name(args)`.

    ftype:
        The operator type (function type). Can either be an element wise,
        aggregate or window function.

    signatures:
        A list of default signatures for this operator.

    context_kwargs:
        A set of keyword argument names. Can later be used by the translator
        to handle these keyword arguments specially.

    """

    name: str = NotImplemented
    ftype: Ftype | None = NotImplemented
    signatures: list[Signature] = []
    context_kwargs: list[str] = None

    is_expression_method: bool = True
    has_rversion: bool = False
    defaults: list = None
    arg_names: list[str] = None

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def get_return_type(*args: type[Dtype]) -> type[Dtype]: ...


class ElementWise(Operator):
    ftype = Ftype.EWISE


class Aggregate(Operator):
    ftype = Ftype.AGGREGATE
    context_kwargs = ["partition_by", "filter"]


class Window(Operator):
    ftype = Ftype.WINDOW
    context_kwargs = ["partition_by", "arrange", "filter"]


class Unary(Operator):
    arg_names = ["self"]


class Binary(Operator):
    arg_names = ["self", "rhs"]
