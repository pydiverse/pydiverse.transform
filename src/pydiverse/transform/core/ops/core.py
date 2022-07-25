from __future__ import annotations

import enum
import typing
from collections import ChainMap

from pydiverse.transform.core.ops import registry

__all__ = [
    "OPType",
    "Operator",
    "OperatorExtension",
    "Arity",
    "Nullary",
    "Unary",
    "Binary",
    "ElementWise",
    "Aggregate",
    "Window",
]


class OPType(enum.IntEnum):
    EWISE = 1
    AGGREGATE = 2
    WINDOW = 3


_OPERATOR_VALID = "_operator_valid_"
_OPERATOR_MISSING_ATTR = "_operator_missing_attr_"


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
    ftype: OPType = NotImplemented
    signatures: list[str] = None
    context_kwargs: set[str] = None

    def __new__(cls, *args, **kwargs):
        if not getattr(cls, _OPERATOR_VALID, False):
            missing = getattr(cls, _OPERATOR_MISSING_ATTR, None)
            raise NotImplementedError(
                "Can't initialize operator. Some class attributes are not implemented:"
                f" {missing}."
            )
        return super().__new__(cls, *args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        mro_dict = ChainMap(*(sup.__dict__ for sup in cls.__mro__))
        not_implemented = [k for k, v in mro_dict.items() if v is NotImplemented]
        setattr(cls, _OPERATOR_VALID, len(not_implemented) == 0)
        setattr(cls, _OPERATOR_MISSING_ATTR, not_implemented)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def validate_signature(self, signature: registry.OperatorSignature) -> bool:
        pass


class OperatorExtension:
    """
    An extension to an operator. Provides additional signatures.
    """

    operator: type[Operator] = NotImplemented
    signatures: list[str] = NotImplemented

    def __new__(cls, *args, **kwargs):
        if cls.operator == NotImplemented or cls.signatures == NotImplemented:
            raise NotImplementedError(
                "Can't initialize operator extension. "
                "Either the operator or signatures attribute is missing."
            )
        if not issubclass(cls.operator, Operator):
            raise TypeError
        return super().__new__(cls, *args, **kwargs)


# Arity


class Arity(Operator):
    """Base class for checking the number of arguments."""

    n_arguments: int = NotImplemented

    def validate_signature(self, signature):
        assert len(signature.args) == self.n_arguments
        super().validate_signature(signature)


class Nullary(Arity):
    n_arguments = 0


class Unary(Arity):
    n_arguments = 1


class Binary(Arity):
    n_arguments = 2


# Base operator types


class ElementWise(Operator):
    ftype = OPType.EWISE


class Aggregate(Operator):
    ftype = OPType.AGGREGATE


class Window(Operator):
    ftype = OPType.WINDOW
    context_kwargs = {
        "arrange",  # List[Column | LambdaColumn]
    }
