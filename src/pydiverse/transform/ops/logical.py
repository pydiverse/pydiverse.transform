from __future__ import annotations

from pydiverse.transform.core import dtypes
from pydiverse.transform.ops.core import Binary, ElementWise, Unary

__all__ = [
    "Equal",
    "NotEqual",
    "Less",
    "LessEqual",
    "Greater",
    "GreaterEqual",
    "And",
    "RAnd",
    "Or",
    "ROr",
    "Xor",
    "RXor",
    "Invert",
]


#### Comparison Operators ####


class Comparison(ElementWise, Binary):
    signatures = [
        "int, int -> bool",
        "int, float -> bool",
        "float, int -> bool",
        "float, float -> bool",
        "str, str -> bool",
        "bool, bool -> bool",
    ]

    def validate_signature(self, signature):
        assert isinstance(signature.rtype, dtypes.Bool)
        super().validate_signature(signature)


class Equal(Comparison):
    name = "__eq__"


class NotEqual(Comparison):
    name = "__ne__"


class Less(Comparison):
    name = "__lt__"


class LessEqual(Comparison):
    name = "__le__"


class Greater(Comparison):
    name = "__gt__"


class GreaterEqual(Comparison):
    name = "__ge__"


#### Boolean Operators ####


class BooleanBinary(ElementWise, Binary):
    signatures = [
        "bool, bool -> bool",
    ]

    def validate_signature(self, signature):
        assert len(signature.args) == 2

        for arg_dtype in signature.args:
            assert isinstance(arg_dtype, dtypes.Bool)
            assert not arg_dtype.vararg
            assert not arg_dtype.const

        assert isinstance(signature.rtype, dtypes.Bool)
        super().validate_signature(signature)


class And(BooleanBinary):
    name = "__and__"


class RAnd(BooleanBinary):
    name = "__rand__"


class Or(BooleanBinary):
    name = "__or__"


class ROr(BooleanBinary):
    name = "__ror__"


class Xor(BooleanBinary):
    name = "__xor__"


class RXor(BooleanBinary):
    name = "__rxor__"


class Invert(ElementWise, Unary):
    name = "__invert__"
    signatures = [
        "bool -> bool",
    ]
