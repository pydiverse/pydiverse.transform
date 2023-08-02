from __future__ import annotations

from pydiverse.transform.core import dtypes
from pydiverse.transform.ops.core import Binary, ElementWise, Operator, Unary

__all__ = [
    "Equal",
    "NotEqual",
    "Less",
    "LessEqual",
    "Greater",
    "GreaterEqual",
    "IsIn",
    "And",
    "RAnd",
    "Or",
    "ROr",
    "Xor",
    "RXor",
    "Invert",
]


class Logical(Operator):
    # Operator that returns a "REAL" boolean.
    # This is mostly relevant for mssql
    pass


#### Comparison Operators ####


class Comparison(ElementWise, Binary, Logical):
    signatures = [
        "int, int -> bool",
        "float, float -> bool",
        "str, str -> bool",
        "bool, bool -> bool",
        "datetime, datetime -> bool",
    ]

    def validate_signature(self, signature):
        assert isinstance(signature.rtype, dtypes.Bool)
        super().validate_signature(signature)


class Equal(Comparison):
    name = "__eq__"
    signatures = Comparison.signatures + [
        "T, const none -> bool",
    ]


class NotEqual(Comparison):
    name = "__ne__"
    signatures = Comparison.signatures + [
        "T, const none -> bool",
    ]


class Less(Comparison):
    name = "__lt__"


class LessEqual(Comparison):
    name = "__le__"


class Greater(Comparison):
    name = "__gt__"


class GreaterEqual(Comparison):
    name = "__ge__"


class IsIn(ElementWise, Logical):
    name = "isin"
    signatures = [
        # TODO: A signature like "T, const list[const T] -> bool" would be better
        "T, const T... -> bool",
    ]


#### Boolean Operators ####


class BooleanBinary(ElementWise, Binary, Logical):
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


class Invert(ElementWise, Unary, Logical):
    name = "__invert__"
    signatures = [
        "bool -> bool",
    ]
