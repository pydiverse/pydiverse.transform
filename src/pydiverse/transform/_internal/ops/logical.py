from __future__ import annotations

from pydiverse.transform._internal.ops.core import Binary, ElementWise, Operator, Unary
from pydiverse.transform._internal.ops.registry import Signature
from pydiverse.transform._internal.tree import dtypes

__all__ = [
    "Equal",
    "NotEqual",
    "Less",
    "LessEqual",
    "Greater",
    "GreaterEqual",
    "IsNull",
    "IsNotNull",
    "FillNull",
    "IsIn",
    "And",
    "Or",
    "Xor",
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
        "datetime, date -> bool",
        "date, datetime -> bool",
        "date, date -> bool",
    ]

    def validate_signature(self, signature: Signature):
        assert isinstance(signature.return_type, dtypes.Bool)
        super().validate_signature(signature)


class Equal(Comparison):
    name = "__eq__"
    signatures = Comparison.signatures


class NotEqual(Comparison):
    name = "__ne__"
    signatures = Comparison.signatures


class IsNull(ElementWise, Unary, Logical):
    name = "is_null"
    signatures = ["T -> bool"]


class IsNotNull(ElementWise, Unary, Logical):
    name = "is_not_null"
    signatures = ["T -> bool"]


class FillNull(ElementWise, Binary):
    name = "fill_null"
    signatures = ["T, T -> T"]


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
    signatures = ["T, T... -> bool"]
    arg_names = ["self", "args"]


class BooleanBinary(ElementWise, Binary, Logical):
    signatures = [
        "bool, bool -> bool",
    ]

    def validate_signature(self, signature: Signature):
        assert len(signature.params) == 2

        for arg_dtype in signature.params:
            assert isinstance(arg_dtype, dtypes.Bool)
            assert not arg_dtype.vararg
            assert not arg_dtype.const

        assert isinstance(signature.return_type, dtypes.Bool)
        super().validate_signature(signature)


class And(BooleanBinary):
    name = "__and__"


class Or(BooleanBinary):
    name = "__or__"


class Xor(BooleanBinary):
    name = "__xor__"


class Invert(ElementWise, Unary, Logical):
    name = "__invert__"
    signatures = [
        "bool -> bool",
    ]
