from __future__ import annotations

import pytest

from pydiverse.transform._internal.ops import Operator
from pydiverse.transform._internal.tree import dtypes
from pydiverse.transform._internal.tree.registry import (
    OperatorRegistry,
    Signature,
)


def assert_signature(s: Signature, args: list[dtypes.Dtype], return_type: dtypes.Dtype):
    assert len(s.params) == len(args)

    for actual, expected in zip(s.params, args, strict=True):
        assert expected.same_kind(actual)

    assert return_type.same_kind(s.return_type)


class TestOperatorSignature:
    def test_parse_simple(self):
        s = Signature.parse("int64, int64 -> int64")
        assert_signature(s, [dtypes.Int64(), dtypes.Int64()], dtypes.Int64())

        s = Signature.parse("bool->bool ")
        assert_signature(s, [dtypes.Bool()], dtypes.Bool())

        s = Signature.parse("-> int64")
        assert_signature(s, [], dtypes.Int64())

        with pytest.raises(ValueError):
            Signature.parse("int64, int64 -> ")
        with pytest.raises(ValueError):
            Signature.parse("int64, int64 -> int64, int64")

        with pytest.raises(ValueError):
            Signature.parse("o#r -> int64")
        with pytest.raises(ValueError):
            Signature.parse("int64 -> a#")

    def test_parse_template(self):
        s = Signature.parse("T, int64 -> int64")
        assert isinstance(s.params[0], dtypes.Template)

        s = Signature.parse("T -> T")
        assert isinstance(s.params[0], dtypes.Template)
        assert isinstance(s.return_type, dtypes.Template)

        with pytest.raises(ValueError):
            Signature.parse("T, T -> U")

    def test_parse_varargs(self):
        s = Signature.parse("int64, str... -> int64")
        assert not s.params[0].vararg
        assert s.params[1].vararg

        s = Signature.parse("int64... -> bool")
        assert s.params[0].vararg

        with pytest.raises(ValueError):
            Signature.parse("int64..., str -> int64")
        with pytest.raises(ValueError):
            Signature.parse("int64, str -> int64...")

        s0 = Signature.parse("int64, str    -> int64")
        s1 = Signature.parse("int64, str... -> int64")

        assert not s0.is_vararg
        assert s1.is_vararg

    def test_parse_const(self):
        s = Signature.parse("const int64 -> int64")
        assert s.params[0].const
        assert not s.return_type.const


def parse_dtypes(*strings):
    return [dtypes.dtype_from_string(s) for s in strings]


class TestOperatorRegistry:
    class Op1(Operator):
        name = "op1"
        ftype = None

    class Op2(Operator):
        name = "op2"
        ftype = None

    class Op3(Operator):
        name = "op3"
        ftype = None

    def test_simple(self):
        op1 = self.Op1()
        op2 = self.Op2()

        reg = OperatorRegistry("TestRegistry")
        reg.register_op(op1)
        reg.register_op(op2)

        reg.add_impl(op1, lambda: 1, "int64, int64 -> int64")
        reg.add_impl(op1, lambda: 2, "str, str -> str")

        reg.add_impl(op2, lambda: 10, "int64, int64 -> int64")
        reg.add_impl(op2, lambda: 20, "str, str -> str")

        assert reg.get_impl("op1", parse_dtypes("int64", "int64"))() == 1
        assert isinstance(
            reg.get_impl("op1", parse_dtypes("int64", "int64")).return_type,
            dtypes.Int64,
        )
        assert reg.get_impl("op2", parse_dtypes("int64", "int64"))() == 10

        assert reg.get_impl("op1", parse_dtypes("str", "str"))() == 2
        assert reg.get_impl("op2", parse_dtypes("str", "str"))() == 20

        with pytest.raises(TypeError):
            reg.get_impl("op1", parse_dtypes("int64", "str"))
        with pytest.raises(ValueError):
            reg.get_impl(
                "not_implemented",
                parse_dtypes("int64"),
            )

        reg.add_impl(op1, lambda: 100, "-> int64")
        assert reg.get_impl("op1", tuple())() == 100

    def test_template(self):
        reg = OperatorRegistry("TestRegistry")

        op1 = self.Op1()
        op2 = self.Op2()
        op3 = self.Op3()

        reg.register_op(op1)
        reg.register_op(op2)
        reg.register_op(op3)

        reg.add_impl(op1, lambda: 1, "T, T -> bool")
        reg.add_impl(op1, lambda: 2, "T, U -> U")

        with pytest.raises(ValueError, match="already defined"):
            reg.add_impl(op1, lambda: 3, "T, U -> U")

        assert reg.get_impl("op1", parse_dtypes("int64", "int64"))() == 1
        assert reg.get_impl("op1", parse_dtypes("int64", "str"))() == 2
        # int64 can be promoted to float; results in "float, float -> bool" signature
        assert reg.get_impl("op1", parse_dtypes("int64", "float64"))() == 1
        assert reg.get_impl("op1", parse_dtypes("float64", "int64"))() == 1

        # More template matching... Also check matching precedence
        reg.add_impl(op2, lambda: 1, "int64, int64, int64 -> int64")
        reg.add_impl(op2, lambda: 2, "int64, str, T -> int64")
        reg.add_impl(op2, lambda: 3, "int64, T, str -> int64")
        reg.add_impl(op2, lambda: 4, "int64, T, T -> int64")
        reg.add_impl(op2, lambda: 5, "T, T, T -> int64")
        reg.add_impl(op2, lambda: 6, "A, T, T -> int64")

        assert reg.get_impl("op2", parse_dtypes("int64", "int64", "int64"))() == 1
        assert reg.get_impl("op2", parse_dtypes("int64", "str", "str"))() == 2
        assert reg.get_impl("op2", parse_dtypes("int64", "int64", "str"))() == 3
        assert reg.get_impl("op2", parse_dtypes("int64", "bool", "bool"))() == 4
        assert reg.get_impl("op2", parse_dtypes("str", "str", "str"))() == 5
        assert reg.get_impl("op2", parse_dtypes("float64", "str", "str"))() == 6

        with pytest.raises(TypeError):
            reg.get_impl("op2", parse_dtypes("int64", "bool", "float64"))

        # Return type
        reg.add_impl(op3, lambda: 1, "T -> T")
        reg.add_impl(op3, lambda: 2, "int64, T, U -> T")
        reg.add_impl(op3, lambda: 3, "str, T, U -> U")

        with pytest.raises(ValueError, match="already defined."):
            reg.add_impl(op3, lambda: 4, "int64, T, U -> U")

        assert isinstance(
            reg.get_impl("op3", parse_dtypes("str")).return_type,
            dtypes.String,
        )
        assert isinstance(
            reg.get_impl("op3", parse_dtypes("int64")).return_type,
            dtypes.Int64,
        )
        assert isinstance(
            reg.get_impl("op3", parse_dtypes("int64", "int64", "float64")).return_type,
            dtypes.Int64,
        )
        assert isinstance(
            reg.get_impl("op3", parse_dtypes("str", "int64", "float64")).return_type,
            dtypes.Float64,
        )

    def test_vararg(self):
        reg = OperatorRegistry("TestRegistry")

        op1 = self.Op1()
        reg.register_op(op1)

        reg.add_impl(op1, lambda: 1, "int64... -> int64")
        reg.add_impl(op1, lambda: 2, "int64, int64... -> int64")
        reg.add_impl(op1, lambda: 3, "int64, T... -> T")

        assert (
            reg.get_impl(
                "op1",
                parse_dtypes(
                    "int64",
                ),
            )()
            == 1
        )
        assert reg.get_impl("op1", parse_dtypes("int64", "int64"))() == 2
        assert reg.get_impl("op1", parse_dtypes("int64", "int64", "int64"))() == 2
        assert reg.get_impl("op1", parse_dtypes("int64", "str", "str"))() == 3

        assert isinstance(
            reg.get_impl("op1", parse_dtypes("int64", "str", "str")).return_type,
            dtypes.String,
        )

    def test_variant(self):
        op1 = self.Op1()

        reg = OperatorRegistry("TestRegistry")
        reg.register_op(op1)

        with pytest.raises(ValueError):
            reg.add_impl(op1, lambda: 2, "-> int64", variant="VAR")

        reg.add_impl(op1, lambda: 1, "-> int64")
        reg.add_impl(op1, lambda: 2, "-> int64", variant="VAR")

        assert reg.get_impl("op1", tuple())() == 1
        assert reg.get_impl("op1", tuple()).get_variant("VAR")() == 2

        with pytest.raises(ValueError):
            reg.add_impl(op1, lambda: 2, "-> int64", variant="VAR")
