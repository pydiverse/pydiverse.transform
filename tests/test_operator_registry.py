from __future__ import annotations

import pytest

from pydiverse.transform.core import dtypes
from pydiverse.transform.core.registry import (
    OperatorRegistry,
    OperatorSignature,
)
from pydiverse.transform.ops import Operator


def assert_signature(
    s: OperatorSignature, args: list[dtypes.DType], rtype: dtypes.DType
):
    assert len(s.args) == len(args)

    for actual, expected in zip(s.args, args):
        assert expected.same_kind(actual)

    assert rtype.same_kind(s.rtype)


class TestOperatorSignature:
    def test_parse_simple(self):
        s = OperatorSignature.parse("int, int -> int")
        assert_signature(s, [dtypes.Int(), dtypes.Int()], dtypes.Int())

        s = OperatorSignature.parse("bool->bool ")
        assert_signature(s, [dtypes.Bool()], dtypes.Bool())

        s = OperatorSignature.parse("-> int")
        assert_signature(s, [], dtypes.Int())

        with pytest.raises(ValueError):
            OperatorSignature.parse("int, int -> ")
        with pytest.raises(ValueError):
            OperatorSignature.parse("int, int -> int, int")

        with pytest.raises(ValueError):
            OperatorSignature.parse("o#r -> int")
        with pytest.raises(ValueError):
            OperatorSignature.parse("int -> a#")

    def test_parse_template(self):
        s = OperatorSignature.parse("T, int -> int")
        assert isinstance(s.args[0], dtypes.Template)

        s = OperatorSignature.parse("T -> T")
        assert isinstance(s.args[0], dtypes.Template)
        assert isinstance(s.rtype, dtypes.Template)

        with pytest.raises(ValueError):
            OperatorSignature.parse("T, T -> U")

    def test_parse_varargs(self):
        s = OperatorSignature.parse("int, str... -> int")
        assert not s.args[0].vararg
        assert s.args[1].vararg

        s = OperatorSignature.parse("int... -> bool")
        assert s.args[0].vararg

        with pytest.raises(ValueError):
            OperatorSignature.parse("int..., str -> int")
        with pytest.raises(ValueError):
            OperatorSignature.parse("int, str -> int...")

        s0 = OperatorSignature.parse("int, str    -> int")
        s1 = OperatorSignature.parse("int, str... -> int")

        assert not s0.is_vararg
        assert s1.is_vararg

    def test_parse_const(self):
        s = OperatorSignature.parse("const int -> int")
        assert s.args[0].const
        assert not s.rtype.const


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

        reg.add_implementation(op1, lambda: 1, "int, int -> int")
        reg.add_implementation(op1, lambda: 2, "str, str -> str")

        reg.add_implementation(op2, lambda: 10, "int, int -> int")
        reg.add_implementation(op2, lambda: 20, "str, str -> str")

        assert reg.get_implementation("op1", parse_dtypes("int", "int"))() == 1
        assert isinstance(
            reg.get_implementation("op1", parse_dtypes("int", "int")).rtype,
            dtypes.Int,
        )
        assert reg.get_implementation("op2", parse_dtypes("int", "int"))() == 10

        assert reg.get_implementation("op1", parse_dtypes("str", "str"))() == 2
        assert reg.get_implementation("op2", parse_dtypes("str", "str"))() == 20

        with pytest.raises(ValueError):
            reg.get_implementation("op1", parse_dtypes("int", "str"))
        with pytest.raises(ValueError):
            reg.get_implementation(
                "not_implemented",
                parse_dtypes(
                    "int",
                ),
            )

        reg.add_implementation(op1, lambda: 100, "-> int")
        assert reg.get_implementation("op1", tuple())() == 100

    def test_template(self):
        reg = OperatorRegistry("TestRegistry")

        op1 = self.Op1()
        op2 = self.Op2()
        op3 = self.Op3()

        reg.register_op(op1)
        reg.register_op(op2)
        reg.register_op(op3)

        reg.add_implementation(op1, lambda: 1, "T, T -> bool")
        reg.add_implementation(op1, lambda: 2, "T, U -> U")

        with pytest.raises(ValueError, match="already defined"):
            reg.add_implementation(op1, lambda: 3, "T, U -> U")

        assert reg.get_implementation("op1", parse_dtypes("int", "int"))() == 1
        assert reg.get_implementation("op1", parse_dtypes("int", "str"))() == 2
        # int can be promoted to float; results in "float, float -> bool" signature
        assert reg.get_implementation("op1", parse_dtypes("int", "float"))() == 1
        assert reg.get_implementation("op1", parse_dtypes("float", "int"))() == 1

        # More template matching... Also check matching precedence
        reg.add_implementation(op2, lambda: 1, "int, int, int -> int")
        reg.add_implementation(op2, lambda: 2, "int, str, T -> int")
        reg.add_implementation(op2, lambda: 3, "int, T, str -> int")
        reg.add_implementation(op2, lambda: 4, "int, T, T -> int")
        reg.add_implementation(op2, lambda: 5, "T, T, T -> int")
        reg.add_implementation(op2, lambda: 6, "A, T, T -> int")

        assert reg.get_implementation("op2", parse_dtypes("int", "int", "int"))() == 1
        assert reg.get_implementation("op2", parse_dtypes("int", "str", "str"))() == 2
        assert reg.get_implementation("op2", parse_dtypes("int", "int", "str"))() == 3
        assert reg.get_implementation("op2", parse_dtypes("int", "bool", "bool"))() == 4
        assert reg.get_implementation("op2", parse_dtypes("str", "str", "str"))() == 5
        assert reg.get_implementation("op2", parse_dtypes("float", "str", "str"))() == 6

        with pytest.raises(ValueError):
            reg.get_implementation("op2", parse_dtypes("int", "bool", "float"))

        # Return type
        reg.add_implementation(op3, lambda: 1, "T -> T")
        reg.add_implementation(op3, lambda: 2, "int, T, U -> T")
        reg.add_implementation(op3, lambda: 3, "str, T, U -> U")

        with pytest.raises(ValueError, match="already defined."):
            reg.add_implementation(op3, lambda: 4, "int, T, U -> U")

        assert isinstance(
            reg.get_implementation("op3", parse_dtypes("str")).rtype,
            dtypes.String,
        )
        assert isinstance(
            reg.get_implementation("op3", parse_dtypes("int")).rtype,
            dtypes.Int,
        )
        assert isinstance(
            reg.get_implementation("op3", parse_dtypes("int", "int", "float")).rtype,
            dtypes.Int,
        )
        assert isinstance(
            reg.get_implementation("op3", parse_dtypes("str", "int", "float")).rtype,
            dtypes.Float,
        )

    def test_vararg(self):
        reg = OperatorRegistry("TestRegistry")

        op1 = self.Op1()
        reg.register_op(op1)

        reg.add_implementation(op1, lambda: 1, "int... -> int")
        reg.add_implementation(op1, lambda: 2, "int, int... -> int")
        reg.add_implementation(op1, lambda: 3, "int, T... -> T")

        assert (
            reg.get_implementation(
                "op1",
                parse_dtypes(
                    "int",
                ),
            )()
            == 1
        )
        assert reg.get_implementation("op1", parse_dtypes("int", "int"))() == 2
        assert reg.get_implementation("op1", parse_dtypes("int", "int", "int"))() == 2
        assert reg.get_implementation("op1", parse_dtypes("int", "str", "str"))() == 3

        assert isinstance(
            reg.get_implementation("op1", parse_dtypes("int", "str", "str")).rtype,
            dtypes.String,
        )

    def test_variant(self):
        op1 = self.Op1()

        reg = OperatorRegistry("TestRegistry")
        reg.register_op(op1)

        with pytest.raises(ValueError):
            reg.add_implementation(op1, lambda: 2, "-> int", variant="VAR")

        reg.add_implementation(op1, lambda: 1, "-> int")
        reg.add_implementation(op1, lambda: 2, "-> int", variant="VAR")

        assert reg.get_implementation("op1", tuple())() == 1
        assert reg.get_implementation("op1", tuple()).get_variant("VAR")() == 2

        with pytest.raises(ValueError):
            reg.add_implementation(op1, lambda: 2, "-> int", variant="VAR")
