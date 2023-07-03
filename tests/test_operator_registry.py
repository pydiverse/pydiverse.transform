from __future__ import annotations

import pytest

from pydiverse.transform.core.ops import Operator
from pydiverse.transform.core.ops.registry import (
    OperatorRegistry,
    OperatorSignature,
)


class TestOperatorSignature:
    def test_parse_simple(self):
        s = OperatorSignature.parse("int, int -> int")
        assert s.args == ("int", "int") and s.rtype == "int"

        s = OperatorSignature.parse("bool->bool ")
        assert s.args == ("bool",) and s.rtype == "bool"

        s = OperatorSignature.parse("-> int")
        assert s.args == tuple() and s.rtype == "int"

        with pytest.raises(ValueError):
            OperatorSignature.parse("int, int -> ")
        with pytest.raises(ValueError):
            OperatorSignature.parse("int, int -> int, int")

        with pytest.raises(ValueError):
            OperatorSignature.parse("o#r -> int")
        with pytest.raises(ValueError):
            OperatorSignature.parse("int -> a#")

    def test_parse_template(self):
        OperatorSignature.parse("T, int -> int")
        OperatorSignature.parse("T -> T")

        with pytest.raises(ValueError):
            OperatorSignature.parse("T, T -> U")

    def test_parse_varargs(self):
        OperatorSignature.parse("int, str... -> int")
        OperatorSignature.parse("int... -> bool")

        with pytest.raises(ValueError):
            OperatorSignature.parse("int..., str -> int")
        with pytest.raises(ValueError):
            OperatorSignature.parse("int, str -> int...")

        s0 = OperatorSignature.parse("int, str    -> int")
        s1 = OperatorSignature.parse("int, str... -> int")

        assert not s0.is_vararg
        assert s1.is_vararg


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

        assert reg.get_implementation("op1", ("int", "int"))() == 1
        assert reg.get_implementation("op1", ("int", "int")).rtype == "int"
        assert reg.get_implementation("op2", ("int", "int"))() == 10

        assert reg.get_implementation("op1", ("str", "str"))() == 2
        assert reg.get_implementation("op2", ("str", "str"))() == 20

        with pytest.raises(ValueError):
            reg.get_implementation("op1", ("int", "str"))
        with pytest.raises(ValueError):
            reg.get_implementation("not_implemented", ("int",))

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

        assert reg.get_implementation("op1", ("int", "int"))() == 1
        assert reg.get_implementation("op1", ("int", "float"))() == 2

        # More template matching... Also check matching precedence
        reg.add_implementation(op2, lambda: 1, "int, int, int -> int")
        reg.add_implementation(op2, lambda: 2, "int, T, T -> int")
        reg.add_implementation(op2, lambda: 3, "T, T, T -> int")
        reg.add_implementation(op2, lambda: 4, "A, T, T -> int")

        assert reg.get_implementation("op2", ("int", "int", "int"))() == 1
        assert reg.get_implementation("op2", ("int", "str", "str"))() == 2
        assert reg.get_implementation("op2", ("str", "str", "str"))() == 3
        assert reg.get_implementation("op2", ("float", "str", "str"))() == 4

        with pytest.raises(ValueError):
            reg.get_implementation("op2", ("int", "str", "float"))

        # Return type
        reg.add_implementation(op3, lambda: 1, "T -> T")
        reg.add_implementation(op3, lambda: 2, "int, T, U -> T")
        reg.add_implementation(op3, lambda: 3, "str, T, U -> U")

        with pytest.raises(ValueError, match="already defined."):
            reg.add_implementation(op3, lambda: 4, "int, T, U -> U")

        assert reg.get_implementation("op3", ("str",)).rtype == "str"
        assert reg.get_implementation("op3", ("int",)).rtype == "int"
        assert reg.get_implementation("op3", ("int", "int", "float")).rtype == "int"
        assert reg.get_implementation("op3", ("str", "int", "float")).rtype == "float"

    def test_vararg(self):
        reg = OperatorRegistry("TestRegistry")

        op1 = self.Op1()
        reg.register_op(op1)

        reg.add_implementation(op1, lambda: 1, "int... -> int")
        reg.add_implementation(op1, lambda: 2, "int, int... -> int")
        reg.add_implementation(op1, lambda: 3, "int, T... -> T")

        assert reg.get_implementation("op1", ("int",))() == 1
        assert reg.get_implementation("op1", ("int", "int"))() == 2
        assert reg.get_implementation("op1", ("int", "int", "int"))() == 2
        assert reg.get_implementation("op1", ("int", "str", "str"))() == 3
        assert reg.get_implementation("op1", ("int", "str", "str")).rtype == "str"

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
