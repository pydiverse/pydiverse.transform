from __future__ import annotations

import pytest

from pydiverse.transform import C
from pydiverse.transform.core.expressions import FunctionCall, SymbolicExpression


def compare_sexpr(expr1, expr2):
    # Must compare using repr, because using == would result in another sexpr
    expr1 = expr1 if not isinstance(expr1, SymbolicExpression) else expr1._
    expr2 = expr2 if not isinstance(expr2, SymbolicExpression) else expr2._
    assert expr1 == expr2


class TestExpressions:
    def test_symbolic_expression(self):
        s1 = SymbolicExpression(1)
        s2 = SymbolicExpression(2)

        compare_sexpr(s1 + s1, FunctionCall("__add__", 1, 1))
        compare_sexpr(s1 + s2, FunctionCall("__add__", 1, 2))
        compare_sexpr(s1 + 10, FunctionCall("__add__", 1, 10))
        compare_sexpr(10 + s1, FunctionCall("__radd__", 1, 10))

        compare_sexpr(s1.argument(), FunctionCall("argument", 1))
        compare_sexpr(s1.str.argument(), FunctionCall("str.argument", 1))
        compare_sexpr(s1.argument(s2, 3), FunctionCall("argument", 1, 2, 3))

    def test_lambda_col(self):
        compare_sexpr(C.something, C["something"])
        compare_sexpr(C.something.chained(), C["something"].chained())

    def test_banned_methods(self):
        s1 = SymbolicExpression(1)

        with pytest.raises(TypeError):
            bool(s1)
        with pytest.raises(TypeError):
            _ = s1 in s1
        with pytest.raises(TypeError):
            _ = iter(s1)
