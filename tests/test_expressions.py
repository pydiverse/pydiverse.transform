from pdtransform import λ
from pdtransform.core.expressions.expression import SymbolicExpression, FunctionCall


def compare_sexpr(expr1, expr2):
    # Must compare using repr, because using == would result in another sexpr
    assert repr(expr1) == repr(expr2)


class TestExpressions:

    def test_symbolic_expression(self):
        s1 = SymbolicExpression()
        s2 = SymbolicExpression()

        compare_sexpr(s1 + s1, FunctionCall('__add__', s1, s1))
        compare_sexpr(s1 + s2, FunctionCall('__add__', s1, s2))
        compare_sexpr(s1 + 10, FunctionCall('__add__', s1, 10))
        compare_sexpr(10 + s1, FunctionCall('__radd__', s1, 10))

        compare_sexpr(s1.argument(), FunctionCall('argument', s1))
        compare_sexpr(s1.str.argument(), FunctionCall('str.argument', s1))

    def test_lambda_col(self):
        assert λ.something._name == λ['something']._name