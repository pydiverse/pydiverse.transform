import operator

import pytest

from pdtransform.core.expressions.operator_registry import OperatorRegistry, OperatorSignature, TypedOperatorImpl


class TestOperatorSignature:

    def test_parse_simple(self):
        s = OperatorSignature.parse('int, int -> int')
        assert s.args == ('int', 'int') and s.rtype == 'int'

        s = OperatorSignature.parse('bool->bool ')
        assert s.args == ('bool',) and s.rtype == 'bool'

        with pytest.raises(ValueError):
            OperatorSignature.parse('-> int')
        with pytest.raises(ValueError):
            OperatorSignature.parse('int, int -> ')
        with pytest.raises(ValueError):
            OperatorSignature.parse('int, int -> int, int')

        with pytest.raises(ValueError):
            OperatorSignature.parse('o#r -> int')
        with pytest.raises(ValueError):
            OperatorSignature.parse('int -> a#')

    def test_parse_template(self):
        OperatorSignature.parse('T, int -> int')
        OperatorSignature.parse('T -> T')

        with pytest.raises(ValueError):
            OperatorSignature.parse('T, T -> U')

    def test_parse_varargs(self):
        OperatorSignature.parse('int, str... -> int')
        OperatorSignature.parse('int... -> bool')

        with pytest.raises(ValueError):
            OperatorSignature.parse('int..., str -> int')
        with pytest.raises(ValueError):
            OperatorSignature.parse('int, str -> int...')

    def test_parse_f_type(self):
        assert OperatorSignature.parse('int -> int').f_type == 's'
        assert OperatorSignature.parse('int |> int').f_type == 'a'

        with pytest.raises(ValueError):
            OperatorSignature.parse('int -|> int')
        with pytest.raises(ValueError):
            OperatorSignature.parse('int ->> int')
        with pytest.raises(ValueError):
            OperatorSignature.parse('int |>|> int')


class TestOperatorRegistry:

    def test_simple(self):
        reg = OperatorRegistry('TestRegistry')
        reg.register_op('__add__')
        reg.register_op('__sub__')

        reg.add_implementation(operator.add, '__add__', 'int, int -> int')
        reg.add_implementation(NotImplemented, '__add__', 'str, str -> str')

        reg.add_implementation(operator.sub, '__sub__', 'int, int -> int')
        reg.add_implementation(NotImplemented, '__sub__', 'str, str -> str')

        assert reg.get_implementation('__add__', ('int', 'int')).func == operator.add
        assert reg.get_implementation('__add__', ('int', 'int')).rtype == 'int'

        assert reg.get_implementation('__sub__', ('int', 'int')).func == operator.sub

        assert reg.get_implementation('__add__', ('str', 'str')).func == NotImplemented
        assert reg.get_implementation('__sub__', ('str', 'str')).func == NotImplemented

        with pytest.raises(ValueError):
            reg.get_implementation('not_registered', ('int'))

    def test_template(self):
        reg = OperatorRegistry('TestRegistry')

        reg.add_implementation(1, 'equal', 'T, T -> bool')
        reg.add_implementation(2, 'equal', 'T, U -> U')
        assert reg.get_implementation('equal', ('int', 'int')).func == 1
        assert reg.get_implementation('equal', ('int', 'float')).func == 2

        # More template matching... Also check matching precedence
        reg.add_implementation(1, 'x', 'int, int, int -> int')
        reg.add_implementation(2, 'x', 'int, T, T -> int')
        reg.add_implementation(3, 'x', 'T, T, T -> int')
        reg.add_implementation(4, 'x', 'A, T, T -> int')

        assert reg.get_implementation('x', ('int', 'int', 'int')).func == 1
        assert reg.get_implementation('x', ('int', 'str', 'str')).func == 2
        assert reg.get_implementation('x', ('str', 'str', 'str')).func == 3
        assert reg.get_implementation('x', ('float', 'str', 'str')).func == 4

        with pytest.raises(ValueError):
            reg.get_implementation('x', ('int', 'str', 'float'))

        # Return type
        reg.add_implementation(1, 'y', 'T -> T')
        reg.add_implementation(2, 'y', 'int, T, U -> T')
        reg.add_implementation(3, 'y', 'str, T, U -> U')

        with pytest.raises(ValueError, match = 'already defined.'):
            reg.add_implementation(4, 'y', 'int, T, U -> U')

        assert reg.get_implementation('y', ('str', )) == TypedOperatorImpl('y', 1, 'str', 's')
        assert reg.get_implementation('y', ('int', 'int', 'float')) == TypedOperatorImpl('y', 2, 'int', 's')
        assert reg.get_implementation('y', ('str', 'int', 'float')) == TypedOperatorImpl('y', 3, 'float', 's')

    def test_vararg(self):
        reg = OperatorRegistry('TestRegistry')
        reg.add_implementation(1, 'x', 'int... -> int')
        reg.add_implementation(2, 'x', 'int, int... -> int')
        reg.add_implementation(3, 'x', 'int, T... -> T')

        assert reg.get_implementation('x', ('int',)).func == 1
        assert reg.get_implementation('x', ('int', 'int')).func == 2
        assert reg.get_implementation('x', ('int', 'int', 'int')).func == 2
        assert reg.get_implementation('x', ('int', 'str', 'str')) == TypedOperatorImpl('x', 3, 'str', 's')

    def test_f_type(self):
        reg = OperatorRegistry('TestRegistry')
        reg.add_implementation(1, 'x', 'int -> int')

        with pytest.raises(ValueError):
            reg.add_implementation(1, 'x', 'int |> int')
