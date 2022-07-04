import pytest

from pdtransform import λ
from pdtransform.core import Table, AbstractTableImpl, Column
from pdtransform.core.dispatchers import inverse_partial, verb, col_to_table, wrap_tables, unwrap_tables
from pdtransform.core.expressions.translator import TypedValue
from pdtransform.core.util import bidict, ordered_set
from pdtransform.core.verbs import collect, select, mutate, join, filter, arrange


@pytest.fixture
def tbl1():
    return Table(MockTableImpl('mock1', ['col1', 'col2']))


@pytest.fixture
def tbl2():
    return Table(MockTableImpl('mock2', ['col1', 'col2', 'col3']))


class TestTable:

    def test_getattr(self, tbl1):
        assert tbl1.col1._.name == 'col1'
        assert tbl1.col2._.table == tbl1._impl

        with pytest.raises(AttributeError, match = 'colXXX'):
            _ = tbl1.colXXX

    def test_getitem(self, tbl1):
        assert tbl1.col1._ == tbl1['col1']._
        assert tbl1.col2._ == tbl1['col2']._

        assert tbl1.col2._ == tbl1[tbl1.col2]._
        assert tbl1.col2._ == tbl1[λ.col2]._

    def test_setitem(self, tbl1):
        tbl1['col1'] = 1
        tbl1[tbl1.col1] = 1
        tbl1[λ.col1] = 1

    def test_iter(self, tbl1, tbl2):
        assert len(list(tbl1)) == len(list(tbl1._impl.selected_cols()))
        assert len(list(tbl2)) == len(list(tbl2._impl.selected_cols()))

        assert repr(list(tbl1)) == repr([tbl1.col1, tbl1.col2])
        assert repr(list(tbl2)) == repr([tbl2.col1, tbl2.col2, tbl2.col3])

        assert repr(list(tbl2 >> select(tbl2.col2))) == repr([tbl2.col2])

        joined = tbl1 >> join(tbl2 >> select(tbl2.col3), tbl1.col1 == tbl2.col2, 'left')
        assert repr(list(joined)) == repr([tbl1.col1, tbl1.col2, joined.col3_mock2])

    def test_dir(self, tbl1):
        assert dir(tbl1) == ['col1', 'col2']
        assert dir(tbl1 >> mutate(x = tbl1.col1)) == ['col1', 'col2', 'x']

    def test_contains(self, tbl1, tbl2):
        assert tbl1.col1 in tbl1
        assert tbl1.col2 in tbl1

        assert tbl1.col1 not in tbl2
        assert tbl1.col2 not in tbl2

        assert λ.col1 in tbl1
        assert λ.col2 in tbl1
        assert λ.col3 not in tbl1

        assert λ.col1 in tbl2
        assert λ.col2 in tbl2
        assert λ.col3 in tbl2
        assert λ.col4 not in tbl2

        assert all(col in tbl1 for col in tbl1)
        assert all(col in tbl2 for col in tbl2)


class TestDispatchers:

    def test_inverse_partial(self):
        def x(a, b, c):
            return (a, b, c)
        assert inverse_partial(x, 1, 2)(0) == (0, 1, 2)
        assert inverse_partial(x, 1, c = 2)(0, c = 3) == (0, 1, 3)

    def test_pipeable(self):
        @verb
        def add(v1, v2):
            return v1 + v2

        @verb
        def subtract(v1, v2):
            return v1 - v2

        assert 1 >> add(2) == 3

        add_10 = add(5) >> add(5)
        assert 5 >> add_10 == 15

        assert 5 >> subtract(3) == 2
        assert 5 >> add_10 >> subtract(5) == 10

    def test_col_to_table(self, tbl1):
        assert col_to_table(15) == 15
        assert col_to_table(tbl1) == tbl1

        c1_tbl = col_to_table(tbl1.col1._)
        assert isinstance(c1_tbl, AbstractTableImpl)
        assert c1_tbl.available_cols == { tbl1.col1._.uuid }
        assert list(c1_tbl.named_cols.fwd) == ['col1']

    def test_unwrap_tables(self):
        impl_1 = MockTableImpl('impl_1', dict())
        impl_2 = MockTableImpl('impl_2', dict())
        tbl_1 = Table(impl_1)
        tbl_2 = Table(impl_2)

        assert unwrap_tables( 15 ) == 15
        assert unwrap_tables( impl_1 ) == impl_1
        assert unwrap_tables( tbl_1 ) == impl_1

        assert unwrap_tables( [tbl_1] ) == [impl_1]
        assert unwrap_tables( [[tbl_1], tbl_2] ) == [[impl_1], impl_2]

        assert unwrap_tables( (tbl_1, tbl_2) ) == (impl_1, impl_2)
        assert unwrap_tables( (tbl_1, (tbl_2, 15)) ) == (impl_1, (impl_2, 15))

        assert unwrap_tables( {tbl_1: tbl_1, 15: (15, tbl_2)} ) == {tbl_1: impl_1, 15: (15, impl_2)}

    def test_wrap_tables(self):
        impl_1 = MockTableImpl('impl_1', dict())
        impl_2 = MockTableImpl('impl_2', dict())
        tbl_1 = Table(impl_1)
        tbl_2 = Table(impl_2)

        assert wrap_tables( 15 ) == 15
        assert wrap_tables( tbl_1 ) == tbl_1
        assert wrap_tables( impl_1 ) == tbl_1

        assert wrap_tables( [impl_1] ) == [tbl_1]
        assert wrap_tables( [impl_1, [impl_2]] ) == [tbl_1, [tbl_2]]

        assert wrap_tables( (impl_1, )) == (tbl_1, )


class TestBuiltinVerbs:

    def test_collect(self, tbl1):
        assert (tbl1 >> collect()) == ['col1', 'col2']

    def test_select(self, tbl1, tbl2):
        assert (tbl1 >> select() >> collect()) == []
        assert (tbl1 >> select(tbl1.col1) >> collect()) == ['col1']

        with pytest.raises(ValueError):
            tbl1 >> select(tbl2.col1)

        with pytest.raises(ValueError):
            tbl1 >> select(tbl1.col1 + tbl1.col1)

    def test_mutate(self, tbl1, tbl2):
        assert (tbl1 >> mutate() >> collect()) == ['col1', 'col2']
        assert (tbl1 >> mutate(x = tbl1.col1) >> collect()) == ['col1', 'col2', 'x']
        assert (tbl1 >> select() >> mutate(x = tbl1.col1) >> collect()) == ['x']

        with pytest.raises(ValueError):
            tbl1 >> mutate(x = tbl2.col1)

        t = tbl1 >> mutate(x = tbl1.col1 + tbl1.col2)
        t >> mutate(y = t.x)

    def test_join(self, tbl1, tbl2):
        assert len(tbl1 >> join(tbl2, tbl1.col1 == tbl2.col1, 'left') >> collect()) == 5

        assert len(tbl1 >> select() >> join(tbl2, tbl1.col1 == tbl2.col1, 'left') >> collect()) == 3
        assert len(tbl1 >> join(tbl2 >> select(), tbl1.col1 == tbl2.col1, 'left') >> collect()) == 2
        assert len(tbl1 >> select() >> join(tbl2 >> select(), tbl1.col1 == tbl2.col1, 'left') >> select(tbl1.col1, tbl2.col2) >> collect()) == 2

        with pytest.raises(ValueError, match='Ambiguous'):
            # self join without alias
            tbl1 >> join(tbl1, tbl1.col1 == tbl1.col1, 'inner')

        # Test that joined columns are accessible
        t = tbl1 >> join(tbl2, tbl1.col1 == tbl2.col1, 'left')
        t >> select(t.col1, t.col1_mock2)

    def test_filter(self, tbl1, tbl2):
        tbl1 >> filter()
        tbl1 >> filter(tbl1.col1)

        with pytest.raises(ValueError):
            tbl1 >> filter(tbl2.col1)

    def test_arrange(self, tbl1, tbl2):
        tbl1 >> arrange(tbl1.col1)
        tbl1 >> arrange(-tbl1.col1)
        tbl1 >> arrange(tbl1.col1, tbl1.col2)
        tbl1 >> arrange(tbl1.col1, -tbl1.col2)

        with pytest.raises(ValueError):
            tbl1 >> arrange(tbl2.col1)
        with pytest.raises(ValueError):
            tbl1 >> arrange(tbl1.col1, -tbl2.col1)

        with pytest.raises(ValueError):
            tbl1 >> arrange(tbl1.col1 * 4)

    def test_col_pipeable(self, tbl1, tbl2):
        result = tbl1.col1 >> mutate(x = tbl1.col1 * 2)

        assert result._impl.selects == ordered_set(['col1', 'x'])
        assert list(result._impl.named_cols.fwd) == ['col1', 'x']

        with pytest.raises(ValueError):
            (tbl1.col1 + 2) >> mutate(x = 1)


class TestDataStructures:

    def test_bidict(self):
        d = bidict({'a': 1, 'b': 2, 'c': 3})

        assert len(d) == 3
        assert tuple(d.fwd.keys()) == ('a', 'b', 'c')
        assert tuple(d.fwd.values()) == (1, 2, 3)

        assert tuple(d.fwd.keys()) == tuple(d.bwd.values())
        assert tuple(d.bwd.keys()) == tuple(d.fwd.values())

        d.fwd['d'] = 4
        d.bwd[4] = 'x'

        assert tuple(d.fwd.keys()) == ('a', 'b', 'c', 'x')
        assert tuple(d.fwd.values()) == (1, 2, 3, 4)
        assert tuple(d.fwd.keys()) == tuple(d.bwd.values())
        assert tuple(d.bwd.keys()) == tuple(d.fwd.values())

        assert 'x' in d.fwd
        assert 'd' not in d.fwd

        d.clear()

        assert len(d) == 0
        assert len(d.fwd.items()) == len(d.fwd) == 0
        assert len(d.bwd.items()) == len(d.bwd) == 0

        with pytest.raises(ValueError):
            bidict({'a': 1, 'b': 1})

    def test_ordered_set(self):
        s = ordered_set([0, 1, 2])
        assert list(s) == [0, 1, 2]

        s.add(1)  # Already in set -> Noop
        assert list(s) == [0, 1, 2]
        s.add(3)  # Not in set -> add to the end
        assert list(s) == [0, 1, 2, 3]

        s.remove(1)
        assert list(s) == [0, 2, 3]
        s.add(1)
        assert list(s) == [0, 2, 3, 1]

        assert 1 in s
        assert 4 not in s
        assert len(s) == 4

        s.clear()
        assert len(s) == 0
        assert list(s) == []

        # Set Operations

        s1 = ordered_set([0, 1, 2, 3])
        s2 = ordered_set([5, 4, 3, 2])

        assert not s1.isdisjoint(s2)
        assert list(s1 | s2) == [0, 1, 2, 3, 5, 4]
        assert list(s1 ^ s2) == [0, 1, 5, 4]
        assert list(s1 & s2) == [3, 2]
        assert list(s1 - s2) == [0, 1]

        # Pop order

        s = ordered_set([0, 1, 2, 3])
        assert s.pop() == 0
        assert s.pop() == 1
        assert s.pop_back() == 3
        assert s.pop_back() == 2


class MockTableImpl(AbstractTableImpl):
    def __init__(self, name, col_names):
        super().__init__(name, {
            name: Column(name, self, 'int')
            for name in col_names
        })

    def resolve_lambda_cols(self, expr):
        return expr

    def collect(self):
        return list(self.selects)

    class ExpressionCompiler(AbstractTableImpl.ExpressionCompiler):
        def _translate(self, expr, **kwargs):
            return TypedValue(None, None)
