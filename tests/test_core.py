import pytest

from pdtransform import λ
from pdtransform.core import Table, AbstractTableImpl, Column
from pdtransform.core.dispatchers import inverse_partial, verb, wrap_tables, unwrap_tables
from pdtransform.core.verbs import collect, select, mutate, join, filter


@pytest.fixture
def tbl1():
    return Table(MockTableImpl('mock1', ['col1', 'col2']))


@pytest.fixture
def tbl2():
    return Table(MockTableImpl('mock2', ['col1', 'col2', 'col3']))


class TestTable:

    def test_getattr(self, tbl1):
        assert tbl1.col1 == tbl1._impl.columns['col1']
        assert tbl1.col2 == tbl1._impl.columns['col2']

        with pytest.raises(KeyError, match = 'colXXX'):
            _ = tbl1.colXXX


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

    def test_unwrap_tables(self):
        impl_1 = AbstractTableImpl('impl_1', dict())
        impl_2 = AbstractTableImpl('impl_2', dict())
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
        impl_1 = AbstractTableImpl('impl_1', dict())
        impl_2 = AbstractTableImpl('impl_2', dict())
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

        tbl1 >> mutate(x = tbl1.col1 + tbl1.col2)

    def test_join(self, tbl1, tbl2):
        assert (tbl1 >> join(tbl2, tbl1.col1 == tbl2.col1, 'left'))._impl.available_columns == tbl1._impl.available_columns | tbl2._impl.available_columns
        assert len(tbl1 >> join(tbl2, tbl1.col1 == tbl2.col1, 'left') >> collect()) == 5

        assert len(tbl1 >> select() >> join(tbl2, tbl1.col1 == tbl2.col1, 'left') >> collect()) == 3
        assert len(tbl1 >> join(tbl2 >> select(), tbl1.col1 == tbl2.col1, 'left') >> collect()) == 2
        assert len(tbl1 >> select() >> join(tbl2 >> select(), tbl1.col1 == tbl2.col1, 'left') >> select(tbl1.col1, tbl2.col2) >> collect()) == 2

        with pytest.raises(ValueError, match='Ambiguous'):
            # self join without alias
            tbl1 >> join(tbl1, tbl1.col1 == tbl1.col1, 'inner')

    def test_filter(self, tbl1, tbl2):
        tbl1 >> filter()
        tbl1 >> filter(tbl1.col1)

        with pytest.raises(ValueError):
            tbl1 >> filter(tbl2.col1)


class MockTableImpl(AbstractTableImpl):
    def __init__(self, name, col_names):
        super().__init__(name, {
            name: Column(name, self, 'int')  # TODO: it should be possible to specify the dtype
            for name in col_names
        })

    def resolve_lambda_cols(self, expr):
        return expr

    def collect(self):
        return list(self.selects.keys())

