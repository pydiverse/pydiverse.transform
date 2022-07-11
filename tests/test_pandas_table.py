import numpy as np
import pandas as pd
import pytest

from pdtransform import λ
from pdtransform.core import functions as f
from pdtransform.core.alignment import aligned, eval_aligned
from pdtransform.core.dispatchers import Pipeable, verb
from pdtransform.core.table import Table
from pdtransform.core.verbs import *
from pdtransform.eager.pandas_table import PandasTableImpl
from pdtransform.util.testing import assert_equal

df1 = pd.DataFrame({
    'col1': [1, 2, 3, 4],
    'col2': ['a', 'b', 'c', 'd'],
})

df2 = pd.DataFrame({
    'col1': [1, 2, 2, 4, 5, 6],
    'col2': [2, 2, 0, 0, 2, None],
    'col3': [0.0, 0.1, 0.2, 0.3, 0.01, 0.02],
})

df3 = pd.DataFrame({
    'col1': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    'col2': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    'col3': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    'col4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11],
    'col5': list('abcdefghijkl')
})

df_left = pd.DataFrame({
    'a': [1, 2, 3, 4],
})

df_right = pd.DataFrame({
    'b': [0, 1, 2, 2],
    'c': [5, 6, 7, 8],
})


@pytest.fixture
def tbl1():
    return Table(PandasTableImpl('df1', df1.copy()))

@pytest.fixture
def tbl2():
    return Table(PandasTableImpl('df2', df2.copy()))

@pytest.fixture
def tbl3():
    return Table(PandasTableImpl('df3', df3.copy()))

@pytest.fixture
def tbl_left():
    return Table(PandasTableImpl('df_left', df_left.copy()))

@pytest.fixture
def tbl_right():
    return Table(PandasTableImpl('df_right', df_right.copy()))

def assert_not_inplace(tbl: Table[PandasTableImpl], operation: Pipeable):
    """
    Operations should not happen in-place. They should always return a new dataframe.
    """
    initial = tbl._impl.df.copy()
    tbl >> operation
    after = tbl._impl.df

    pd.util.testing.assert_frame_equal(initial, after)


class TestPandasTable:

    def test_dtype(self, tbl1, tbl2):
        assert tbl1.col1._.dtype == 'int'
        assert tbl1.col2._.dtype == 'str'
        
        assert tbl2.col1._.dtype == 'int'
        assert tbl2.col2._.dtype == 'int'
        assert tbl2.col3._.dtype == 'float'

    def test_build_query(self, tbl1):
        assert (tbl1 >> build_query()) is None

    def test_select(self, tbl1):
        assert_not_inplace(tbl1, select(tbl1.col1))
        assert_equal(tbl1 >> select(tbl1.col1), df1[['col1']])
        assert_equal(tbl1 >> select(tbl1.col2), df1[['col2']])
        assert_equal(tbl1 >> select(), df1[[]])

    def test_mutate(self, tbl1):
        assert_not_inplace(tbl1, mutate(x = tbl1.col1))

        assert_equal(
            tbl1 >> mutate(col1times2 = tbl1.col1 * 2),
            pd.DataFrame({
                'col1': [1, 2, 3, 4],
                'col2': ['a', 'b', 'c', 'd'],
                'col1times2': [2, 4, 6, 8],
            })
        )

        assert_equal(
            tbl1 >> select() >> mutate(col1times2 = tbl1.col1 * 2),
            pd.DataFrame({
                'col1times2': [2, 4, 6, 8],
            })
        )

        # Check proper column referencing
        t = tbl1 >> mutate(col2 = tbl1.col1, col1 = tbl1.col2) >> select()
        assert_equal(
            t >> mutate(x = t.col1, y = t.col2),
            tbl1 >> select() >> mutate(x = tbl1.col2, y = tbl1.col1)
        )
        assert_equal(
            t >> mutate(x = tbl1.col1, y = tbl1.col2),
            tbl1 >> select() >> mutate(x = tbl1.col1, y = tbl1.col2)
        )

    def test_join(self, tbl_left, tbl_right):
        assert_not_inplace(tbl_left, join(tbl_right, tbl_left.a == tbl_right.b, 'left'))

        assert_equal(
            tbl_left >> join(tbl_right, tbl_left.a == tbl_right.b, 'left') >> select(tbl_left.a, tbl_right.b),
            pd.DataFrame({
                'a': [1, 2, 2, 3, 4],
                'b_df_right': [1.0, 2.0, 2.0, np.nan, np.nan]
            })
        )

        assert_equal(
            tbl_left >> join(tbl_right, tbl_left.a == tbl_right.b, 'inner') >> select(tbl_left.a, tbl_right.b),
            pd.DataFrame({
                'a': [1, 2, 2],
                'b_df_right': [1, 2, 2]
            })
        )

        assert_equal(
            tbl_left >> join(tbl_right, tbl_left.a == tbl_right.b, 'outer') >> select(tbl_left.a, tbl_right.b),
            pd.DataFrame({
                'a': [1.0, 2.0, 2.0, 3.0, 4.0, np.nan],
                'b_df_right': [1.0, 2.0, 2.0, np.nan, np.nan, 0.0]
            })
        )

    def test_join_order(self):
        dfA     = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': [2, 1, 3, 4, 6, 5]}, index=[1, 1, 'x', 'y', 3, 'x'])
        dfA_bad = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': [2, 1, 3, 4, 6, 6]})
        dfB     = pd.DataFrame({'b': [3, 4, 1], 'c': [7, 8, 9]}, index=[0, 0, -1])
        dfB_bad = pd.DataFrame({'b': [3, 4, 4], 'c': [7, 8, 9]})

        tblA     = Table(PandasTableImpl('dfA', dfA))
        tblA_bad = Table(PandasTableImpl('dfA_bad', dfA_bad))
        tblB     = Table(PandasTableImpl('dfB', dfB))
        tblB_bad = Table(PandasTableImpl('dfB_bad', dfB_bad))

        # Preserve tblA index
        pd.testing.assert_index_equal(
            (tblA >> left_join(tblB, tblA.b == tblB.b, validate = '1:?') >> collect()).index,
            dfA.index
        )

        pd.testing.assert_series_equal(
            (tblA >> left_join(tblB, tblA.b == tblB.b, validate = '1:?') >> collect())['a'],
            (tblA >> collect())['a']
        )

        with pytest.raises(Exception):
            tblA >> left_join(tblB_bad, tblA.b == tblB_bad.b, validate='1:?')
        with pytest.raises(Exception):
            tblA_bad >> left_join(tblB, tblA_bad.b == tblB_bad.b, validate='1:?')

    def test_filter(self, tbl1, tbl2):
        assert_not_inplace(tbl1, filter(tbl1.col1 == 3))

        # Simple filter expressions
        assert_equal(
            tbl1 >> filter(),
            df1
        )

        assert_equal(
            tbl1 >> filter(tbl1.col1 == tbl1.col1),
            df1
        )

        assert_equal(
            tbl1 >> filter(tbl1.col1 == 3),
            df1[df1['col1'] == 3]
        )

        # More complex expressions
        assert_equal(
            tbl1 >> filter(tbl1.col1 // 2 == 1),
            pd.DataFrame({
                'col1': [2, 3],
                'col2': ['b', 'c']
            }, index = [1, 2])
        )

        assert_equal(
            tbl1 >> filter(1 < tbl1.col1) >> filter(tbl1.col1 < 4),
            df1.loc[(1 < df1['col1']) & (df1['col1'] < 4)]
        )

    def test_arrange(self, tbl2):
        assert_not_inplace(tbl2, arrange(tbl2.col2))

        assert_equal(
            tbl2 >> arrange(tbl2.col3) >> select(tbl2.col3),
            df2[['col3']].sort_values('col3', ascending = True, kind = 'mergesort')
        )

        assert_equal(
            tbl2 >> arrange(-tbl2.col3) >> select(tbl2.col3),
            df2[['col3']].sort_values('col3', ascending = False, kind = 'mergesort')
        )

        assert_equal(
            tbl2 >> arrange(tbl2.col1, tbl2.col2),
            df2.sort_values(['col1', 'col2'], ascending = [True, True], kind = 'mergesort')
        )

        assert_equal(
            tbl2 >> arrange(tbl2.col1, tbl2.col2),
            tbl2 >> arrange(tbl2.col2) >> arrange(tbl2.col1)
        )

        assert_equal(
            tbl2 >> arrange(--tbl2.col3),
            tbl2 >> arrange(tbl2.col3)
        )

    def test_summarise(self, tbl3):
        assert_equal(
            tbl3 >> summarise(mean = tbl3.col1.mean(), max = tbl3.col4.max()),
            pd.DataFrame({
                'mean': [1.0],
                'max': [11]
            })
        )

        assert_equal(
            tbl3 >> group_by(tbl3.col1) >> summarise(mean = tbl3.col4.mean()),
            pd.DataFrame({
                'col1': [0, 1, 2],
                'mean': [1.5, 5.5, 9.5]
            })
        )

        assert_equal(
            tbl3 >> summarise(mean = tbl3.col4.mean()) >> mutate(mean_2x = λ.mean * 2),
            pd.DataFrame({
                'mean': [5.5],
                'mean_2x': [11.0]
            })
        )

    def test_group_by(self, tbl3):
        # Grouping doesn't change the result
        assert_equal(
            tbl3 >> group_by(tbl3.col1),
            tbl3
        )
        assert_equal(
            tbl3 >> summarise(mean4 = tbl3.col4.mean()) >> group_by(λ.mean4),
            tbl3 >> summarise(mean4 = tbl3.col4.mean())
        )

        # Groupings can be added
        assert_equal(
            tbl3 >> group_by(tbl3.col1) >> group_by(tbl3.col2, add=True) >> summarise(mean3 = tbl3.col3.mean(), mean4 = tbl3.col4.mean()),
            tbl3 >> group_by(tbl3.col1, tbl3.col2) >> summarise(mean3 = tbl3.col3.mean(), mean4 = tbl3.col4.mean())
        )

        # Ungroup doesn't change the result
        assert_equal(
            tbl3 >> group_by(tbl3.col1) >> summarise(mean4 = tbl3.col4.mean()) >> ungroup(),
            tbl3 >> group_by(tbl3.col1) >> summarise(mean4 = tbl3.col4.mean())
        )

    def test_alias(self, tbl1, tbl2):
        assert_not_inplace(tbl1, alias('tblxxx'))

        x = tbl2 >> alias('x')
        assert(x._impl.name == 'x')

        # Check that applying alias doesn't change the output
        a = tbl1 >> mutate(xyz = (tbl1.col1 * tbl1.col1) // 2) >> join(tbl2, tbl1.col1 == tbl2.col1, 'left') >> mutate(col1 = tbl1.col1 - λ.xyz)
        b = a >> alias('b')

        assert_equal(
            a,
            b
        )

        # Self Join
        assert_equal(
            tbl2 >> join(x, tbl2.col1 == x.col1, 'left'),
            df2.merge(df2.rename(columns = {'col1': 'col1_x', 'col2': 'col2_x', 'col3': 'col3_x'}), how = 'left', left_on = 'col1', right_on = 'col1_x')
        )

    def test_window_functions(self, tbl3):
        # Everything else should stay the same (index preserved)
        assert_equal(
            tbl3 >> mutate(x = f.row_number(arrange=[-λ.col4])) >> select(*tbl3),
            df3
        )

        # Assert correct result
        assert_equal(
            (tbl3 >> group_by(λ.col2) >> select() >> mutate(x = f.row_number(arrange=[-λ.col4])) >> collect()),
            pd.DataFrame({'x': [6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1]})
        )

    def test_lambda_column(self, tbl1, tbl2):
        # Select
        assert_equal(
            tbl1 >> select(λ.col1),
            tbl1 >> select(tbl1.col1)
        )

        # Mutate
        assert_equal(
            tbl1 >> mutate(a = tbl1.col1 * 2) >> select() >> mutate(b = λ.a * 2),
            tbl1 >> select() >> mutate(b = tbl1.col1 * 4)
        )

        assert_equal(
            tbl1 >> mutate(a = tbl1.col1 * 2) >> mutate(b = λ.a * 2, a = tbl1.col1) >> select(λ.b),
            tbl1 >> select() >> mutate(b = tbl1.col1 * 4)
        )

        # Join
        assert_equal(
            tbl1 >> select() >> mutate(a = tbl1.col1) >> join(tbl2, λ.a == tbl2.col1, 'left'),
            tbl1 >> select() >> mutate(a = tbl1.col1) >> join(tbl2, tbl1.col1 == tbl2.col1, 'left')
        )

        # Join that also uses lambda for the right table
        assert_equal(
            tbl1 >> select() >> mutate(a = tbl1.col1) >> join(tbl2, λ.a == λ.col1_df2, 'left'),
            tbl1 >> select() >> mutate(a = tbl1.col1) >> join(tbl2, tbl1.col1 == tbl2.col1, 'left')
        )

        # Filter
        assert_equal(
            tbl1 >> mutate(a = tbl1.col1 * 2) >> filter(λ.a % 2 == 0),
            tbl1 >> mutate(a = tbl1.col1 * 2) >> filter((tbl1.col1 * 2) % 2 == 0)
        )

        # Arrange
        assert_equal(
            tbl1 >> mutate(a = tbl1.col1 * 2) >> arrange(λ.a),
            tbl1 >> arrange(tbl1.col1) >> mutate(a = tbl1.col1 * 2),
        )

    def test_table_setitem(self, tbl_left, tbl_right):
        tl = tbl_left >> alias('df_left')
        tr = tbl_right >> alias('df_right')

        # Iterate over cols and modify
        for col in tl:
            tl[col] = (col * 2) % 3
        for col in tr:
            tr[col] = (col * 2) % 5

        # Check if it worked...
        assert_equal(
            (
                tl >> join(tr, λ.a == λ.b_df_right, 'left')
            ),
            (tbl_left
                >> mutate(a = (tbl_left.a * 2) % 3)
                >> join(
                        tbl_right >> mutate(b = (tbl_right.b * 2) % 5, c = (tbl_right.c * 2) % 5),
                        λ.a == λ.b_df_right,
                        'left'
                    )

             )
        )

    def test_custom_verb(self, tbl1):
        @verb
        def double_col1(tbl):
            tbl[λ.col1] = λ.col1 * 2
            return tbl

        # Custom verb should not mutate input object
        assert_not_inplace(tbl1, double_col1())

        assert_equal(
            tbl1 >> double_col1(),
            tbl1 >> mutate(col1 = λ.col1 * 2)
        )


class TestPandasAligned:

    def test_eval_aligned(self, tbl1, tbl3, tbl_left, tbl_right):
        # No exception with correct length
        eval_aligned(tbl_left.a + tbl_left.a)
        eval_aligned(tbl_left.a + tbl_right.b)

        with pytest.raises(ValueError):
            eval_aligned(tbl1.col1 + tbl3.col1)

        # Test aggregate functions still work
        eval_aligned(tbl1.col1 + tbl3.col1.mean())

        # Test that `with_` argument gets enforced
        eval_aligned(tbl1.col1 + tbl1.col1, with_ = tbl1)
        eval_aligned(tbl_left.a * 2, with_ = tbl_left)
        eval_aligned(tbl_left.a * 2, with_ = tbl_right)  # Same length
        eval_aligned(tbl1.col1.mean(), with_ = tbl_left)  # Aggregate is aligned with everything

        with pytest.raises(ValueError):
            eval_aligned(tbl3.col1 * 2, with_ = tbl1)

    def test_aligned_decorator(self, tbl1, tbl3, tbl_left, tbl_right):
        @aligned(with_ = 'a')
        def f(a, b):
            return a + b

        f(tbl3.col1, tbl3.col2)
        f(tbl_left.a, tbl_right.b)

        with pytest.raises(ValueError):
            f(tbl1.col1, tbl3.col1)

        # Bad Alignment of return type
        @aligned(with_ = 'a')
        def f(a, b):
            return a.mean() + b

        with pytest.raises(ValueError):
            f(tbl1.col1, tbl3.col1)

        # Invalid with_ argument
        with pytest.raises(Exception):
            aligned(with_ = 'x')(lambda a: 0)

    def test_col_addition(self, tbl_left, tbl_right):
        @aligned(with_='a')
        def f(a, b):
            return a + b

        assert_equal(
            tbl_left >> mutate(x = f(tbl_left.a, tbl_right.b)) >> select(λ.x),
            pd.DataFrame({'x': (df_left['a'] + df_right['b']) })
        )

        with pytest.raises(ValueError):
            f(tbl_left.a, (tbl_right >> filter(λ.b == 2)).b)

        with pytest.raises(ValueError):
            x = f(tbl_left.a, tbl_right.b)
            tbl_left >> filter(λ.a <= 3) >> mutate(x = x)


class TestPrintAndRepr:

    def test_table_str(self, tbl1):
        # Table: df1, backend: PandasTableImpl
        #    col1 col2
        # 0     1    a
        # 1     2    b
        # 2     3    c
        # 3     4    d

        tbl_str = str(tbl1)

        assert 'df1' in tbl_str
        assert 'PandasTableImpl' in tbl_str
        assert str(df1) in tbl_str

    def test_table_repr_html(self, tbl1):
        # jupyter html
        assert 'exception' not in tbl1._repr_html_()

    def test_col_str(self, tbl1):
        # Symbolic Expression: <df1.col1(int)>
        # dtype: int
        #
        # 0    1
        # 1    2
        # 2    3
        # 3    4
        # Name: df1_col1_XXXXXXXX, dtype: Int64

        col1_str = str(tbl1.col1)
        series = tbl1._impl.df[tbl1._impl.df_name_mapping[tbl1.col1._.uuid]]

        assert str(series) in col1_str
        assert 'exception' not in col1_str

    def test_col_html_repr(self, tbl1):
        assert 'exception' not in tbl1.col1._repr_html_()

    def test_expr_str(self, tbl1):
        expr_str = str(tbl1.col1 * 2)
        assert 'exception' not in expr_str

    def test_expr_html_repr(self, tbl1):
        assert 'exception' not in (tbl1.col1 * 2)._repr_html_()

    def test_lambda_str(self, tbl1):
        assert 'exception' in str(λ.col)
        assert 'exception' in str(λ.col1 + tbl1.col1)

    def test_eval_expr_str(self, tbl_left, tbl_right):
        valid = tbl_left.a + tbl_right.b
        invalid = tbl_left.a + (tbl_right >> filter(λ.b == 2)).b

        assert 'exception' not in str(valid)
        assert 'exception' in str(invalid)
