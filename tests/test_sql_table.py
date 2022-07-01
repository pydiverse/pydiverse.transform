import numpy as np
import pandas as pd
import pytest
import sqlalchemy
from pandas.testing import assert_frame_equal

from pdtransform import λ
from pdtransform.core.alignment import aligned, eval_aligned
from pdtransform.core.table import Table
from pdtransform.core.verbs import alias, arrange, collect, filter, group_by, join, mutate, select, summarise, ungroup
from pdtransform.lazy.sql_table import SQLTableImpl
from pdtransform.lazy.verbs import show_query

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
def engine():
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    df1.to_sql('df1', engine, index = False, if_exists = 'replace')
    df2.to_sql('df2', engine, index = False, if_exists = 'replace')
    df3.to_sql('df3', engine, index = False, if_exists = 'replace')
    df_left.to_sql('df_left', engine, index = False, if_exists = 'replace')
    df_right.to_sql('df_right', engine, index = False, if_exists = 'replace')
    return engine

@pytest.fixture
def tbl1(engine):
    return Table(SQLTableImpl(engine, 'df1'))

@pytest.fixture
def tbl2(engine):
    return Table(SQLTableImpl(engine, 'df2'))

@pytest.fixture
def tbl3(engine):
    return Table(SQLTableImpl(engine, 'df3'))


@pytest.fixture
def tbl_left(engine):
    return Table(SQLTableImpl(engine, 'df_left'))

@pytest.fixture
def tbl_right(engine):
    return Table(SQLTableImpl(engine, 'df_right'))

class TestSQLTable:

    def test_show_query(self, tbl1, capfd):
        tbl1 >> show_query()
        out = capfd.readouterr().out
        expected_out = 'SELECT df1.col1 AS col1, df1.col2 AS col2 FROM df1'

        assert out.lower().split() == expected_out.lower().split()

        # Verify that it is chainable
        tbl1 >> show_query() >> collect()

    def test_select(self, tbl1, tbl2):
        assert_frame_equal(tbl1 >> select(tbl1.col1) >> collect(), df1[['col1']])
        assert_frame_equal(tbl1 >> select(tbl1.col2) >> collect(), df1[['col2']])

    def test_mutate(self, tbl1):
        assert_frame_equal(
            tbl1 >> mutate(col1times2 = tbl1.col1 * 2) >> collect(),
            pd.DataFrame({
                'col1': [1, 2, 3, 4],
                'col2': ['a', 'b', 'c', 'd'],
                'col1times2': [2, 4, 6, 8],
            })
        )

        assert_frame_equal(
            tbl1 >> select() >> mutate(col1times2 = tbl1.col1 * 2) >> collect(),
            pd.DataFrame({
                'col1times2': [2, 4, 6, 8],
            })
        )

        # Check proper column referencing
        t = tbl1 >> mutate(col2 = tbl1.col1, col1 = tbl1.col2) >> select()
        assert_frame_equal(
            t >> mutate(x = t.col1, y = t.col2) >> collect(),
            tbl1 >> select() >> mutate(x = tbl1.col2, y = tbl1.col1) >> collect()
        )
        assert_frame_equal(
            t >> mutate(x = tbl1.col1, y = tbl1.col2) >> collect(),
            tbl1 >> select() >> mutate(x = tbl1.col1, y = tbl1.col2) >> collect()
        )

    def test_join(self, tbl_left, tbl_right):
        assert_frame_equal(
            tbl_left >> join(tbl_right, tbl_left.a == tbl_right.b, 'left') >> select(tbl_left.a, tbl_right.b) >> collect(),
            pd.DataFrame({
                'a': [1, 2, 2, 3, 4],
                'b_df_right': [1.0, 2.0, 2.0, np.nan, np.nan]
            })
        )

        assert_frame_equal(
            tbl_left >> join(tbl_right, tbl_left.a == tbl_right.b, 'inner') >> select(tbl_left.a, tbl_right.b) >> collect(),
            pd.DataFrame({
                'a': [1, 2, 2],
                'b_df_right': [1, 2, 2]
            })
        )

        # TODO: Implement test for outer join.
        #       SQLite doesn't support outer joins :-(

        # assert_frame_equal(
        #     tbl_left >> join(tbl_right, tbl_left.a == tbl_right.b, 'outer') >> select(tbl_left.a, tbl_right.b) >> collect(),
        #     pd.DataFrame({
        #         'a': [1.0, 2.0, 2.0, 3.0, 4.0, np.nan],
        #         'b_df_right': [1.0, 2.0, 2.0, np.nan, np.nan, 0.0]
        #     })
        # )

    def test_filter(self, tbl1, tbl2):
        # Simple filter expressions
        assert_frame_equal(
            tbl1 >> filter() >> collect(),
            df1
        )

        assert_frame_equal(
            tbl1 >> filter(tbl1.col1 == tbl1.col1) >> collect(),
            df1
        )

        assert_frame_equal(
            tbl1 >> filter(tbl1.col1 == 3) >> collect(),
            df1[df1['col1'] == 3].reset_index(drop=True)
        )

        # More complex expressions
        assert_frame_equal(
            tbl1 >> filter(tbl1.col1 // 2 == 1) >> collect(),
            pd.DataFrame({
                'col1': [2, 3],
                'col2': ['b', 'c']
            }).reset_index(drop=True)
        )

        assert_frame_equal(
            tbl1 >> filter(1 < tbl1.col1) >> filter(tbl1.col1 < 4) >> collect(),
            df1.loc[(1 < df1['col1']) & (df1['col1'] < 4)].reset_index(drop=True)
        )

    def test_arrange(self, tbl2):
        assert_frame_equal(
            tbl2 >> arrange(tbl2.col3) >> select(tbl2.col3) >> collect(),
            df2[['col3']].sort_values('col3', ascending = True, kind = 'mergesort').reset_index(drop=True)
        )

        assert_frame_equal(
            tbl2 >> arrange(-tbl2.col3) >> select(tbl2.col3) >> collect(),
            df2[['col3']].sort_values('col3', ascending = False, kind = 'mergesort').reset_index(drop=True)
        )

        assert_frame_equal(
            tbl2 >> arrange(tbl2.col1, tbl2.col2) >> collect(),
            df2.sort_values(['col1', 'col2'], ascending = [True, True], kind = 'mergesort').reset_index(drop=True)
        )

        assert_frame_equal(
            tbl2 >> arrange(tbl2.col1, tbl2.col2) >> collect(),
            tbl2 >> arrange(tbl2.col2) >> arrange(tbl2.col1) >> collect()
        )

        assert_frame_equal(
            tbl2 >> arrange(--tbl2.col3) >> collect(),
            tbl2 >> arrange(tbl2.col3) >> collect()
        )

    def test_summarise(self, tbl3):
        assert_frame_equal(
            tbl3 >> summarise(mean = tbl3.col1.mean(), max = tbl3.col4.max()) >> collect(),
            pd.DataFrame({
                'mean': [1.0],
                'max': [11]
            })
        )

        assert_frame_equal(
            tbl3 >> group_by(tbl3.col1) >> summarise(mean = tbl3.col4.mean()) >> collect(),
            pd.DataFrame({
                'col1': [0, 1, 2],
                'mean': [1.5, 5.5, 9.5]
            })
        )

        assert_frame_equal(
            tbl3 >> summarise(mean = tbl3.col4.mean()) >> mutate(mean_2x = λ.mean * 2) >> collect(),
            pd.DataFrame({
                'mean': [5.5],
                'mean_2x': [11.0]
            })
        )

    def test_group_by(self, tbl3):
        # Grouping doesn't change the result
        assert_frame_equal(
            tbl3 >> group_by(tbl3.col1) >> collect(),
            tbl3 >> collect()
        )
        assert_frame_equal(
            tbl3 >> summarise(mean4 = tbl3.col4.mean()) >> group_by(λ.mean4) >> collect(),
            tbl3 >> summarise(mean4 = tbl3.col4.mean()) >> collect()
        )

        # Groupings can be added
        assert_frame_equal(
            tbl3 >> group_by(tbl3.col1) >> group_by(tbl3.col2, add=True) >> summarise(mean3 = tbl3.col3.mean(), mean4 = tbl3.col4.mean()) >> collect(),
            tbl3 >> group_by(tbl3.col1, tbl3.col2) >> summarise(mean3 = tbl3.col3.mean(), mean4 = tbl3.col4.mean()) >> collect()
        )

        # Ungroup doesn't change the result
        assert_frame_equal(
            tbl3 >> group_by(tbl3.col1) >> summarise(mean4 = tbl3.col4.mean()) >> ungroup() >> collect(),
            tbl3 >> group_by(tbl3.col1) >> summarise(mean4 = tbl3.col4.mean()) >> collect()
        )

    def test_alias(self, tbl1, tbl2):
        x = tbl2 >> alias('x')
        assert(x._impl.name == 'x')

        # Check that applying alias doesn't change the output
        a = tbl1 >> mutate(xyz = (tbl1.col1 * tbl1.col1) // 2) >> join(tbl2, tbl1.col1 == tbl2.col1, 'left') >> mutate(col1 = tbl1.col1 - λ.xyz)
        b = a >> alias('b')

        assert_frame_equal(
            a >> collect(),
            b >> collect()
        )

        # Self Join
        self_join = tbl2 >> join(x, tbl2.col1 == x.col1, 'left') >> alias('self_join')
        self_join >>= arrange(*self_join)

        self_join_expected = df2.merge(df2.rename(columns = {'col1': 'col1_x', 'col2': 'col2_x', 'col3': 'col3_x'}), how = 'left', left_on = 'col1', right_on = 'col1_x')
        self_join_expected = self_join_expected.sort_values(by = [col._.name for col in self_join])

        assert_frame_equal(
            self_join >> collect(),
            self_join_expected.reset_index(drop = True)
        )

    def test_lambda_column(self, tbl1, tbl2):
        # Select
        assert_frame_equal(
            tbl1 >> select(λ.col1) >> collect(),
            tbl1 >> select(tbl1.col1) >> collect()
        )

        # Mutate
        assert_frame_equal(
            tbl1 >> mutate(a = tbl1.col1 * 2) >> select() >> mutate(b = λ.a * 2) >> collect(),
            tbl1 >> select() >> mutate(b = tbl1.col1 * 4) >> collect()
        )

        assert_frame_equal(
            tbl1 >> mutate(a = tbl1.col1 * 2) >> mutate(b = λ.a * 2, a = tbl1.col1) >> select(λ.b) >> collect(),
            tbl1 >> select() >> mutate(b = tbl1.col1 * 4) >> collect()
        )

        # Join
        assert_frame_equal(
            tbl1 >> select() >> mutate(a = tbl1.col1 * 2) >> join(tbl2, λ.a == tbl2.col1, 'left') >> collect(),
            tbl1 >> select() >> mutate(a = tbl1.col1 * 2) >> join(tbl2, tbl1.col1 * 2 == tbl2.col1, 'left') >> collect()
        )

        # Join that also uses lambda for the right table
        assert_frame_equal(
            tbl1 >> select() >> mutate(a = tbl1.col1) >> join(tbl2, λ.a == λ.col1_df2, 'left') >> collect(),
            tbl1 >> select() >> mutate(a = tbl1.col1) >> join(tbl2, tbl1.col1 == tbl2.col1, 'left') >> collect()
        )

        # Filter
        assert_frame_equal(
            tbl1 >> mutate(a = tbl1.col1 * 2) >> filter(λ.a % 2 == 0) >> collect(),
            tbl1 >> mutate(a = tbl1.col1 * 2) >> filter((tbl1.col1 * 2) % 2 == 0) >> collect()
        )

        # Arrange
        assert_frame_equal(
            tbl1 >> mutate(a = tbl1.col1 * 2) >> arrange(λ.a) >> collect(),
            tbl1 >> arrange(tbl1.col1) >> mutate(a = tbl1.col1 * 2) >> collect(),
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
        assert_frame_equal(
            (
                tl >> join(tr, λ.a == λ.b_df_right, 'left') >> collect()
            ),
            (tbl_left
                >> mutate(a = (tbl_left.a * 2) % 3)
                >> join(
                        tbl_right >> mutate(b = (tbl_right.b * 2) % 5, c = (tbl_right.c * 2) % 5),
                        λ.a == λ.b_df_right,
                        'left'
                    )
                >> collect()
            )
        )


class TestSQLAligned:

    def test_eval_aligned(self, tbl1, tbl3, tbl_left, tbl_right):
        # Columns must be from same table
        eval_aligned(tbl_left.a + tbl_left.a)
        eval_aligned(tbl3.col1 + tbl3.col2)

        # Derived columns are also OK
        tbl1_mutate = tbl1 >> mutate(x = tbl1.col1 * 2)
        eval_aligned(tbl1.col1 + tbl1_mutate.x)


        with pytest.raises(ValueError):
            eval_aligned(tbl1.col1 + tbl3.col1)
        with pytest.raises(ValueError):
            eval_aligned(tbl_left.a + tbl_right.b)
        with pytest.raises(ValueError):
            eval_aligned(tbl1.col1 + tbl3.col1.mean())
        with pytest.raises(ValueError):
            tbl1_joined = tbl1 >> join(tbl3, tbl1.col1 == tbl3.col1, how='left')
            eval_aligned(tbl1.col1 + tbl1_joined.col1)

        # Test that `with_` argument gets enforced
        eval_aligned(tbl1.col1 + tbl1.col1, with_ = tbl1)
        eval_aligned(tbl_left.a * 2, with_ = tbl_left)
        eval_aligned(tbl1.col1, with_ = tbl1_mutate)

        with pytest.raises(ValueError):
            eval_aligned(tbl1.col1.mean(), with_ = tbl_left)

        with pytest.raises(ValueError):
            eval_aligned(tbl3.col1 * 2, with_ = tbl1)

        with pytest.raises(ValueError):
            eval_aligned(tbl_left.a, with_ = tbl_right)

    def test_aligned_decorator(self, tbl1, tbl3, tbl_left, tbl_right):
        @aligned(with_ = 'a')
        def f(a, b):
            return a + b

        f(tbl3.col1, tbl3.col2)
        f(tbl_right.b, tbl_right.c)

        with pytest.raises(ValueError):
            f(tbl1.col1, tbl3.col1)

        with pytest.raises(ValueError):
            f(tbl_left.a, tbl_right.b)

        # Check with_ parameter gets enforced
        @aligned(with_ = 'a')
        def f(a, b):
            return b

        f(tbl1.col1, tbl1.col2)
        with pytest.raises(ValueError):
            f(tbl1.col1, tbl3.col1)

        # Invalid with_ argument
        with pytest.raises(Exception):
            aligned(with_ = 'x')(lambda a: 0)

    def test_col_addition(self, tbl3):
        @aligned(with_='a')
        def f(a, b):
            return a + b

        assert_frame_equal(
            tbl3 >> mutate(x = f(tbl3.col1, tbl3.col2)) >> select(λ.x) >> collect(),
            pd.DataFrame({'x': [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]})
        )

        # Test if it also works with derived tables
        tbl3_mutate = tbl3 >> mutate(x = tbl3.col1 * 2)
        tbl3 >> mutate(x = f(tbl3_mutate.col1, tbl3_mutate.x))

        with pytest.raises(ValueError):
            tbl3 >> arrange(λ.col1) >> mutate(x = f(tbl3.col1, tbl3.col2))

        with pytest.raises(ValueError):
            tbl3 >> filter(λ.col1 == 1) >> mutate(x = f(tbl3.col1, tbl3.col2))
