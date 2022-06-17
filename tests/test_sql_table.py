import numpy as np
import pandas as pd
import pytest
import sqlalchemy
from pandas.testing import assert_frame_equal

from pdtransform import λ
from pdtransform.core.table import Table
from pdtransform.core.verbs import collect, select, mutate, join, filter, arrange, alias
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
        self_join_expected = self_join_expected.sort_values(by = [col._name for col in self_join])

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
                tl >> join(tr, λ.a == λ.b_df_right, 'left') >> show_query() >> collect()
            ),
            (tbl_left
                >> mutate(a = (tbl_left.a * 2) % 3)
                >> join(
                        tbl_right >> mutate(b = (tbl_right.b * 2) % 5, c = (tbl_right.c * 2) % 5),
                        λ.a == λ.b_df_right,
                        'left'
                    )
                >> show_query()
                >> collect()
            )
        )

