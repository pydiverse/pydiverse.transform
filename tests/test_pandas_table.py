import numpy as np
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from pdtransform import λ
from pdtransform.core.table import Table
from pdtransform.eager.pandas_table import PandasTableImpl
from pdtransform.core.verbs import collect, select, mutate, join, filter


df1 = pd.DataFrame({
    'col1': [1, 2, 3, 4],
    'col2': ['a', 'b', 'c', 'd'],
})

df2 = pd.DataFrame({
    'col1': [1, 2, 2, 4, 5, 6],
    'col2': [2, 2, 0, 0, 2, None],
    'col3': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
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
def tbl_left():
    return Table(PandasTableImpl('df_left', df_left.copy()))

@pytest.fixture
def tbl_right():
    return Table(PandasTableImpl('df_right', df_right.copy()))


class TestPandasTable:

    def test_select(self, tbl1):
        assert_frame_equal(tbl1 >> select(tbl1.col1) >> collect(), df1[['col1']])
        assert_frame_equal(tbl1 >> select(tbl1.col2) >> collect(), df1[['col2']])
        assert_frame_equal(tbl1 >> select() >> collect(), df1[[]])

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
                'df_right_b': [1.0, 2.0, 2.0, np.nan, np.nan]
            })
        )

        assert_frame_equal(
            tbl_left >> join(tbl_right, tbl_left.a == tbl_right.b, 'inner') >> select(tbl_left.a, tbl_right.b) >> collect(),
            pd.DataFrame({
                'a': [1, 2, 2],
                'df_right_b': [1, 2, 2]
            })
        )

        assert_frame_equal(
            tbl_left >> join(tbl_right, tbl_left.a == tbl_right.b, 'outer') >> select(tbl_left.a, tbl_right.b) >> collect(),
            pd.DataFrame({
                'a': [1.0, 2.0, 2.0, 3.0, 4.0, np.nan],
                'df_right_b': [1.0, 2.0, 2.0, np.nan, np.nan, 0.0]
            })
        )

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
            df1[df1['col1'] == 3]
        )

        # More complex expressions
        assert_frame_equal(
            tbl1 >> filter(tbl1.col1 // 2 == 1) >> collect(),
            pd.DataFrame({
                'col1': [2, 3],
                'col2': ['b', 'c']
            }, index = [1, 2])
        )

        assert_frame_equal(
            tbl1 >> filter(1 < tbl1.col1) >> filter(tbl1.col1 < 4) >> collect(),
            df1.loc[(1 < df1['col1']) & (df1['col1'] < 4)]
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
            tbl1 >> select() >> mutate(a = tbl1.col1) >> join(tbl2, λ.a == tbl2.col1, 'left') >> collect(),
            tbl1 >> select() >> mutate(a = tbl1.col1) >> join(tbl2, tbl1.col1 == tbl2.col1, 'left') >> collect()
        )

        # Filter
        assert_frame_equal(
            tbl1 >> mutate(a = tbl1.col1 * 2) >> filter(λ.a % 2 == 0) >> collect(),
            tbl1 >> mutate(a = tbl1.col1 * 2) >> filter((tbl1.col1 * 2) % 2 == 0) >> collect()
        )
