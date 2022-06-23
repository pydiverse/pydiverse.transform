import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import sqlalchemy
from pandas.testing import assert_frame_equal

from pdtransform import λ
from pdtransform.core.table import Table
from pdtransform.core.verbs import alias, arrange, collect, filter, group_by, join, mutate, select, summarise, ungroup
from pdtransform.eager.pandas_table import PandasTableImpl
from pdtransform.lazy.sql_table import SQLTableImpl


dataframes = {
    'df1': pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': ['a', 'b', 'c', 'd'],
    }),

    'df2': pd.DataFrame({
        'col1': [1, 2, 2, 4, 5, 6],
        'col2': [2, 2, 0, 0, 2, None],
        'col3': [0.0, 0.1, 0.2, 0.3, 0.01, 0.02],
    }),

    'df3': pd.DataFrame({
        'col1': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        'col2': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        'col3': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        'col4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11],
        'col5': list('abcdefghijkl')
    }),

    'df_left': pd.DataFrame({
        'a': [1, 2, 3, 4],
    }),

    'df_right': pd.DataFrame({
        'b': [0, 1, 2, 2],
        'c': [5, 6, 7, 8],
    }),
}


def pandas_impls():
    return { name: PandasTableImpl(name, df) for name, df in dataframes.items() }


def sql_impls():
    engine = sqlalchemy.create_engine('sqlite:///:memory:')
    impls = {}
    for name, df in dataframes.items():
        df.to_sql(name, engine, index = False, if_exists = 'replace')
        impls[name] = SQLTableImpl(engine, name)
    return impls


impls = {
    'pandas': pandas_impls,
    'sql_sqlite': sql_impls
}


def tables(names: list[str]):
    param_names = ','.join([f'{name}_x,{name}_y' for name in names])

    tables = defaultdict(lambda: [])
    impl_names = impls.keys()
    for _, factory in impls.items():
        for df_name, impl in factory().items():
            tables[df_name].append(Table(impl))

    param_combinations = ((zip(*itertools.combinations(tables[name], 2))) for name in names)
    param_combinations = itertools.chain(*param_combinations)
    param_combinations = list(zip(*param_combinations))

    names_combinations = list(itertools.combinations(impl_names, 2))

    params = [
        pytest.param(*p, id=f'{id[0]} {id[1]}')
        for p, id in zip(param_combinations, names_combinations)
    ]

    return pytest.mark.parametrize(param_names, params)


def assert_result_equal(x, y, pipe_factory):
    if not isinstance(x, (list, tuple)):
        x = (x,)
        y = (y,)

    dfx = pipe_factory(*x) >> collect()
    dfy = pipe_factory(*y) >> collect()

    assert_frame_equal(dfx, dfy)


class TestSelect:

    @tables(['df1'])
    def test_simple_select(self, df1_x, df1_y):
        assert_result_equal(df1_x, df1_y, lambda t: t >> select(t.col1))
        assert_result_equal(df1_x, df1_y, lambda t: t >> select(t.col2))


class TestMutate:

    @tables(['df1'])
    def test_multiply(self, df1_x, df1_y):
        assert_result_equal(df1_x, df1_y, lambda t: t >> mutate(x2 = t.col1 * 2))
        assert_result_equal(df1_x, df1_y, lambda t: t >> select() >> mutate(x2 = t.col1 * 2))


class TestSummarise:

    @tables(['df3'])
    def test_ungrouped(self, df3_x, df3_y):
        assert_result_equal(df3_x, df3_y, lambda t: t >> summarise(mean3 = t.col3.mean()))
        assert_result_equal(df3_x, df3_y, lambda t: t >> summarise(mean3 = t.col3.mean(), mean4 = t.col4.mean()))

    @tables(['df3'])
    def test_simple_grouped(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1) >> summarise(mean3 = t.col3.mean())
        )

    @tables(['df3'])
    def test_multi_grouped(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1, t.col2) >> summarise(mean3 = t.col3.mean())
        )

    @tables(['df3'])
    def test_chained_summarised(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1, t.col2) >> summarise(mean3 = t.col3.mean()) >> summarise(mean_of_mean3 = λ.mean3.mean())
        )
