import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import sqlalchemy
from pandas.testing import assert_frame_equal

import pdtransform.core.dispatchers
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


def assert_result_equal(x, y, pipe_factory, exception = None, **kwargs):
    if not isinstance(x, (list, tuple)):
        x = (x,)
        y = (y,)

    if exception:
        with pytest.raises(exception):
            pipe_factory(*x) >> collect()
        with pytest.raises(exception):
            pipe_factory(*y) >> collect()
        return
    else:
        dfx = (pipe_factory(*x) >> collect()).reset_index(drop = True)
        dfy = (pipe_factory(*y) >> collect()).reset_index(drop = True)

    try:
        assert_frame_equal(dfx, dfy, **kwargs)
    except Exception as e:
        print('First dataframe:')
        print(dfx)
        print('Second dataframe:')
        print(dfy)
        raise e


@pdtransform.core.dispatchers.verb
def full_sort(t: Table):
    """
    Ordering after join is not determined.
    This helper applies a deterministic ordering to a table.
    """
    return t >> arrange(*t)


class TestSelect:

    @tables(['df1'])
    def test_simple_select(self, df1_x, df1_y):
        assert_result_equal(df1_x, df1_y, lambda t: t >> select(t.col1))
        assert_result_equal(df1_x, df1_y, lambda t: t >> select(t.col2))

    @tables(['df1'])
    def test_reorder(self, df1_x, df1_y):
        assert_result_equal(df1_x, df1_y, lambda t: t >> select(t.col2, t.col1))


class TestMutate:

    @tables(['df2'])
    def test_noop(self, df2_x, df2_y):
        assert_result_equal(df2_x, df2_y, lambda t: t >> mutate(col1 = t.col1, col2 = t.col2, col3 = t.col3))

    @tables(['df1'])
    def test_multiply(self, df1_x, df1_y):
        assert_result_equal(df1_x, df1_y, lambda t: t >> mutate(x2 = t.col1 * 2))
        assert_result_equal(df1_x, df1_y, lambda t: t >> select() >> mutate(x2 = t.col1 * 2))

    @tables(['df2'])
    def test_reorder(self, df2_x, df2_y):
        assert_result_equal(
            df2_x, df2_y,
            lambda t:
            t >> mutate(col1 = t.col2, col2 = t.col1)
        )

        assert_result_equal(
            df2_x, df2_y,
            lambda t:
            t >> mutate(col1 = t.col2, col2 = t.col1) >> mutate(col1 = t.col2, col2 = λ.col3, col3 = λ.col2)
        )


class TestJoin:

    # TODO: Implement test for outer join. This doesn't work with sqlite
    @tables(['df1', 'df2'])
    @pytest.mark.parametrize('how', ['inner', 'left'])
    def test_join(self, df1_x, df1_y, df2_x, df2_y, how):
        assert_result_equal(
            (df1_x, df2_x), (df1_y, df2_y),
            lambda t, u:
            t >> join(u, t.col1 == u.col1, how=how) >> full_sort()
        )

        assert_result_equal(
            (df1_x, df2_x), (df1_y, df2_y),
            lambda t, u:
            t >> join(u, (t.col1 == u.col1) & (t.col1 == u.col2), how=how) >> full_sort()
        )

    # TODO: Implement test for outer join. This doesn't work with sqlite
    @tables(['df1', 'df2'])
    @pytest.mark.parametrize('how', ['inner', 'left'])
    def test_join_and_select(self, df1_x, df1_y, df2_x, df2_y, how):
        assert_result_equal(
            (df1_x, df2_x), (df1_y, df2_y),
            lambda t, u:
            t >> select() >> join(u, t.col1 == u.col1, how=how) >> full_sort()
        )

        assert_result_equal(
            (df1_x, df2_x), (df1_y, df2_y),
            lambda t, u:
            t >> join(u >> select(), (t.col1 == u.col1) & (t.col1 == u.col2), how=how) >> full_sort()
        )

    # TODO: Implement test for outer join. This doesn't work with sqlite
    @tables(['df3'])
    @pytest.mark.parametrize('how', ['inner', 'left'])
    def test_self_join(self, df3_x, df3_y, how):
        # Self join without alias should raise an exception
        assert_result_equal(
            df3_x, df3_y,
            lambda t: t >> join(t, t.col1 == t.col1, how=how),
            exception = ValueError
        )

        def self_join_1(t):
            u = t >> alias('self_join')
            return t >> join(u, t.col1 == u.col1, how=how) >> full_sort()
        assert_result_equal(df3_x, df3_y, self_join_1)

        def self_join_2(t):
            u = t >> alias('self_join')
            return t >> join(u, (t.col1 == u.col1) & (t.col2 == u.col2), how = how) >> full_sort()
        assert_result_equal(df3_x, df3_y, self_join_2)

        def self_join_3(t):
            u = t >> alias('self_join')
            return t >> join(u, (t.col2 == u.col3), how = how) >> full_sort()
        assert_result_equal(df3_x, df3_y, self_join_3)


class TestFilter:

    @tables(['df2'])
    def test_noop(self, df2_x, df2_y):
        assert_result_equal(df2_x, df2_y, lambda t: t >> filter())
        assert_result_equal(df2_x, df2_y, lambda t: t >> filter(t.col1 == t.col1))

    @tables(['df2'])
    def test_simple_filter(self, df2_x, df2_y):
        assert_result_equal(df2_x, df2_y, lambda t: t >> filter(t.col1 == 2))
        assert_result_equal(df2_x, df2_y, lambda t: t >> filter(t.col1 != 2))

    @tables(['df2'])
    def test_chained_filters(self, df2_x, df2_y):
        assert_result_equal(
            df2_x, df2_y,
            lambda t:
            t >> filter(1 < t.col1) >> filter(t.col1 < 5)
        )

        assert_result_equal(
            df2_x, df2_y,
            lambda t:
            t >> filter(1 < t.col1) >> filter(t.col3 < 0.25)
        )

    @tables(['df3'])
    def test_filter_empty_result(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> filter(t.col1 == 0) >> filter(t.col2 == 2) >> filter(t.col4 < 2),

            check_dtype = False
        )


class TestArrange:

    @tables(['df1'])
    def test_noop(self, df1_x, df1_y):
        assert_result_equal(df1_x, df1_y, lambda t: t >> arrange())

    @tables(['df2'])
    def test_arrange(self, df2_x, df2_y):
        assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(t.col1))
        assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(-t.col1))
        assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(t.col3))
        assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(-t.col3))

    @tables(['df2'])
    def test_arrange_null(self, df2_x, df2_y):
        assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(t.col2))
        assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(-t.col2))

    @tables(['df3'])
    def test_multiple(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t: t >> arrange(t.col2, -t.col3, -t.col4)
        )

        assert_result_equal(
            df3_x, df3_y,
            lambda t: t >> arrange(t.col2) >> arrange(-t.col3, -t.col4)
        )


class TestGroupBy:

    @tables(['df3'])
    def test_ungroup(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1, t.col2) >> ungroup()
        )

    @tables(['df3'])
    def test_select(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1, t.col2) >> select(t.col1, t.col3)
        )

    @tables(['df3'])
    def test_mutate(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> mutate(c1xc2 = t.col1 * t.col2) >> group_by(λ.c1xc2)
        )

        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1, t.col2) >> mutate(c1xc2 = t.col1 * t.col2)
        )

        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1, t.col2) >> mutate(col1 = t.col1 * t.col2)
        )

    @tables(['df1', 'df3'])
    def test_grouped_join(self, df1_x, df1_y, df3_x, df3_y):
        # Joining a grouped table should always throw an exception
        assert_result_equal(
            (df1_x, df3_x), (df1_y, df3_y),
            lambda t, u: t >> group_by(λ.col1) >> join(u, t.col1 == u.col1, how = 'left'),
            exception = ValueError
        )

        assert_result_equal(
            (df1_x, df3_x), (df1_y, df3_y),
            lambda t, u: t >> join(u >> group_by(λ.col1), t.col1 == u.col1, how = 'left'),
            exception = ValueError
        )

    @tables(['df1', 'df3'])
    @pytest.mark.parametrize('how', ['inner', 'left'])
    def test_ungrouped_join(self, df1_x, df1_y, df3_x, df3_y, how):
        # After ungrouping joining should work again
        assert_result_equal(
            (df1_x, df3_x), (df1_y, df3_y),
            lambda t, u:
            t >> group_by(t.col1) >> ungroup() >> join(u, t.col1 == u.col1, how=how) >> full_sort()
        )

    @tables(['df3'])
    def test_filter(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1) >> filter(t.col3 >= 2)
        )

    @tables(['df3'])
    def test_arrange(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1) >> arrange(t.col1, -t.col3)
        )

        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1) >> arrange(-t.col4)
        )


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

    @tables(['df3'])
    def test_nested(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1, t.col2) >> summarise(mean_of_mean3 = t.col3.mean().mean()),
            exception = ValueError
        )


    @tables(['df3'])
    def test_select(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1, t.col2) >> summarise(mean3 = t.col3.mean()) >> select(t.col1, λ.mean3, t.col2)
        )

    @tables(['df3'])
    def test_mutate(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1, t.col2) >> summarise(mean3 = t.col3.mean()) >> mutate(x10 = λ.mean3 * 10)
        )

    @tables(['df3'])
    def test_filter(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1, t.col2) >> summarise(mean3 = t.col3.mean()) >> filter(λ.mean3 <= 2.0)
        )

    @tables(['df3'])
    def test_arrange(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1, t.col2) >> summarise(mean3 = t.col3.mean()) >> arrange(λ.mean3)
        )

        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> arrange(-t.col4) >> group_by(t.col1, t.col2) >> summarise(mean3 = t.col3.mean()) >> arrange(λ.mean3)
        )

    # TODO: Implement more test cases for summarise verb


class TestWindowFunction:

    @tables(['df3'])
    def test_simple_ungrouped(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> mutate(min = t.col4.min(), max = t.col4.max(), mean = t.col4.mean())
        )

    @tables(['df3'])
    def test_simple_grouped(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1) >> mutate(min = t.col4.min(), max = t.col4.max(), mean = t.col4.mean())
        )

        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1, t.col2) >> mutate(min = t.col4.min(), max = t.col4.max(), mean = t.col4.mean())
        )

    @tables(['df3'])
    def test_chained(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1) >> mutate(min = t.col4.min()) >> mutate(max = t.col4.max(), mean = t.col4.mean())
        )

        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1) >> mutate(min = t.col4.min(), max = t.col4.max()) >> mutate(span = λ.max - λ.min)
        )

    @tables(['df3'])
    def test_nested(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> group_by(t.col1) >> mutate(range = t.col4.max() - 10) >> ungroup() >> mutate(range_mean = λ.range.mean())
        )

        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> mutate(x = λ.col4.max()) >> mutate(y = λ.x.min() * 1) >> mutate(z = λ.y.mean()) >> mutate(w = λ.x / λ.y),
            exception = ValueError
        )

        assert_result_equal(
            df3_x, df3_y,
            lambda t:
            t >> mutate(x = (λ.col4.max().min() + λ.col2.mean()).max()),
            exception = ValueError
        )
