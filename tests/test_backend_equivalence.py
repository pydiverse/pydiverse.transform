from __future__ import annotations

import itertools
import sqlite3
from collections import defaultdict

import pandas as pd
import pytest
import sqlalchemy

# from google.oauth2 import service_account
from pandas.testing import assert_frame_equal

import pydiverse.transform.core.dispatchers
from pydiverse.transform import λ
from pydiverse.transform.core import functions as f
from pydiverse.transform.core.table import Table
from pydiverse.transform.core.verbs import (
    alias,
    arrange,
    collect,
    filter,
    group_by,
    join,
    left_join,
    mutate,
    rename,
    select,
    show_query,
    slice_head,
    summarise,
    ungroup,
)
from pydiverse.transform.eager.pandas_table import PandasTableImpl
from pydiverse.transform.lazy.sql_table import SQLTableImpl

dataframes = {
    "df1": pd.DataFrame(
        {
            "col1": [1, 2, 3, 4],
            "col2": ["a", "b", "c", "d"],
        }
    ),
    "df2": pd.DataFrame(
        {
            "col1": [1, 2, 2, 4, 5, 6],
            "col2": [2, 2, 0, 0, 2, None],
            "col3": [0.0, 0.1, 0.2, 0.3, 0.01, 0.02],
        }
    ),
    "df3": pd.DataFrame(
        {
            "col1": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "col2": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "col3": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "col4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "col5": list("abcdefghijkl"),
        }
    ),
    "df_left": pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
        }
    ),
    "df_right": pd.DataFrame(
        {
            "b": [0, 1, 2, 2],
            "c": [5, 6, 7, 8],
        }
    ),
}


def pandas_impls():
    return {name: PandasTableImpl(name, df) for name, df in dataframes.items()}


def sql_conn_to_impls(conn: str, project_id=None, dataset=None):
    engine = sqlalchemy.create_engine(conn)
    impls = {}
    print("THIS IS VERY EXPENSIVE")
    for name, df in dataframes.items():
        if engine.dialect.name == "bigquery":
            if not (project_id and dataset):
                raise ValueError(
                    "Project Id and dataset names are required for bigquery"
                )
            # name = f"{dataset}.{name}"
            impls[name] = SQLTableImpl(engine, f"{dataset}.{name}")
        else:
            df.to_sql(name, engine, index=False, if_exists="replace")
            impls[name] = SQLTableImpl(engine, name)
    return impls


def sqlite_impls():
    return sql_conn_to_impls("sqlite:///:memory:")


def mssql_impls():
    user = "sa"
    password = "PidyQuant27"
    localhost = "127.0.0.1"
    db_name = "master"
    local_conn = f"mssql+pyodbc://{user}:{password}@{localhost}:1433/{db_name}?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
    return sql_conn_to_impls(local_conn)


def postgresql_impls():
    user = "sa"
    password = "Pydiverse23"
    local_conn = f"postgresql://{user}:{password}@localhost:5432/"
    return sql_conn_to_impls(local_conn)


backend_impls = {
    "pandas": pandas_impls(),
    "sqlite": sqlite_impls(),
    # "mssql": mssql_impls(),
    "postgres": postgresql_impls(),
}


def tables(names: list[str]):
    param_names = ",".join([f"{name}_x,{name}_y" for name in names])

    tables = defaultdict(lambda: [])
    backend_names = backend_impls.keys()
    for _, impls in backend_impls.items():
        for table_name, impl in impls.items():
            tables[table_name].append(Table(impl))

    param_combinations = (
        (zip(*itertools.combinations(tables[name], 2))) for name in names
    )
    param_combinations = itertools.chain(*param_combinations)
    param_combinations = list(zip(*param_combinations))

    names_combinations = list(itertools.combinations(backend_names, 2))

    params = [
        pytest.param(*p, id=f"{id[0]} {id[1]}")
        for p, id in zip(param_combinations, names_combinations)
    ]

    return pytest.mark.parametrize(param_names, params)


# TODO: when should we still consider the order when comparing?
def assert_result_equal(
    x, y, pipe_factory, *, exception=None, check_order=False, may_throw=False, **kwargs
):
    if not isinstance(x, (list, tuple)):
        x = (x,)
        y = (y,)

    if exception and not may_throw:
        with pytest.raises(exception):
            pipe_factory(*x) >> collect()
        with pytest.raises(exception):
            pipe_factory(*y) >> collect()
        return
    else:
        try:
            query_x = pipe_factory(*x)
            query_y = pipe_factory(*y)
            dfx = (query_x >> collect()).reset_index(drop=True)
            dfy = (query_y >> collect()).reset_index(drop=True)

            if not check_order:
                dfx.sort_values(
                    by=dfx.columns.tolist(), inplace=True, ignore_index=True
                )
                dfy.sort_values(
                    by=dfy.columns.tolist(), inplace=True, ignore_index=True
                )
        except Exception as e:
            if may_throw:
                if exception is not None:
                    if isinstance(exception, type):
                        exception = (exception,)
                    if not isinstance(e, exception):
                        raise Exception(
                            f"Raised the wrong type of exception: {type(e)} instead of"
                            f" {exception}."
                        ) from e
                # TODO: Replace with logger
                print(f"An exception was thrown:\n{e}")
                return
            else:
                raise e

    try:
        assert_frame_equal(dfx, dfy, check_dtype=False, **kwargs)
    except Exception as e:
        print("First dataframe:")
        print(dfx)
        query_x >> show_query()
        print("")
        print("Second dataframe:")
        print(dfy)
        query_y >> show_query()
        print("")
        raise e


@pydiverse.transform.core.dispatchers.verb
def full_sort(t: Table):
    """
    Ordering after join is not determined.
    This helper applies a deterministic ordering to a table.
    """
    return t >> arrange(*t)


class TestSyntax:
    @tables(["df3"])
    def test_lambda_cols(self, df3_x, df3_y):
        assert_result_equal(df3_x, df3_y, lambda t: t >> select(λ.col1, λ.col2))
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> mutate(col1=λ.col1, col2=λ.col1)
        )

        assert_result_equal(
            df3_x, df3_y, lambda t: t >> select(λ.col10), exception=ValueError
        )

    @tables(["df3"])
    def test_columns_pipeable(self, df3_x, df3_y):
        assert_result_equal(df3_x, df3_y, lambda t: t.col1 >> mutate(x=t.col1))

        # Test invalid operations
        assert_result_equal(
            df3_x, df3_y, lambda t: t.col1 >> mutate(x=t.col2), exception=ValueError
        )

        assert_result_equal(
            df3_x, df3_y, lambda t: t.col1 >> mutate(x=λ.col2), exception=ValueError
        )

        assert_result_equal(
            df3_x, df3_y, lambda t: (t.col1 + 1) >> select(), exception=ValueError
        )


class TestSelect:
    @tables(["df1"])
    def test_simple_select(self, df1_x, df1_y):
        assert_result_equal(df1_x, df1_y, lambda t: t >> select(t.col1))
        assert_result_equal(df1_x, df1_y, lambda t: t >> select(t.col2))

    @tables(["df1"])
    def test_reorder(self, df1_x, df1_y):
        assert_result_equal(df1_x, df1_y, lambda t: t >> select(t.col2, t.col1))

    @tables(["df3"])
    def test_ellipsis(self, df3_x, df3_y):
        assert_result_equal(df3_x, df3_y, lambda t: t >> select(...))
        assert_result_equal(df3_x, df3_y, lambda t: t >> select(t.col1) >> select(...))
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> mutate(x=t.col1 * 2) >> select() >> select(...)
        )

    @tables(["df3"])
    def test_negative_select(self, df3_x, df3_y):
        assert_result_equal(df3_x, df3_y, lambda t: t >> select(-t.col1))
        assert_result_equal(df3_x, df3_y, lambda t: t >> select(-λ.col1, -t.col2))
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> select() >> mutate(x=t.col1 * 2) >> select(-λ.col3),
        )


class TestRename:
    @tables(["df3"])
    def test_noop(self, df3_x, df3_y):
        assert_result_equal(df3_x, df3_y, lambda t: t >> rename({}))
        assert_result_equal(df3_x, df3_y, lambda t: t >> rename({"col1": "col1"}))

    @tables(["df3"])
    def test_simple(self, df3_x, df3_y):
        assert_result_equal(df3_x, df3_y, lambda t: t >> rename({"col1": "X"}))
        assert_result_equal(df3_x, df3_y, lambda t: t >> rename({"col2": "Y"}))
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> rename({"col1": "A", "col2": "B"})
        )
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> rename({"col2": "B", "col1": "A"})
        )

    @tables(["df3"])
    def test_chained(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> rename({"col1": "X"}) >> rename({"X": "Y"})
        )
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> rename({"col1": "X"}) >> rename({"X": "col1"})
        )
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> rename({"col1": "1", "col2": "2"})
            >> rename({"1": "col1", "2": "col2"}),
        )

    @tables(["df3"])
    def test_complex(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> rename({"col1": "col2", "col2": "col3", "col3": "col1"}),
        )


class TestMutate:
    @tables(["df2"])
    def test_noop(self, df2_x, df2_y):
        assert_result_equal(
            df2_x, df2_y, lambda t: t >> mutate(col1=t.col1, col2=t.col2, col3=t.col3)
        )

    @tables(["df1"])
    def test_multiply(self, df1_x, df1_y):
        assert_result_equal(df1_x, df1_y, lambda t: t >> mutate(x2=t.col1 * 2))
        assert_result_equal(
            df1_x, df1_y, lambda t: t >> select() >> mutate(x2=t.col1 * 2)
        )

    @tables(["df2"])
    def test_reorder(self, df2_x, df2_y):
        assert_result_equal(
            df2_x, df2_y, lambda t: t >> mutate(col1=t.col2, col2=t.col1)
        )

        assert_result_equal(
            df2_x,
            df2_y,
            lambda t: t
            >> mutate(col1=t.col2, col2=t.col1)
            >> mutate(col1=t.col2, col2=λ.col3, col3=λ.col2),
        )

    @tables(["df1"])
    def test_literals(self, df1_x, df1_y):
        assert_result_equal(df1_x, df1_y, lambda t: t >> mutate(x=1))
        assert_result_equal(df1_x, df1_y, lambda t: t >> mutate(x=1.1))
        assert_result_equal(df1_x, df1_y, lambda t: t >> mutate(x=True))
        assert_result_equal(df1_x, df1_y, lambda t: t >> mutate(x="test"))


class TestJoin:
    @tables(["df1", "df2"])
    @pytest.mark.parametrize(
        "how",
        [
            "inner",
            "left",
            pytest.param(
                "outer",
                marks=pytest.mark.skipif(
                    sqlite3.sqlite_version < "3.39.0",
                    reason="SQLite version doesn't support OUTER JOIN",
                ),
            ),
        ],
    )
    def test_join(self, df1_x, df1_y, df2_x, df2_y, how):
        assert_result_equal(
            (df1_x, df2_x),
            (df1_y, df2_y),
            lambda t, u: t >> join(u, t.col1 == u.col1, how=how) >> full_sort(),
        )

        assert_result_equal(
            (df1_x, df2_x),
            (df1_y, df2_y),
            lambda t, u: t
            >> join(u, (t.col1 == u.col1) & (t.col1 == u.col2), how=how)
            >> full_sort(),
        )

    @tables(["df1", "df2"])
    @pytest.mark.parametrize(
        "how",
        [
            "inner",
            "left",
            pytest.param(
                "outer",
                marks=pytest.mark.skipif(
                    sqlite3.sqlite_version < "3.39.0",
                    reason="SQLite version doesn't support OUTER JOIN",
                ),
            ),
        ],
    )
    def test_join_and_select(self, df1_x, df1_y, df2_x, df2_y, how):
        assert_result_equal(
            (df1_x, df2_x),
            (df1_y, df2_y),
            lambda t, u: t
            >> select()
            >> join(u, t.col1 == u.col1, how=how)
            >> full_sort(),
        )

        assert_result_equal(
            (df1_x, df2_x),
            (df1_y, df2_y),
            lambda t, u: t
            >> join(u >> select(), (t.col1 == u.col1) & (t.col1 == u.col2), how=how)
            >> full_sort(),
        )

    @tables(["df3"])
    @pytest.mark.parametrize(
        "how",
        [
            "inner",
            "left",
            pytest.param(
                "outer",
                marks=pytest.mark.skipif(
                    sqlite3.sqlite_version < "3.39.0",
                    reason="SQLite version doesn't support OUTER JOIN",
                ),
            ),
        ],
    )
    def test_self_join(self, df3_x, df3_y, how):
        # Self join without alias should raise an exception
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> join(t, t.col1 == t.col1, how=how),
            exception=ValueError,
        )

        def self_join_1(t):
            u = t >> alias("self_join")
            return t >> join(u, t.col1 == u.col1, how=how) >> full_sort()

        assert_result_equal(df3_x, df3_y, self_join_1)

        def self_join_2(t):
            u = t >> alias("self_join")
            return (
                t
                >> join(u, (t.col1 == u.col1) & (t.col2 == u.col2), how=how)
                >> full_sort()
            )

        assert_result_equal(df3_x, df3_y, self_join_2)

        def self_join_3(t):
            u = t >> alias("self_join")
            return t >> join(u, (t.col2 == u.col3), how=how) >> full_sort()

        assert_result_equal(df3_x, df3_y, self_join_3)


class TestFilter:
    @tables(["df2"])
    def test_noop(self, df2_x, df2_y):
        assert_result_equal(df2_x, df2_y, lambda t: t >> filter())
        assert_result_equal(df2_x, df2_y, lambda t: t >> filter(t.col1 == t.col1))

    @tables(["df2"])
    def test_simple_filter(self, df2_x, df2_y):
        assert_result_equal(df2_x, df2_y, lambda t: t >> filter(t.col1 == 2))
        assert_result_equal(df2_x, df2_y, lambda t: t >> filter(t.col1 != 2))

    @tables(["df2"])
    def test_chained_filters(self, df2_x, df2_y):
        assert_result_equal(
            df2_x, df2_y, lambda t: t >> filter(1 < t.col1) >> filter(t.col1 < 5)
        )

        assert_result_equal(
            df2_x, df2_y, lambda t: t >> filter(1 < t.col1) >> filter(t.col3 < 0.25)
        )

    @tables(["df3"])
    def test_filter_empty_result(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> filter(t.col1 == 0)
            >> filter(t.col2 == 2)
            >> filter(t.col4 < 2),
        )


class TestArrange:
    @tables(["df1"])
    def test_noop(self, df1_x, df1_y):
        assert_result_equal(df1_x, df1_y, lambda t: t >> arrange())

    @tables(["df2"])
    def test_arrange(self, df2_x, df2_y):
        assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(t.col1))
        assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(-t.col1))
        assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(t.col3))
        assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(-t.col3))

    @tables(["df2"])
    def test_arrange_null(self, df2_x, df2_y):
        assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(t.col2))
        assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(-t.col2))

    @tables(["df3"])
    def test_multiple(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> arrange(t.col2, -t.col3, -t.col4)
        )

        assert_result_equal(
            df3_x, df3_y, lambda t: t >> arrange(t.col2) >> arrange(-t.col3, -t.col4)
        )


class TestGroupBy:
    @tables(["df3"])
    def test_ungroup(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> group_by(t.col1, t.col2) >> ungroup()
        )

    @tables(["df3"])
    def test_select(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> group_by(t.col1, t.col2) >> select(t.col1, t.col3),
        )

    @tables(["df3"])
    def test_mutate(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> mutate(c1xc2=t.col1 * t.col2) >> group_by(λ.c1xc2),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> group_by(t.col1, t.col2) >> mutate(c1xc2=t.col1 * t.col2),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> group_by(t.col1, t.col2) >> mutate(col1=t.col1 * t.col2),
        )

    @tables(["df1", "df3"])
    def test_grouped_join(self, df1_x, df1_y, df3_x, df3_y):
        # Joining a grouped table should always throw an exception
        assert_result_equal(
            (df1_x, df3_x),
            (df1_y, df3_y),
            lambda t, u: t >> group_by(λ.col1) >> join(u, t.col1 == u.col1, how="left"),
            exception=ValueError,
        )

        assert_result_equal(
            (df1_x, df3_x),
            (df1_y, df3_y),
            lambda t, u: t >> join(u >> group_by(λ.col1), t.col1 == u.col1, how="left"),
            exception=ValueError,
        )

    @tables(["df1", "df3"])
    @pytest.mark.parametrize("how", ["inner", "left"])
    def test_ungrouped_join(self, df1_x, df1_y, df3_x, df3_y, how):
        # After ungrouping joining should work again
        assert_result_equal(
            (df1_x, df3_x),
            (df1_y, df3_y),
            lambda t, u: t
            >> group_by(t.col1)
            >> ungroup()
            >> join(u, t.col1 == u.col1, how=how)
            >> full_sort(),
        )

    @tables(["df3"])
    def test_filter(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> group_by(t.col1) >> filter(t.col3 >= 2)
        )

    @tables(["df3"])
    def test_arrange(self, df3_x, df3_y):
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> group_by(t.col1) >> arrange(t.col1, -t.col3)
        )

        assert_result_equal(
            df3_x, df3_y, lambda t: t >> group_by(t.col1) >> arrange(-t.col4)
        )


class TestSummarise:
    @tables(["df3"])
    def test_ungrouped(self, df3_x, df3_y):
        assert_result_equal(df3_x, df3_y, lambda t: t >> summarise(mean3=t.col3.mean()))
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> summarise(mean3=t.col3.mean(), mean4=t.col4.mean()),
        )

    @tables(["df3"])
    def test_simple_grouped(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> group_by(t.col1) >> summarise(mean3=t.col3.mean()),
        )

    @tables(["df3"])
    def test_multi_grouped(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> group_by(t.col1, t.col2) >> summarise(mean3=t.col3.mean()),
        )

    @tables(["df3"])
    def test_chained_summarised(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> summarise(mean3=t.col3.mean())
            >> summarise(mean_of_mean3=λ.mean3.mean()),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> mutate(k=(λ.col1 + λ.col2) * λ.col4)
            >> group_by(λ.k)
            >> summarise(x=λ.col4.mean())
            >> summarise(y=λ.k.mean()),
        )

    @tables(["df3"])
    def test_nested(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> summarise(mean_of_mean3=t.col3.mean().mean()),
            exception=ValueError,
        )

    @tables(["df3"])
    def test_select(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> summarise(mean3=t.col3.mean())
            >> select(t.col1, λ.mean3, t.col2),
        )

    @tables(["df3"])
    def test_mutate(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> summarise(mean3=t.col3.mean())
            >> mutate(x10=λ.mean3 * 10),
        )

    @tables(["df3"])
    def test_filter(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> summarise(mean3=t.col3.mean())
            >> filter(λ.mean3 <= 2.0),
        )

    @tables(["df3"])
    def test_arrange(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> summarise(mean3=t.col3.mean())
            >> arrange(λ.mean3),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> arrange(-t.col4)
            >> group_by(t.col1, t.col2)
            >> summarise(mean3=t.col3.mean())
            >> arrange(λ.mean3),
        )

    @tables(["df3"])
    def test_intermediate_select(self, df3_x, df3_y):
        # Check that subqueries happen transparently
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> summarise(x=t.col4.mean())
            >> mutate(x2=λ.x * 2)
            >> select()
            >> summarise(y=(λ.x - λ.x2).min()),
        )

    # TODO: Implement more test cases for summarise verb


class TestWindowFunction:
    @tables(["df3"])
    def test_simple_ungrouped(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> mutate(min=t.col4.min(), max=t.col4.max(), mean=t.col4.mean()),
        )

    @tables(["df3"])
    def test_simple_grouped(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1)
            >> mutate(min=t.col4.min(), max=t.col4.max(), mean=t.col4.mean()),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> mutate(min=t.col4.min(), max=t.col4.max(), mean=t.col4.mean()),
        )

    @tables(["df3"])
    def test_chained(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1)
            >> mutate(min=t.col4.min())
            >> mutate(max=t.col4.max(), mean=t.col4.mean()),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1)
            >> mutate(min=t.col4.min(), max=t.col4.max())
            >> mutate(span=λ.max - λ.min),
        )

    @tables(["df3"])
    def test_nested(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1)
            >> mutate(range=t.col4.max() - 10)
            >> ungroup()
            >> mutate(range_mean=λ.range.mean()),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> mutate(x=λ.col4.max())
            >> mutate(y=λ.x.min() * 1)
            >> mutate(z=λ.y.mean())
            >> mutate(w=λ.x / λ.y),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> mutate(x=(λ.col4.max().min() + λ.col2.mean()).max()),
            exception=ValueError,
            may_throw=True,
        )

    @tables(["df3"])
    def test_filter(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> mutate(mean3=t.col3.mean())
            >> filter(λ.mean3 <= 2.0),
        )

    @tables(["df3"])
    def test_arrange(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> mutate(mean3=t.col3.mean())
            >> arrange(λ.mean3),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> arrange(-t.col4)
            >> group_by(t.col1, t.col2)
            >> mutate(mean3=t.col3.mean())
            >> arrange(λ.mean3),
        )

    @tables(["df3"])
    def test_summarise(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> mutate(range=t.col4.max() - t.col4.min())
            >> summarise(mean_range=λ.range.mean()),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> summarise(range=t.col4.max() - t.col4.min())
            >> mutate(mean_range=λ.range.mean()),
        )

    @tables(["df3"])
    def test_intermediate_select(self, df3_x, df3_y):
        # Check that subqueries happen transparently
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> mutate(x=t.col4.mean())
            >> select()
            >> mutate(y=λ.x.min())
            >> select()
            >> mutate(z=(λ.x - λ.y).mean()),
        )

    @tables(["df3"])
    def test_arrange_argument(self, df3_x, df3_y):
        # Grouped
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1)
            >> mutate(x=λ.col4.shift(1, arrange=[-λ.col3]))
            >> full_sort()
            >> select(λ.x),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col2)
            >> mutate(x=f.row_number(arrange=[-λ.col4]))
            >> full_sort()
            >> select(λ.x),
        )

        # Ungrouped
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> mutate(x=λ.col4.shift(1, arrange=[-λ.col3]))
            >> full_sort()
            >> select(λ.x),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> mutate(x=f.row_number(arrange=[-λ.col4]))
            >> full_sort()
            >> select(λ.x),
        )

    @tables(["df3"])
    def test_complex(self, df3_x, df3_y):
        # Window function before summarise
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> mutate(mean3=t.col3.mean(), rn=f.row_number(arrange=[λ.col1, λ.col2]))
            >> filter(λ.mean3 > λ.rn)
            >> summarise(meta_mean=λ.mean3.mean())
            >> filter(t.col1 >= λ.meta_mean)
            >> filter(t.col1 != 1)
            >> arrange(λ.meta_mean),
        )

        # Window function after summarise
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> group_by(t.col1, t.col2)
            >> summarise(mean3=t.col3.mean())
            >> mutate(minM3=λ.mean3.min(), maxM3=λ.mean3.max())
            >> mutate(span=λ.maxM3 - λ.minM3)
            >> filter(λ.span < 3)
            >> arrange(λ.span),
        )


class TestSliceHead:
    @tables(["df3"])
    def test_simple(self, df3_x, df3_y):
        assert_result_equal(df3_x, df3_y, lambda t: t >> arrange(*t) >> slice_head(1))
        assert_result_equal(df3_x, df3_y, lambda t: t >> arrange(*t) >> slice_head(10))
        assert_result_equal(df3_x, df3_y, lambda t: t >> arrange(*t) >> slice_head(100))

        assert_result_equal(
            df3_x, df3_y, lambda t: t >> arrange(*t) >> slice_head(1, offset=8)
        )
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> arrange(*t) >> slice_head(10, offset=8)
        )
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> arrange(*t) >> slice_head(100, offset=8)
        )

        assert_result_equal(
            df3_x, df3_y, lambda t: t >> arrange(*t) >> slice_head(1, offset=100)
        )
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> arrange(*t) >> slice_head(10, offset=100)
        )
        assert_result_equal(
            df3_x, df3_y, lambda t: t >> arrange(*t) >> slice_head(100, offset=100)
        )

    @tables(["df3"])
    def test_chained(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> arrange(*t) >> slice_head(1) >> arrange(*t) >> slice_head(1),
        )
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> arrange(*t)
            >> slice_head(10)
            >> arrange(*t)
            >> slice_head(5),
        )
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> arrange(*t)
            >> slice_head(100)
            >> arrange(*t)
            >> slice_head(5),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> arrange(*t)
            >> slice_head(2, offset=5)
            >> arrange(*t)
            >> slice_head(2, offset=1),
        )
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> arrange(*t)
            >> slice_head(10, offset=8)
            >> arrange(*t)
            >> slice_head(10, offset=1),
        )

    @tables(["df3"])
    def test_with_select(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> select()
            >> arrange(*t)
            >> slice_head(4, offset=2)
            >> select(*t),
        )

    @tables(["df3"])
    def test_with_mutate(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> mutate(a=λ.col1 * 2)
            >> arrange(*t)
            >> slice_head(4, offset=2)
            >> mutate(b=λ.col2 + λ.a),
        )

    @tables(["df1", "df2"])
    def test_with_join(self, df1_x, df1_y, df2_x, df2_y):
        assert_result_equal(
            (df1_x, df2_x),
            (df1_y, df2_y),
            lambda t, u: t
            >> full_sort()
            >> arrange(*t)
            >> slice_head(3)
            >> left_join(u, t.col1 == u.col1)
            >> full_sort(),
        )

        assert_result_equal(
            (df1_x, df2_x),
            (df1_y, df2_y),
            lambda t, u: t
            >> left_join(u >> arrange(*t) >> slice_head(2, offset=1), t.col1 == u.col1)
            >> full_sort(),
            exception=ValueError,
            may_throw=True,
        )

    @tables(["df3"])
    def test_with_filter(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> filter(t.col4 % 2 == 0)
            >> arrange(*t)
            >> slice_head(4, offset=2),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> arrange(*t)
            >> slice_head(4, offset=2)
            >> filter(t.col1 == 1),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> filter(t.col4 % 2 == 0)
            >> arrange(*t)
            >> slice_head(4, offset=2)
            >> filter(t.col1 == 1),
        )

    @tables(["df3"])
    def test_with_arrange(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> mutate(x=t.col4 - (t.col1 * t.col2))
            >> arrange(λ.x, *t)
            >> slice_head(4, offset=2),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> mutate(x=(t.col1 * t.col2))
            >> arrange(*t)
            >> slice_head(4)
            >> arrange(-λ.x, λ.col5),
        )

    @tables(["df3"])
    def test_with_group_by(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> arrange(*t)
            >> slice_head(1)
            >> group_by(λ.col1)
            >> mutate(x=f.count()),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> arrange(λ.col1, *t)
            >> slice_head(6, offset=1)
            >> group_by(λ.col1)
            >> select()
            >> mutate(x=λ.col4.mean()),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> mutate(key=λ.col4 % (λ.col3 + 1))
            >> arrange(λ.key, *t)
            >> slice_head(4)
            >> group_by(λ.key)
            >> summarise(x=f.count()),
        )

    @tables(["df3"])
    def test_with_summarise(self, df3_x, df3_y):
        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t >> arrange(*t) >> slice_head(4) >> summarise(count=f.count()),
        )

        assert_result_equal(
            df3_x,
            df3_y,
            lambda t: t
            >> arrange(*t)
            >> slice_head(4)
            >> summarise(c3_mean=λ.col3.mean()),
        )
