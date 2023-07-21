from __future__ import annotations

import functools

import pandas as pd
import pytest
import sqlalchemy as sa

from pydiverse.transform import Table
from pydiverse.transform.eager import PandasTableImpl
from pydiverse.transform.lazy import SQLTableImpl
from tests.fixtures.backend import flatten

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
    "df4": pd.DataFrame(
        {
            "col1": [None, 0, 0, 0, 0, None, 1, 1, 1, 2, 2, 2, 2],
            "col2": [0, 0, 1, 1, 0, 0, 1, None, 1, 0, 0, 1, 1],
            "col3": [None, None, None, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "col4": [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "col5": list("abcdefghijkl") + [None],
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


@functools.cache
def pandas_impls():
    return {name: PandasTableImpl(name, df) for name, df in dataframes.items()}


def sql_conn_to_impls(conn: str):
    engine = sa.create_engine(conn)
    impls = {}
    for name, df in dataframes.items():
        df.to_sql(name, engine, index=False, if_exists="replace")
        impls[name] = SQLTableImpl(engine, name)
    return impls


@functools.cache
def sqlite_impls():
    return sql_conn_to_impls("sqlite:///:memory:")


@functools.cache
def duckdb_impls():
    return sql_conn_to_impls("duckdb:///:memory:")


@functools.cache
def mssql_impls():
    user = "sa"
    password = "PydiQuant27"
    localhost = "127.0.0.1"
    db_name = "master"
    local_conn = f"mssql+pyodbc://{user}:{password}@{localhost}:1433/{db_name}?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
    return sql_conn_to_impls(local_conn)


@functools.cache
def postgresql_impls():
    user = "sa"
    password = "Pydiverse23"
    local_conn = f"postgresql://{user}:{password}@localhost:6543/"
    return sql_conn_to_impls(local_conn)


backend_impls = {
    "pandas": pandas_impls,
    "sqlite": sqlite_impls,
    "duckdb": duckdb_impls,
    "postgres": postgresql_impls,
    "mssql": mssql_impls,
}

# compare one dataframe and one SQL backend to all others
# (some tests are ignored if either backend does not support a feature)
reference_backends = ["pandas", "duckdb"]


def pytest_generate_tests(metafunc: pytest.Metafunc):
    # Parametrize tests based on `backends` and `skip_backends` mark.

    from tests.fixtures.backend import BACKEND_MARKS

    backends = dict.fromkeys(backend_impls)
    # if mark := metafunc.definition.get_closest_marker("backends"):
    #     backends = dict.fromkeys(mark.args)
    if mark := metafunc.definition.get_closest_marker("skip_backends"):
        for backend in mark.args:
            if backend in backends:
                del backends[backend]

    backends = {k: i for i, k in enumerate(backends)}

    backend_combinations = [
        (reference_backend, backend)
        for reference_backend in reference_backends
        for backend in backends
    ]

    params = []
    table_names = metafunc.definition.get_closest_marker("request_tables").args[0]
    for reference_backend, backend in backend_combinations:
        # Skip some redundant backend combinations
        if reference_backend == backend:
            continue
        if reference_backend not in backends:
            continue
        if backend in reference_backends:
            if backends[backend] < backends[reference_backend]:
                continue

        marks = list(
            flatten(
                [
                    BACKEND_MARKS.get(reference_backend, ()),
                    BACKEND_MARKS.get(backend, ()),
                ]
            )
        )

        arguments = [reference_backend, backend] * len(table_names)

        params.append(
            pytest.param(
                *arguments,
                marks=marks,
                id=f"{reference_backend}-{backend}",
            )
        )

    param_names = ",".join([f"{name}_x, {name}_y" for name in table_names])
    return metafunc.parametrize(param_names, params, indirect=True)


def generate_df_fixtures():
    def gen_fixture(table_name):
        @pytest.fixture(scope="function")
        def table_fixture(request):
            table = Table(backend_impls[request.param]()[table_name])
            return table

        return table_fixture

    for name in dataframes:
        globals()[f"{name}_x"] = gen_fixture(name)
        globals()[f"{name}_y"] = gen_fixture(name)


generate_df_fixtures()
