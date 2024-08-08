from __future__ import annotations

import functools
import os

import pandas as pd

from pydiverse.transform.core import Table
from pydiverse.transform.eager.pandas_table import fast_pd_convert_dtypes


def _cached_impl(fn):
    cache = {}

    @functools.wraps(fn)
    def wrapped(df: pd.DataFrame, name: str):
        if name in cache:
            return cache[name]

        impl = fn(df, name)
        cache[name] = impl

        return impl

    return wrapped


@_cached_impl
def pandas_impl(df: pd.DataFrame, name: str):
    from pydiverse.transform.eager import PandasTableImpl

    return PandasTableImpl(name, df)


_sql_engine_cache = {}


def _sql_table(
    df: pd.DataFrame, name: str, url: str, dtypes_map: dict = None, schema=None
):
    import sqlalchemy as sa

    from pydiverse.transform.lazy import SQLTableImpl

    global _sql_engine_cache

    dtypes_map = dtypes_map or {}

    if url in _sql_engine_cache:
        engine = _sql_engine_cache[url]
    else:
        engine = sa.create_engine(url)
        _sql_engine_cache[url] = engine

    df = fast_pd_convert_dtypes(df)
    sql_dtypes = {}
    for col, dtype_ in df.dtypes.items():
        if dtype_ in dtypes_map:
            sql_dtypes[col] = dtypes_map[dtype_]

    df.to_sql(
        name, engine, schema=schema, index=False, if_exists="replace", dtype=sql_dtypes
    )
    return SQLTableImpl(engine, name if schema is None else f"{schema}.{name}")


@_cached_impl
def sqlite_impl(df: pd.DataFrame, name: str):
    return _sql_table(df, name, "sqlite:///:memory:")


@_cached_impl
def duckdb_impl(df: pd.DataFrame, name: str):
    return _sql_table(df, name, "duckdb:///:memory:")


@_cached_impl
def postgres_impl(df: pd.DataFrame, name: str):
    url = "postgresql://sa:Pydiverse23@127.0.0.1:6543"
    return _sql_table(df, name, url)


@_cached_impl
def mssql_impl(df: pd.DataFrame, name: str):
    import numpy as np
    from sqlalchemy.dialects.mssql import DATETIME2

    url = (
        "mssql+pyodbc://sa:PydiQuant27@127.0.0.1:1433"
        "/master?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
    )
    return _sql_table(
        df,
        name,
        url,
        dtypes_map={
            np.dtype("datetime64[ns]"): DATETIME2(),
        },
    )


@_cached_impl
def ibm_db2_impl(df: pd.DataFrame, name: str):
    url = "db2+ibm_db://db2inst1:password@localhost:50000/testdb"
    return _sql_table(df, name, url)


@_cached_impl
def snowflake_impl(df: pd.DataFrame, name: str):
    user = os.environ["SNOWFLAKE_USER"]
    password = os.environ["SNOWFLAKE_PASSWORD"]
    account = os.environ["SNOWFLAKE_ACCOUNT"]
    url = (
        f"snowflake://{user}:{password}@{account}/pipedag/DBO?"
        "warehouse=pipedag&role=accountadmin"
    )
    return _sql_table(df, name, url, schema="public")


def impl_to_table_callable(fn):
    @functools.wraps(fn)
    def wrapped(df: pd.DataFrame, name: str):
        impl = fn(df, name)
        return Table(impl)

    return wrapped


BACKEND_TABLES = {
    "pandas": impl_to_table_callable(pandas_impl),
    "sqlite": impl_to_table_callable(sqlite_impl),
    "duckdb": impl_to_table_callable(duckdb_impl),
    "postgres": impl_to_table_callable(postgres_impl),
    "mssql": impl_to_table_callable(mssql_impl),
    "ibm_db2": impl_to_table_callable(ibm_db2_impl),
    "snowflake": impl_to_table_callable(snowflake_impl),
}
