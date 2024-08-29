from __future__ import annotations

import functools

import polars as pl

from pydiverse.transform.core import Table
from pydiverse.transform.polars.polars_table import PolarsEager
from pydiverse.transform.sql.sql_table import SQLTableImpl


def _cached_impl(fn):
    cache = {}

    @functools.wraps(fn)
    def wrapped(df: pl.DataFrame, name: str):
        if name in cache:
            return cache[name]

        impl = fn(df, name)
        cache[name] = impl

        return impl

    return wrapped


@_cached_impl
def polars_impl(df: pl.DataFrame, name: str):
    return PolarsEager(name, df)


_sql_engine_cache = {}


def _sql_table(df: pl.DataFrame, name: str, url: str, dtypes_map: dict = None):
    import sqlalchemy as sa

    global _sql_engine_cache

    dtypes_map = dtypes_map or {}

    if url in _sql_engine_cache:
        engine = _sql_engine_cache[url]
    else:
        engine = sa.create_engine(url)
        _sql_engine_cache[url] = engine

    sql_dtypes = {}
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype in dtypes_map:
            sql_dtypes[col] = dtypes_map[dtype]

    df.write_database(
        name, engine, if_table_exists="replace", engine_options={"dtype": sql_dtypes}
    )
    return SQLTableImpl(engine, name)


@_cached_impl
def sqlite_impl(df: pl.DataFrame, name: str):
    return _sql_table(df, name, "sqlite:///:memory:")


@_cached_impl
def duckdb_impl(df: pl.DataFrame, name: str):
    return _sql_table(df, name, "duckdb:///:memory:")


@_cached_impl
def postgres_impl(df: pl.DataFrame, name: str):
    url = "postgresql://sa:Pydiverse23@127.0.0.1:6543"
    return _sql_table(df, name, url)


@_cached_impl
def mssql_impl(df: pl.DataFrame, name: str):
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
            pl.Datetime(): DATETIME2(),
        },
    )


def impl_to_table_callable(fn):
    @functools.wraps(fn)
    def wrapped(df: pl.DataFrame, name: str):
        impl = fn(df, name)
        return Table(impl)

    return wrapped


BACKEND_TABLES = {
    "polars": impl_to_table_callable(polars_impl),
    "sqlite": impl_to_table_callable(sqlite_impl),
    "duckdb": impl_to_table_callable(duckdb_impl),
    "postgres": impl_to_table_callable(postgres_impl),
    "mssql": impl_to_table_callable(mssql_impl),
}
