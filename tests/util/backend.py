from __future__ import annotations

import functools

import polars as pl

from pydiverse.transform._internal.backend.targets import SqlAlchemy
from pydiverse.transform._internal.pipe.table import Table


def _cached_table(fn):
    cache = {}

    @functools.wraps(fn)
    def wrapped(df: pl.DataFrame, name: str):
        if name in cache:
            return cache[name]

        impl = fn(df, name)
        cache[name] = impl

        return impl

    return wrapped


@_cached_table
def polars_table(df: pl.DataFrame, name: str):
    return Table(df, name=name)


_sql_engine_cache = {}


def sql_table(df: pl.DataFrame, name: str, url: str, dtypes_map: dict | None = None):
    import sqlalchemy as sqa

    global _sql_engine_cache

    dtypes_map = dtypes_map or {}
    dtypes_map[pl.Float64] = sqa.Double

    if url in _sql_engine_cache:
        engine = _sql_engine_cache[url]
    else:
        engine = sqa.create_engine(url)
        _sql_engine_cache[url] = engine

    sql_dtypes = {}
    for col, dtype in zip(df.columns, df.dtypes, strict=True):
        if dtype in dtypes_map:
            sql_dtypes[col] = dtypes_map[dtype]

    df.write_database(
        name, engine, if_table_exists="replace", engine_options={"dtype": sql_dtypes}
    )
    return Table(name, SqlAlchemy(engine))


@_cached_table
def sqlite_table(df: pl.DataFrame, name: str):
    return sql_table(df, name, "sqlite:///:memory:")


@_cached_table
def duckdb_table(df: pl.DataFrame, name: str):
    return sql_table(df, name, "duckdb:///:memory:")


@_cached_table
def postgres_table(df: pl.DataFrame, name: str):
    url = "postgresql://sa:Pydiverse23@127.0.0.1:6543"

    return sql_table(df, name, url)


@_cached_table
def mssql_table(df: pl.DataFrame, name: str):
    from sqlalchemy.dialects.mssql import DATETIME2

    url = (
        "mssql+pyodbc://sa:PydiQuant27@127.0.0.1:1433"
        "/master?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
    )
    return sql_table(df, name, url, dtypes_map={pl.Datetime(): DATETIME2()})


BACKEND_TABLES = {
    "polars": polars_table,
    "sqlite": sqlite_table,
    "duckdb": duckdb_table,
    "postgres": postgres_table,
    "mssql": mssql_table,
}
