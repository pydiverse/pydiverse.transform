# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import functools
import os
import tempfile
from pathlib import Path

import polars as pl
import sqlalchemy as sqa

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


def sql_table(
    df: pl.DataFrame,
    name: str,
    url: str,
    dtypes_map: dict | None = None,
    sql_dtypes: dict | None = None,
    fix_sql_dtypes: dict | None = None,
    dialect_infix="",
):
    import sqlalchemy as sqa

    global _sql_engine_cache

    dtypes_map = dtypes_map or {}
    dtypes_map[pl.Float64] = sqa.Double

    if url in _sql_engine_cache:
        engine = _sql_engine_cache[url]
    else:
        engine = sqa.create_engine(url)
        _sql_engine_cache[url] = engine

    sql_dtypes = sql_dtypes or {}
    for col, dtype in zip(df.columns, df.dtypes, strict=True):
        if dtype in dtypes_map and col not in sql_dtypes:
            sql_dtypes[col] = dtypes_map[dtype]

    df.write_database(name, engine, if_table_exists="replace", engine_options={"dtype": sql_dtypes})
    if fix_sql_dtypes is not None and len(fix_sql_dtypes) > 0:
        # this is a hack to fix sql types after creation of the table
        # the main reason for this is that ibm_db_sa renders sqa.boolean as SMALLINT
        # (https://github.com/ibmdb/python-ibmdbsa/issues/161)
        with engine.connect() as conn:
            for col, dtype in fix_sql_dtypes.items():
                conn.execute(sqa.text(f"ALTER TABLE {name} ALTER COLUMN {col} {dialect_infix} {dtype}"))
            conn.execute(sqa.text(f"call sysproc.admin_cmd('REORG TABLE {name}')"))
            conn.commit()
    return Table(name, SqlAlchemy(engine))


@_cached_table
def sqlite_table(df: pl.DataFrame, name: str):
    return sql_table(df, name, "sqlite:////tmp/transform/test.sqlite")


@_cached_table
def duckdb_table(df: pl.DataFrame, name: str):
    return sql_table(df, name, "duckdb:////tmp/transform/test.duckdb")


@_cached_table
def duckdb_parquet_table(df: pl.DataFrame, name: str):
    import sqlalchemy as sqa

    global _sql_engine_cache

    if "duckdb_parquet" in _sql_engine_cache:
        engine = _sql_engine_cache["duckdb_parquet"]
    else:
        file = "/tmp/transform/test_parquet.duckdb"
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        engine = sqa.create_engine("duckdb:///" + file)
    _sql_engine_cache["duckdb_parquet"] = engine
    path = Path(tempfile.gettempdir()) / "pdtransform" / "tests"
    file = path / f"{name}.parquet"
    os.makedirs(path, exist_ok=True)
    if file.exists():
        os.unlink(file)
    df.write_parquet(file)

    with engine.connect() as conn:
        conn.execute(sqa.text(f"DROP VIEW IF EXISTS {name}"))
        conn.execute(sqa.text(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path / f'{name}.parquet'}')"))
        conn.commit()

    return Table(name, SqlAlchemy(engine))


@_cached_table
def postgres_table(df: pl.DataFrame, name: str):
    url = "postgresql://sa:Pydiverse23@127.0.0.1:6543"

    return sql_table(df, name, url)


@_cached_table
def mssql_table(df: pl.DataFrame, name: str):
    from sqlalchemy.dialects.mssql import DATETIME2

    url = "mssql+pyodbc://sa:PydiQuant27@127.0.0.1:1433/master?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
    return sql_table(df, name, url, dtypes_map={pl.Datetime(): DATETIME2()})


@_cached_table
def ibm_db2_table(df: pl.DataFrame, name: str):
    url = "db2+ibm_db://db2inst1:password@localhost:50000/testdb"

    map = {}
    fix_map = {}
    for col, dtype in zip(df.columns, df.dtypes, strict=True):
        if dtype == pl.String:
            max_length = df[col].str.len_chars().max()
            if max_length > 32_672:
                map[col] = sqa.CLOB()
            elif max_length > 256:
                map[col] = sqa.VARCHAR(32_672)
            else:
                map[col] = sqa.VARCHAR(256)
        if dtype == pl.Boolean:
            fix_map[col] = sqa.Boolean()

    return sql_table(
        df,
        name,
        url,
        sql_dtypes=map,
        fix_sql_dtypes=fix_map,
        dialect_infix="SET DATA TYPE",
    )


BACKEND_TABLES = {
    "polars": polars_table,
    "duckdb": duckdb_table,
    "duckdb_parquet": duckdb_parquet_table,
    "sqlite": sqlite_table,
    "postgres": postgres_table,
    "mssql": mssql_table,
    "ibm_db2": ibm_db2_table,
}
