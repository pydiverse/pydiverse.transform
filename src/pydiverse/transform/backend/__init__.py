from __future__ import annotations

from .duckdb import DuckDbImpl
from .mssql import MsSqlImpl
from .polars import PolarsImpl
from .postgres import PostgresImpl
from .sql import SqlImpl
from .sqlite import SqliteImpl
from .table_impl import TableImpl
from .targets import DuckDb, Polars, SqlAlchemy
