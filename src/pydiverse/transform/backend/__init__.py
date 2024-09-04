from __future__ import annotations

from .duckdb import DuckDbImpl
from .polars import PolarsImpl
from .sql import SqlImpl
from .table_impl import TableImpl
from .targets import DuckDb, Polars, SqlAlchemy
