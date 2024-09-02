from __future__ import annotations

from pydiverse.transform.backend.sql_table import SQLTableImpl


class DuckDBTableImpl(SQLTableImpl):
    _dialect_name = "duckdb"
