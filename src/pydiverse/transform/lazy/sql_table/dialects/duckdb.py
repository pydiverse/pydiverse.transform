from __future__ import annotations

from pydiverse.transform.lazy.sql_table.dialects.postgres import PostgresTableImpl


class DuckDBTableImpl(PostgresTableImpl):
    _dialect_name = "duckdb"
