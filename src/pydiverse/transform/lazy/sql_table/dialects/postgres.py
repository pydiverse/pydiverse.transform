from __future__ import annotations

from pydiverse.transform.lazy.sql_table.sql_table import SQLTableImpl


class PostgresTableImpl(SQLTableImpl):
    _dialect_name = "postgresql"
