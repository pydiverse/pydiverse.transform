from __future__ import annotations

from pydiverse.transform.backend.sql import SqlImpl


class DuckDbImpl(SqlImpl):
    dialect_name = "duckdb"
