from __future__ import annotations

from pydiverse.transform.lazy.sql_table.sql_table import SQLTableImpl


class MssqlTableImpl(SQLTableImpl):
    _dialect_name = "mssql"

    def post_process_order_by(self, col, o_by):
        col = col.asc() if o_by.asc else col.desc()
        # NULLS is not supported by TSQL, yet
        # col = col.nullsfirst() if o_by.nulls_first else col.nullslast()
        if o_by.asc and not o_by.nulls_first:
            raise NotImplementedError(
                "NULLS LAST is not supported by TSQL on ascending order"
            )
        if not o_by.asc and o_by.nulls_first:
            raise NotImplementedError(
                "NULLS FIRST is not supported by TSQL on descending order"
            )
        return col
