from __future__ import annotations

import sqlalchemy as sa

from pydiverse.transform import ops
from pydiverse.transform.lazy.sql_table.sql_table import SQLTableImpl


class SQLiteTableImpl(SQLTableImpl):
    _dialect_name = "sqlite"


with SQLiteTableImpl.op(ops.Round()) as op:

    @op.auto
    def _round(x, decimals=0):
        if decimals >= 0:
            return sa.func.ROUND(x, decimals, type=x.type)
        # For some reason SQLite doesn't like negative decimals values
        return sa.func.ROUND(x / (10**-decimals), type=x.type) * (10**-decimals)
